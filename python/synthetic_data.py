import numpy as np


def get_random_uniform_normalized_vector(num_values, norm_value=1, decimals=6):
    """
    Functions that returns a random and normalized vector W
    W = [wi] such that wi >= 0, sum(wi) = norm_values, i = [1, num_values]
    """
    initial_vals = np.round(np.random.uniform(0, norm_value, num_values - 1), decimals=decimals)
    initial_vals = np.sort(initial_vals)
    vect = [norm_value - initial_vals[-1]]
    for i in range(len(initial_vals) - 1):
        vect.append(initial_vals[-i - 1] - initial_vals[-i - 2])
    vect.append(initial_vals[0])
    return np.array(vect)


def piecewise_linear_value(x, coeffs, min_x=0, max_x=1):
    """
    Returns the value of the piecewise linear function defined by coeffs at x
    """
    assert x >= min_x and x <= max_x
    interval = (max_x - min_x) / (len(coeffs) - 1)
    for i in range(len(coeffs) - 1):
        if x >= min_x + i * interval and x <= min_x + (i + 1) * interval:
            return coeffs[i] + (coeffs[i + 1] - coeffs[i]) / interval * (x - min_x - i * interval)


def create_piecewise_linear_marginal_utility(n_pieces, min_x=0.0, max_x=1.0):
    coefficients = np.round(np.random.uniform(min_x, max_x, n_pieces - 1), decimals=2)
    coefficients = np.sort(coefficients)
    coefficients = np.concatenate([[min_x], coefficients, [max_x]])

    @np.vectorize
    def piecewise_linear_f(x):
        return piecewise_linear_value(x=x, coeffs=coefficients, min_x=min_x, max_x=max_x)

    return piecewise_linear_f, coefficients

class SyntheticDataGenerator:
    def __init__(
        self,
        n_clusters,
        n_criteria,
        mix_decisions=False,
        method_params={},
        noise=0.0,
        gap=0.0,
        decimals=6,
    ):
        self.n_clusters = n_clusters
        self.n_criteria = n_criteria
        self.mix_decisions = mix_decisions
        self.method_params = method_params
        self.noise = noise  # % of noise = % of pairs that will be reversed
        self.gap = gap
        self.decimals = decimals

        self.instantiate()

    def instantiate(self):
        self.generated_parameters = {}
        dm_utilities = []
        for i in range(self.n_clusters):
            uf, params = self._generate_marginal_utilites_functions()
            dm_utilities.append(uf)
            self.generated_parameters[f"DM_{i}"] = params

        self._marginal_utility = lambda x: np.array([u(x) for u in dm_utilities])
        self._utility = lambda x: np.array([np.sum(u(x)) for u in dm_utilities])

    def _generate_marginal_utilites_functions(self):
        n_pieces = self.method_params.get("n_pieces", 5)
        funcs = []
        weights = []
        for _ in range(self.n_criteria):
            func, ws = create_piecewise_linear_marginal_utility(n_pieces=n_pieces)
            funcs.append(func)
            weights.append(ws)
        params = {"created_functions": funcs, "weights": weights}

        marginal_weights = get_random_uniform_normalized_vector(
            num_values=self.n_criteria, decimals=self.decimals
        )
        params["marginal_weights"] = marginal_weights

        # @np.vectorize
        def _utility(X):
            return np.array([w * f(x) for f, x, w in zip(funcs, X, marginal_weights)])

        params["utility"] = lambda x: np.array([np.sum(_utility(xi)) for xi in np.atleast_2d(x)])

        return _utility, params

    def utility(self, X):
        if len(X.shape) == 1:
            return self._utility(X)
        elif len(X.shape) == 2:
            return np.array([self._utility(x) for x in X])
        else:
            raise ValueError("Unsupported shape of X", X.shape)

    def marginal_utilities(self, X):
        if len(X.shape) == 1:
            return self._marginal_utility(X)
        elif len(X.shape) == 2:
            return np.array([self._marginal_utility(x) for x in X])
        else:
            raise ValueError("Unsupported shape of X", X.shape)

    def generate_data(self, num_pairs, return_utilities=False, return_clusters=False, verbose=0):
        X, Y = [], []

        utilities = [[], []]
        clusters = []
        # Useless now that we have clusters
        populations = [0] * self.n_clusters
        if not isinstance(num_pairs, list):
            num_pairs = [np.ceil(num_pairs / self.n_clusters)] * self.n_clusters
        while len(X) < sum(num_pairs):
            if verbose > 0:
                print(f"{len(X)} events have been created as of now", end="\r")
            x = np.around(np.random.uniform(0, 1, self.n_criteria), decimals=self.decimals)
            y = np.around(np.random.uniform(0, 1, self.n_criteria), decimals=self.decimals)

            ux = np.around(self.utility(x), decimals=self.decimals)
            uy = np.around(self.utility(y), decimals=self.decimals)

            if (ux - uy)[np.argmax(ux - uy)] > self.gap:
                if np.sum(ux > uy) == 1 and not self.mix_decisions:
                    if populations[np.argmax(ux > uy)] < num_pairs[np.argmax(ux > uy)]:
                        if np.random.randint(1000) / 1000 >= self.noise:
                            X.append(x)
                            Y.append(y)
                            utilities[0].append(ux)
                            utilities[1].append(uy)
                        else:
                            X.append(y)
                            Y.append(x)
                            utilities[0].append(uy)
                            utilities[1].append(ux)

                        populations[np.argmax(ux - uy)] += 1
                        clusters.append(np.argmax(ux - uy))

                elif np.sum(ux > uy) >= 1 and self.mix_decisions:
                    if populations[np.argmax(ux - uy)] < num_pairs[np.argmax(ux - uy)]:
                        if np.random.randint(1000) / 1000 >= self.noise:
                            X.append(x)
                            Y.append(y)
                            utilities[0].append(ux)
                            utilities[1].append(uy)
                        else:
                            X.append(y)
                            Y.append(x)
                            utilities[0].append(uy)
                            utilities[1].append(ux)

                        populations[np.argmax(ux - uy)] += 1
                        clusters.append(np.argmax(ux - uy))
        if verbose > 0:
            print("Clusters Populations", populations)
        additional_info = {}
        if "weights" in self.generated_parameters[f"DM_0"]:
            for i in range(self.n_clusters):
                additional_info[f"weights_{i}"] = self.generated_parameters[f"DM_{i}"]["weights"]
                additional_info[f"marginal_weights_{i}"] = self.generated_parameters[f"DM_{i}"][
                    "marginal_weights"
                ]
        if return_utilities:
            additional_info["utilities_x"] = np.array(utilities)[0]
            additional_info["utilities_y"] = np.array(utilities)[1]
        if return_clusters:
            additional_info["clusters"] = np.array(clusters)
        return np.stack(X), np.stack(Y), additional_info
