import numpy as np
import tqdm


def get_min_distance(x, clusters, dist="l2"):
    """Return minimal Euclidian distance and index of closest vector of vector x to list of vectors clusters.

    Args:
        x (np.ndarray): shape = (n_features, ), vector
        clusters (np.ndarray): (n_vectors, n_features), list (cluster) of vectors

    Returns:
        (np.ndarray, np.ndarray): minimal distance from x to elements of clusters and index of closest point
    """
    if dist == "l2":
        dist = (clusters - x) ** 2
    elif dist == "l1":
        dist = np.abs(clusters - x)
    else:
        raise ValueError(f"Unknown distance function {dist}")
    dist = np.sum(dist, axis=1)
    return np.min(dist), np.argmin(dist)


def kmeans_pp_init(data_points, num_clusters=2, dist="l2"):
    """K-Means ++ initalization.

    Args:
        data_points (np.ndarray): shape=(num_data, num_features)
        num_clusters (int, optional): _description_. Defaults to 2.
        dist (str, optional): _description_. Defaults to "l2".

    Returns:
        _type_: _description_
    """
    # FIRST RANDOM CHOICE
    init_point = data_points[np.random.randint(0, len(data_points) - 1)]
    selected_points = np.array([init_point])
    for i in range(num_clusters - 1):
        # Proba as distance between points and selected points
        Ps = np.array([get_min_distance(x, selected_points)[0] for x in data_points])
        Ps = Ps / np.sum(Ps)

        # Random choice with probabilities
        next_C = np.random.choice(np.arange(0, len(data_points)), p=Ps)
        next_C = data_points[next_C]

        selected_points = np.array(selected_points.tolist() + [next_C])
    return selected_points


class PLSHeuristic(object):
    """
    Main class for or PLS Heuristic."""

    def __init__(
        self,
        init="kmeans++",
        n_clusters=2,
        models_class=None,
        models_params={"n_pieces": 5},
        use_proba=False,
        n_init=1,
        max_iter_by_init=100,
        stopping_criterion=None,
    ):
        self.init = init
        if models_class is None:
            raise ValueError("You need to specify a model class to be estimated.")
        self.models_class = models_class
        self.models_params = models_params
        self.n_clusters = n_clusters
        self.use_proba = use_proba
        self.n_init = n_init
        self.max_iter_by_init = max_iter_by_init
        self.stopping_criterion = stopping_criterion

    def _assignment(self, X, Y, group_ids=None):
        """Assignment step of the heuristic.

        Parameters
        ----------
        X : _type_
            _description_
        Y : _type_
            _description_
        group_ids : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        utilities = []
        for model in self.models:
            utilities.append(model.predict_utility(X) - model.predict_utility(Y))
        if len(utilities[0].shape) == 0:
            utilities = [[u] for u in utilities]
        utilities = np.stack(utilities, axis=1)
        utilities = np.squeeze(utilities)
        if len(utilities.shape) == 1:
            utilities = utilities.reshape(-1, 1)

        clusters = np.argmax(utilities, axis=1)

        if group_ids is not None:
            clusters = []
            for unique_id in np.unique(group_ids):
                bool_index = group_ids == unique_id
                max_indexes = np.argmax(utilities[bool_index], axis=1)

                count = np.bincount(max_indexes)
                if np.sum(count == np.max(count)) > 1:
                    possible_cluster_indexes = np.where(count == np.max(count))[0]
                    diff_u = np.sum(utilities[bool_index], axis=0)
                    cluster = possible_cluster_indexes[
                        np.argmax(diff_u[possible_cluster_indexes])
                    ]
                    clusters += [cluster] * len(max_indexes)
                else:
                    clusters += [np.argmax(count)] * len(max_indexes)

        return utilities, clusters

    def _update(self, X, Y, verbose=0):
        """Update step of the heuristic for a single cluster.

        Parameters
        ----------
        X : _type_
            _description_
        Y : _type_
            _description_
        sample_weight : _type_, optional
            _description_, by default None
        verbose : int, optional
            _description_, by default 0

        Returns
        -------
        _type_
            _description_
        """
        model = self.models_class(**self.models_params)
        history = model.fit(X, Y, verbose=verbose)
        return model, history

    def fit(self, X, Y, group_ids=None, plateau_stop=True, verbose=0):
        """Fit the model.

        Parameters
        ----------
        X : _type_
            _description_
        Y : _type_
            _description_
        group_ids : _type_, optional
            _description_, by default None
        plateau_stop : bool, optional
            _description_, by default True
        verbose : int, optional
            _description_, by default 0

        Returns
        -------
        _type_
            _description_
        """
        losses_by_init = []
        best_losses_history = []

        if plateau_stop:
            stopping_criterion = "partitioning"
        else:
            stopping_criterion = None

        # fitting as many times as n_init
        for init_nb in range(self.n_init):
            if verbose > 0:
                print(f"Initialization {init_nb + 1} / {self.n_init}")
            init_losses, init_coeffs = self._single_fit(
                X=X,
                Y=Y,
                group_ids=group_ids,
                max_iter=self.max_iter_by_init,
                stopping_criterion=stopping_criterion,
                verbose=verbose,
            )
            losses_by_init.append(init_losses)
            # Keeping best fit
            if (
                np.argmin(np.sum(np.array([l[-1] for l in losses_by_init]), axis=1))
                == init_nb
            ):
                self._model_coeffs = init_coeffs
                best_losses_history = init_losses

            if (
                np.sum(init_losses[-1]) == 0
            ):  # useless to try new initializations, best fit reached
                if verbose > 0:
                    print("Loss 0 reached, stopping fit.")
                break
            else:
                print("Loss not 0, continuing fit.", np.sum(init_losses[-1]))
        return losses_by_init, np.array(best_losses_history)

    def _single_fit(
        self, X, Y, group_ids=None, max_iter=100, stopping_criterion=None, verbose=0
    ):
        """Estimate the models. - can be called several times with n_init.

        Parameters
        ----------
        X : _type_
            _description_
        Y : _type_
            _description_
        group_ids : _type_, optional
            _description_, by default None
        max_iter : int, optional
            _description_, by default 100
        stopping_criterion : _type_, optional
            _description_, by default None
        verbose : int, optional
            _description_, by default 0

        Returns
        -------
        _type_
            _description_
        """
        all_losses = [[] for _ in range(self.n_clusters)]
        indexes = []

        # Initialization
        if self.init == "kmeans++":
            cluster_centers = kmeans_pp_init(X - Y, num_clusters=self.n_clusters)
            indexes = np.array(
                [get_min_distance(X[i] - Y[i], clusters=cluster_centers)[1] for i in range(len(X))]
            )

        else:
            raise ValueError(f"Unknwon init method: {self.init}")
        all_losses = []
        if verbose > 0:
            t_range = tqdm.trange(self.max_iter_by_init)
            t_range.set_description(f"Loss = ", refresh=True)
        else:
            t_range = range(self.max_iter_by_init)

        for i in t_range:
            # for stuck optim detection

            ### New IMPLEM:

            # Initialization
            if i == 0:
                # Expectation
                if self.init == "kmeans++":
                    cluster_centers = kmeans_pp_init(
                        X - Y, num_clusters=self.n_clusters
                    )
                    indexes = np.array(
                        [
                            get_min_distance(X[i] - Y[i], clusters=cluster_centers)[1]
                            for i in range(len(X))
                        ]
                    )

                else:
                    indexes = np.random.randint(0, self.n_clusters, size=len(X))

                # Maximization
                models = []
                for j in range(self.n_clusters):
                    bool_index = indexes == j
                    model, history = self._update(
                        X[bool_index], Y[bool_index], verbose=verbose
                    )
                    models.append(model)
                self.models = models
            else:
                # Expectation
                utilities, new_indexes = self._assignment(X, Y)

                if self.stopping_criterion == "partitioning":
                    if (new_indexes == indexes).all():
                        if verbose > 0:
                            print(
                                "Ending fit before num_epochs reached as partitioning has reached a stable state"
                            )
                        break
                indexes = new_indexes

                loop_cluster_losses = []
                models = []

                for j in range(self.n_clusters):
                    bool_index = indexes == j
                    model, history = self._update(
                        X[bool_index],
                        Y[bool_index],
                        verbose=verbose,
                    )
                    models.append(model)

                    loop_cluster_losses.append(np.sum(list(history[0].values())))
                all_losses.append(loop_cluster_losses)


                if np.sum(loop_cluster_losses) < 1e-5:
                    model_coeffs = np.stack([m.coeffs for m in self.models], axis=0)
                    if verbose > 0:
                        print("Ending fit before num_epochs reached as loss is 0")
                    break
                if verbose > 0:
                    t_range.set_description(
                        f"Loss= {np.round(np.sum(list(history[0].values())), 4)}"
                    )
                model_coeffs = np.stack([m.coeffs for m in self.models], axis=0)
                if stopping_criterion == "stalestate":
                    similar_coefficients = []
                    for m1, m2 in zip(models, self.models):
                        if np.sum(m1.coeffs - m2.coeffs) < 1e-4:
                            similar_coefficients.append(True)
                    if np.all(similar_coefficients):
                        if verbose > 0:
                            print(
                                "Ending fit before num_epochs reached as stalestate has been reached"
                            )
                        break

                self.models = models
        return all_losses, model_coeffs

    def predict_marginal_utilities(self, X):
        print("Function not implemented yet")
        raise NotImplementedError

    def predict_utility(self, X):
        # Should only take one input ???
        utilities = []
        for model in self.models:
            utilities.append(model.predict_utility(X))
        utilities = np.stack(utilities, axis=1)
        utilities = np.squeeze(utilities)
        return utilities

    def predict_cluster(self, X, Y):
        utilities, clusters, _ = self._assignment(X, Y)
        return clusters

    @property
    def coeffs(self):
        return np.stack([md.coeffs for md in self.models])