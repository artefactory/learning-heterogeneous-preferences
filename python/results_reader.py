"""Classes to read saved files from experiments."""
import itertools
import os

import numpy as np
import pandas as pd
import preference_learning.metrics as metrics
import tqdm
from preference_learning.utils.functions import piecewise_linear_f


class PLResultsReader(object):
    filepaths = {
        "data": "data.csv",
        "params": "params.txt",
        "u_train_x_pred": "u_train_x_pred.csv",
        "u_test_x_pred": "u_test_x_pred.csv",
        "u_train_y_pred": "u_train_y_pred.csv",
        "u_test_y_pred": "u_test_y_pred.csv",
    }

    def __init__(self, dir_path, model_name):
        self.dir_path = dir_path
        self.model_name = model_name

    def read_params(self):
        params_values = {}
        for txt in self.params:
            key, value = txt.split("=")
            try:
                params_values[key] = int(value)
            except ValueError:
                params_values[key] = value
        return params_values

    def read(self):
        self.data = pd.read_csv(os.path.join(self.dir_path, self.filepaths["data"]))

        self.u_train_x = pd.read_csv(
            os.path.join(self.dir_path, self.model_name + "_" + self.filepaths["u_train_x_pred"])
        )
        self.u_train_y = pd.read_csv(
            os.path.join(self.dir_path, self.model_name + "_" + self.filepaths["u_train_y_pred"])
        )
        self.u_test_x = pd.read_csv(
            os.path.join(self.dir_path, self.model_name + "_" + self.filepaths["u_test_x_pred"])
        )
        self.u_test_y = pd.read_csv(
            os.path.join(self.dir_path, self.model_name + "_" + self.filepaths["u_test_y_pred"])
        )

        ux_columns = [col for col in self.data.columns if col.startswith("ux_")]
        uy_columns = [col for col in self.data.columns if col.startswith("uy_")]
        self.gt_ux = self.data[ux_columns].values
        self.gt_uy = self.data[uy_columns].values

        self.data_weights = np.load(os.path.join(self.dir_path, "data_weights.npy"))
        marginal_weights = np.load(os.path.join(self.dir_path, "marginal_data_weights.npy"))
        self.data_weights = self.data_weights * np.expand_dims(marginal_weights, axis=-1)
        self.pred_weights = np.load(
            os.path.join(self.dir_path, self.model_name + "_" + "pred_weights.npy")
        )
        self.fit_time = np.load(os.path.join(self.dir_path, self.model_name + "_" + "fit_time.npy"))
        self.fit_status = np.loadtxt(
            os.path.join(self.dir_path, f"{self.model_name}_status.txt"), dtype=str
        )
        self.params = np.loadtxt(os.path.join(self.dir_path, f"params.txt"), dtype=str)
        self.params = self.read_params()

        self.results_metrics = {}

    def compute_metrics(self):
        lss = self.params["learning_set_size"]
        pe = metrics.PairsExplained()
        ci = metrics.ClusterIntersection()
        cp = metrics.MonteCarloCommonPreferences(n_features=self.params["n_features"])
        new_adjusted_rand_index = metrics.RandIndex(adjusted=True)
        new_rand_index = metrics.RandIndex(adjusted=False)
        self.results_metrics["explained_pairs_test"] = pe(
            Ux=self.u_test_x.values, Uy=self.u_test_y.values
        )
        self.results_metrics["explained_pairs_train"] = pe(
            Ux=self.u_train_x.values, Uy=self.u_train_y.values
        )
        self.results_metrics["rand_index_test"] = ci(
            y_pred=np.argmax(self.u_test_x.values - self.u_test_y.values, axis=1),
            y_true=self.data.cluster.values[lss:],
        )
        self.results_metrics["rand_index_train"] = ci(
            y_pred=np.argmax(self.u_train_x.values - self.u_train_y.values, axis=1),
            y_true=self.data.cluster.values[:lss],
        )
        self.results_metrics["new_rand_index_test"] = new_rand_index(
            y_pred=np.argmax(self.u_test_x.values - self.u_test_y.values, axis=1),
            y_true=self.data.cluster.values[lss:],
        )
        self.results_metrics["new_rand_index_train"] = new_rand_index(
            y_pred=np.argmax(self.u_train_x.values - self.u_train_y.values, axis=1),
            y_true=self.data.cluster.values[:lss],
        )
        self.results_metrics["new_adjusted_rand_index_test"] = new_adjusted_rand_index(
            y_pred=np.argmax(self.u_test_x.values - self.u_test_y.values, axis=1),
            y_true=self.data.cluster.values[lss:],
        )
        self.results_metrics["new_adjusted_rand_index_train"] = new_adjusted_rand_index(
            y_pred=np.argmax(self.u_train_x.values - self.u_train_y.values, axis=1),
            y_true=self.data.cluster.values[:lss],
        )
        self.results_metrics["max_adjusted_rand_index_test"] = new_adjusted_rand_index(
            y_pred=np.argmax(self.u_test_x.values - self.u_test_y.values, axis=1),
            y_true=np.argmax(self.gt_ux[lss:] - self.gt_uy[lss:], axis=1),
        )
        self.results_metrics["max_adjusted_rand_index_train"] = new_adjusted_rand_index(
            y_pred=np.argmax(self.u_train_x.values - self.u_train_y.values, axis=1),
            y_true=np.argmax(self.gt_ux[:lss] - self.gt_uy[:lss], axis=1),
        )

        gt_distance = []
        for i in range(self.params["n_decisions_makers"] - 1):
            for j in range(i + 1, self.params["n_decisions_makers"]):
                ux = piecewise_linear_f(self.data_weights[i])
                uy = piecewise_linear_f(self.data_weights[j])
                gt_distance.append(cp.from_utility_functions(ux, uy))
        self.results_metrics["average_truth_similarity"] = np.mean(gt_distance)
        self.results_metrics["min_truth_similarity"] = np.min(gt_distance)
        self.results_metrics["max_truth_similarity"] = np.max(gt_distance)
        self.results_metrics["median_truth_similarity"] = np.median(gt_distance)
        self.results_metrics["std_truth_similarity"] = np.std(gt_distance)

        truth_predict_distances = []
        for i in range(self.data_weights.shape[0]):
            data_row = []
            for j in range(self.pred_weights.shape[0]):
                ux = piecewise_linear_f(self.data_weights[i])
                uy = piecewise_linear_f(self.pred_weights[j])
                data_row.append(cp.from_utility_functions(ux, uy))
            truth_predict_distances.append(data_row)

        truth_predict_scores = []
        for var in itertools.permutations(
            list(range(self.params["n_decisions_makers"])), self.u_test_x.values.shape[1]
        ):
            score = []
            for i, j in enumerate(var):
                score.append(truth_predict_distances[i][j])
            truth_predict_scores.append(np.mean(score))
        self.results_metrics["average_truth_predict_similarity"] = np.max(truth_predict_scores)

        self.results_metrics["fit_time"] = self.fit_time
        self.results_metrics["fit_status"] = self.fit_status

class HeuristicReader(object):
    """"""
    def __init__(self):
        pass


class ExperimentsReader(object):
    """Reads several PLSResultsReader."""

    def __init__(self, base_dir, model_name, force_read=False, reader=PLResultsReader):
        self.base_dir = base_dir
        self.reader = reader
        self.model_name = model_name
        self.experiments = os.listdir(base_dir)

        self.force_read = force_read

    def read(self):
        self.results = {}
        for exp in tqdm.tqdm(self.experiments):
            try:
                res = self.reader(os.path.join(self.base_dir, exp), model_name=self.model_name)
                res.read()
                self.results[exp] = res
            except Exception as e:
                print("Couldn't read", exp)
                print(e)
        self.results_metrics = self.compute_metrics()
        self.report = self.results_to_df()

    def compute_metrics(self):
        if (
            f"{self.model_name}_results_metrics.csv" in os.listdir(self.base_dir)
            and not self.force_read
        ):
            results_metrics = pd.read_csv(
                os.path.join(self.base_dir, f"{self.model_name}_results_metrics.csv")
            )
            results_metrics = results_metrics.to_dict(orient="list")

        else:
            results_metrics = {"id": []}
        for exp in tqdm.tqdm(self.results.keys()):
            if exp not in results_metrics["id"]:
                exp_metrics = self.compute_metrics_from_id(exp)
                results_metrics = self.update_list_dict(results_metrics, exp_metrics)

        return results_metrics

    def update_list_dict(self, main_dict, compl_dict):
        for key, value in compl_dict.items():
            if key not in main_dict:
                print("Adding key:", key)
                main_dict[key] = value
            else:
                main_dict[key] = main_dict[key] + value

        return main_dict

    def compute_metrics_from_id(self, id):
        metrics_dict = {"id": [id]}

        reader = self.results[id]
        reader.compute_metrics()
        for metric, value in reader.results_metrics.items():
            metrics_dict[metric] = [value]
        for param, value in reader.params.items():
            metrics_dict[param] = [value]

        return metrics_dict

    def results_to_df(self):
        if not hasattr(self, "results_metrics"):
            self.compute_metrics()

        df = pd.DataFrame(self.results_metrics)
        df.to_csv(
            os.path.join(self.base_dir, f"{self.model_name}_results_metrics.csv"), index=False
        )
        return df
