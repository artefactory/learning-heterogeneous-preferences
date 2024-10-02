from abc import abstractmethod

import numpy as np
from scipy.special import comb
from sklearn.metrics import adjusted_rand_score, rand_score


class BaseMetric:
    """Base Class for metrics. Should be inherited by all metrics"""

    def __init__(self):
        """Initialization of the metric, can be used to setup different behaviours of the metric"""
        super().__init__()

    @abstractmethod
    def __call__(self, X, Y):
        """Method where the metric should be defined

        Parameters
        ----------
        X: np.ndarray
            Role can change depending on the metric
        Y: np.ndarray
            Role can change depending on the metric
        """
        return


class PairsExplained(BaseMetric):
    """
    Computes the percentage of pairs that are explained by at least a cluster
    """

    def __init__(self):
        super().__init__()

    def __call__(self, Ux, Uy):
        """main function to call the PairsExplained metric

        Parameters
        ----------
        Ux (np.ndarray of shape (n_elements, n_clusters)):
            utilities of preferred elements for each cluster
        Uy (np.ndarray of shape (n_elements, n_clusters)):
            utilities of non preferred elements for each cluster

        Ux and Uy are organised such that \forall i, x_i > y_i and we compare Ux[i] with Uy[i] for each cluster

        Returns
        -------
        float
            percentage of pairs explained by at least a cluster
        """
        assert Ux.shape == Uy.shape
        if len(Ux.shape) == 1:
            Ux = np.expand_dims(Ux, axis=-1)
            Uy = np.expand_dims(Uy, axis=-1)
        return np.sum(np.sum(Ux - Uy > 0, axis=1) > 0) / len(Ux)

    def from_model(self, model, X, Y):
        """Method to use the metric from a model and data.

        Parameters
        ----------
        model : BaseModel
            Model to be evaluated
        X : np.ndarray
            data to be evaluated: features of preferred elements
        Y : np.ndarray
            data to be evaluated: features of unchosen elements

        Returns
        -------
        float
            percentage of pairs explained by at least a cluster
        """
        Ux = model.predict_utility(X)
        Uy = model.predict_utility(Y)
        return self(Ux, Uy)


class MeanExplainability(BaseMetric):
    """
    Computes the average number of cluster explaining a pair
    """

    def __init__(self):
        super().__init__()

    def __call__(self, Ux, Uy):
        """main function to call the PairsExplained metric

        Parameters
        ----------
        Ux (np.ndarray of shape (n_elements, n_clusters)):
            utilities of preferred elements for each cluster
        Uy (np.ndarray of shape (n_elements, n_clusters)):
            utilities of non preferred elements for each cluster

        Ux and Uy are organised such that \forall i, x_i >g y_i and we compare Ux[i] with Uy[i] for each cluster

        Returns
        -------
        float
            percentage of pairs explained by at least a cluster
        """
        assert Ux.shape == Uy.shape
        if len(Ux.shape) == 1:
            Ux = np.expand_dims(Ux, axis=-1)
            Uy = np.expand_dims(Uy, axis=-1)
        return np.sum(Ux - Uy > 0, axis=1) / len(Ux)

    def from_model(self, model, X, Y):
        """Method to use the metric from a model and data.

        Parameters
        ----------
        model : BaseModel
            Model to be evaluated
        X : np.ndarray
            data to be evaluated: features of preferred elements
        Y : np.ndarray
            data to be evaluated: features of unchosen elements

        Returns
        -------
        float
            percentage of pairs explained by at least a cluster
        """
        Ux = model.predict_utility(X)
        Uy = model.predict_utility(Y)
        return self(Ux, Uy)
