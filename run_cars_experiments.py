import argparse
import os
import numpy as np
import time


from sklearn.model_selection import GroupShuffleSplit

from python.real_data import load_cars_preferences_pairs
from python.models import ClusterUTA, UTA
from python.heuristics import PLSHeuristic


def run_xp(
    base_dir,
    run_id,
    timeout,
    seed,
    test_size=0.46,
    train_sizes=[100, 150, 200, 400, 600, 1000, 2500, 10000],
    clusters=[2, 3, 4, 5],
    epsilon=0.05,
    n_pieces=5,
):
    results_dir = os.path.join(base_dir, f"results/{run_id}")
    X, Y, choice_ids = load_cars_preferences_pairs()

    gss = GroupShuffleSplit(n_splits=1, train_size=1-test_size, random_state=seed)
    for i, (train_index, test_index) in enumerate(gss.split(X, Y, choice_ids)):
        X_train = X[train_index]
        X_test = X[test_index]
        Y_train = Y[train_index]
        Y_test = Y[test_index]
        choice_ids_train = choice_ids[train_index]
        choice_ids_test = choice_ids[test_index]

    np.save(os.path.join(results_dir, "X_train.npy"), X_train)
    np.save(os.path.join(results_dir, "Y_train.npy"), Y_train)
    np.save(os.path.join(results_dir, "X_test.npy"), X_test)
    np.save(os.path.join(results_dir, "Y_test.npy"), Y_test)
    np.save(os.path.join(results_dir, "ids_train.npy"), choice_ids_train)
    np.save(os.path.join(results_dir, "ids_test.npy"), choice_ids_test)
    
    for ds in train_sizes:
        for cluster in clusters:
            t_start = time.time()

            milo_model = ClusterUTA(n_clusters=cluster, n_pieces=n_pieces, epsilon=epsilon)
            hist = milo_model.fit(
                X_train[:ds],
                Y_train[:ds],
                cluster_grouping=choice_ids_train[:ds],
                time_limit=timeout,
                n_threads=12,
            )
            t_end = time.time()
            np.save(os.path.join(base_dir, f"milo_{cluster}_clusters_{ds}.npy"), milo_model.coeffs)
            np.save(os.path.join(base_dir, f"{cluster}_clusters_{ds}_milo_fit_time.npy"), np.array(t_end - t_start))
            np.save(
                os.path.join(base_dir, f"{cluster}_clusters_{ds}_milo_fit_status.npy"), np.array(milo_model.status)
            )

            heuristic = PLSHeuristic(
                models_class=UTA, n_clusters=n_clusters, n_init=20
            )
            t_start = time.time()
            hist = heuristic.fit(X_train, Y_train)
            t_end = time.time()
            np.save(os.path.join(base_dir, f"heuristic_{cluster}_clusters_{ds}.npy"), heuristic.coeffs)
            np.save(os.path.join(base_dir, f"{cluster}_clusters_{ds}_heuristic_fit_time.npy"), np.array(t_end - t_start))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("save_dir", type=str, help="Directory to save results.")

    parser.add_argument(
        "-r",
        "--repetitions",
        default=1,
        type=int,
        help="Number of experiments for each combination of parameters.",
    )
    parser.add_argument(
        "-to", "--timeout", default=1800, type=int, help="TimeOut for the solver."
    )
    parser.add_argument(
        "-tss",
        "--test_set_size",
        default=0.46,
        type=float,
        help="Number of samples in the testing set.",
    )

    parser.add_argument(
        "-cl",
        "--n_clusters",
        type=int,
        nargs="+",
        default=2,
        help="Number of clusters considered in data generation and modeling - can be int or list.",
    )
    parser.add_argument(
        "-p",
        "--n_pieces",
        type=int,
        nargs="+",
        default=5,
        help="Number of pieces for the UTA models - can be int or list.",
    )
    parser.add_argument(
        "-lss",
        "--learning_set_size",
        type=int,
        nargs="+",
        default=2**10,
        help="Learning set size  - can be int or list.",
    )

    parser.add_argument(
        "-e",
        "--epsilon",
        type=float,
        default=0.05,
        help="Magin of utility between preferences.",
    )

    args = parser.parse_args()

    base_dir = args.save_dir
    repetitions = args.repetitions
    timeout = args.timeout
    test_set_size = args.test_set_size
    epsilon = args.epsilon

    n_clusters = args.n_clusters
    if isinstance(n_clusters, int):
        n_clusters = [n_clusters]
    if not isinstance(n_clusters, list):
        raise ValueError(
            f"n_clusters should be int or list of int and is: {type(n_clusters)}"
        )

    n_pieces = args.n_pieces
    if isinstance(n_pieces, int):
        n_pieces = [n_pieces]
    if not isinstance(n_pieces, list):
        raise ValueError(
            f"n_pieces should be int or list of int and is: {type(n_pieces)}"
        )


    train_set_size = args.learning_set_size
    if isinstance(train_set_size, int):
        train_set_size = [train_set_size]
    if not isinstance(train_set_size, list):
        raise ValueError(
            f"train_set_size should be int or list of int and is: {type(train_set_size)}"
        )
    
    for seed in np.random.randint(low=0, size=(repetitions,)):
        for n_p in n_pieces:
            for lss in train_set_size:
                run_id = f"{lss}_{n_p}_{seed}"
                if os.path.exists(
                    os.path.join(base_dir, f"results/{run_id}")
                ):
                    print(f"Skipping {run_id}")
                else:
                    run_xp(
                        base_dir=base_dir,
                        run_id=run_id,
                        timeout=timeout,
                        seed=seed,
                        test_size=test_set_size,
                        train_sizes=train_set_size,
                        clusters=n_clusters,
                        epsilon=epsilon,
                        n_pieces=n_p,
                    )