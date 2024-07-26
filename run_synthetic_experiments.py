import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from python.synthetic_data import SyntheticDataGenerator
from python.models import ClusterUTA, UTA
from python.heuristics import PLSHeuristic


def evaluation_routine(
    base_dir,
    run_id,
    n_clusters,
    n_criteria,
    n_linear_pieces,
    data_error,
    learning_set_size,
    test_set_size,
    time_limit,
):
    results_dir = os.path.join(base_dir, f"results/{run_id}")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "params.json"), "w") as f:
        json.dump(
            {
                "n_clusters": n_clusters,
                "n_criteria": n_criteria,
                "n_linera_pieces": n_linear_pieces,
                "error": data_error,
                "learning_set_size": learning_set_size,
            },
            f,
        )

    # Draw data
    datagen = SyntheticDataGenerator(
        n_clusters=n_clusters,
        n_criteria=n_criteria,
        mix_decisions=True,
        gap=0.0,
        noise=data_error,
    )

    X, Y, data_metadata = datagen.generate_data(
        test_set_size + learning_set_size, return_clusters=True, return_utilities=True
    )

    df_X = pd.DataFrame(X, columns=[f"x_{i}" for i in range(n_criteria)])
    df_Y = pd.DataFrame(Y, columns=[f"y_{i}" for i in range(n_criteria)])
    df_ux = pd.DataFrame(
        data_metadata["utilities_x"], columns=[f"ux_{i}" for i in range(n_clusters)]
    )
    df_uy = pd.DataFrame(
        data_metadata["utilities_y"], columns=[f"uy_{i}" for i in range(n_clusters)]
    )

    # Save data generation
    df_data = pd.concat([df_X, df_Y, df_ux, df_uy], axis=1)
    df_data["cluster"] = data_metadata["clusters"]
    df_data.to_csv(os.path.join(results_dir, "data.csv"), index=False)

    weights = np.stack([data_metadata[f"weights_{i}"] for i in range(n_clusters)])
    np.save(os.path.join(results_dir, f"data_weights.npy"), weights)
    mweights = np.stack(
        [data_metadata[f"marginal_weights_{i}"] for i in range(n_clusters)]
    )
    np.save(os.path.join(results_dir, f"marginal_data_weights.npy"), mweights)

    X_train, X_test = X[:learning_set_size], X[learning_set_size:]
    Y_train, Y_test = Y[:learning_set_size], Y[learning_set_size:]

    # Compute the model
    model = ClusterUTA(n_clusters=n_clusters, n_pieces=n_linear_pieces)
    t_start = time.time()
    hist = model.fit(X_train, Y_train, time_limit=time_limit)
    uta_train_time = time.time() - t_start

    # Compute and save results
    U_train = model.predict_utility(X_train)
    df_u_train_x = pd.DataFrame(
        U_train, columns=[f"u_x_train_{i}" for i in range(n_clusters)]
    )
    U_test = model.predict_utility(X_test)
    df_u_test_x = pd.DataFrame(
        U_test, columns=[f"u_x_test_{i}" for i in range(n_clusters)]
    )

    U_train = model.predict_utility(Y_train)
    df_u_train_y = pd.DataFrame(
        U_train, columns=[f"u_y_train_{i}" for i in range(n_clusters)]
    )
    U_test = model.predict_utility(Y_test)
    df_u_test_y = pd.DataFrame(
        U_test, columns=[f"u_y_test_{i}" for i in range(n_clusters)]
    )

    df_u_train_x.to_csv(
        os.path.join(results_dir, f"milo_u_x_train.csv"), index=False
    )
    df_u_test_x.to_csv(os.path.join(results_dir, f"milo_u_x_test.csv"), index=False)
    df_u_train_y.to_csv(
        os.path.join(results_dir, f"milo_u_y_train.csv"), index=False
    )
    df_u_test_y.to_csv(os.path.join(results_dir, f"milo_u_y_test.csv"), index=False)

    model.save_model(results_dir)
    np.save(os.path.join(results_dir, f"milo_fit_time.npy"), np.array(uta_train_time))

    with open(os.path.join(results_dir, "status.txt"), "w") as f:
        f.write(f"{model.status}\n")

    
    model = PLSHeuristic(
        models_class=UTA, n_clusters=n_clusters, n_init=20
    )
    t_start = time.time()
    hist = model.fit(X_train, Y_train)
    heuristic_train_time = time.time() - t_start

    U_train = model.predict_utility(X_train)
    df_u_train_x = pd.DataFrame(
        U_train, columns=[f"u_x_train_{i}" for i in range(n_clusters)]
    )
    U_test = model.predict_utility(X_test)
    df_u_test_x = pd.DataFrame(U_test, columns=[f"u_x_test_{i}" for i in range(n_clusters)])

    U_train = model.predict_utility(Y_train)
    df_u_train_y = pd.DataFrame(
        U_train, columns=[f"u_y_train_{i}" for i in range(n_clusters)]
    )
    U_test = model.predict_utility(Y_test)
    df_u_test_y = pd.DataFrame(U_test, columns=[f"u_y_test_{i}" for i in range(n_clusters)])

    df_u_train_x.to_csv(os.path.join(results_dir, "heuristic_u_x_train.csv"), index=False)
    df_u_test_x.to_csv(os.path.join(results_dir, "heuristic_u_x_test.csv"), index=False)
    df_u_train_y.to_csv(os.path.join(results_dir, "heuristic_u_y_train.csv"), index=False)
    df_u_test_y.to_csv(os.path.join(results_dir, "heuristic_u_y_test.csv"), index=False)
    np.save(os.path.join(results_dir, f"heuristic_fit_time.npy"), np.array(heuristic_train_time))
    


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
        default=2**12,
        type=int,
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
        "-cr",
        "--n_criteria",
        type=int,
        nargs="+",
        default=6,
        help="Number of criteria for the data - can be int or list",
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
        "--error",
        type=int,
        nargs="+",
        default=0,
        help="Error percentage - can be int or list.",
    )

    args = parser.parse_args()

    base_dir = args.save_dir
    repetitions = args.repetitions
    timeout = args.timeout
    test_set_size = args.test_set_size

    n_clusters = args.n_clusters
    if isinstance(n_clusters, int):
        n_clusters = [n_clusters]
    if not isinstance(n_clusters, list):
        raise ValueError(
            f"n_clusters should be int or list of int and is: {type(n_clusters)}"
        )

    n_criteria = args.n_criteria
    if isinstance(n_criteria, int):
        n_criteria = [n_criteria]
    if not isinstance(n_criteria, list):
        raise ValueError(
            f"n_criteria should be int or list of int and is: {type(n_criteria)}"
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

    error = args.error
    if isinstance(error, int):
        error = [error]
    if not isinstance(error, list):
        raise ValueError(f"error should be int or list of int and is: {type(error)}")

    for rep in range(repetitions):
        print(f"Currently at rep: {rep}")
        for err in error:
            for n_cl in n_clusters:
                for n_cr in n_criteria:
                    for n_p in n_pieces:
                        for lss in train_set_size:
                            run_id = f"{n_cl}_{n_cr}_{err}_{lss}_{n_p}_{rep}"
                            if os.path.exists(
                                os.path.join(base_dir, f"results/{run_id}")
                            ):
                                print(f"Skipping {run_id}")
                            else:
                                evaluation_routine(
                                    base_dir=base_dir,
                                    run_id=run_id,
                                    n_clusters=n_cl,
                                    n_criteria=n_cr,
                                    n_linear_pieces=n_p,
                                    data_error=err / 100,
                                    learning_set_size=lss,
                                    test_set_size=test_set_size,
                                    time_limit=timeout,
                                )
