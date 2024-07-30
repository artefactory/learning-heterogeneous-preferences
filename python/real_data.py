"""Loading the stated preferences for cars dataset."""
import numpy as np
from choice_learn.datasets.base import load_car_preferences

def load_cars_preferences_pairs():
    df = load_car_preferences(as_frame=True)

    monotonous_features = ["price", "range", "acc", "speed", "pollution", "space", "cost", "station"]
    features_objective = ["min", "max", "min", "max", "min", "max", "min", "max"]

    # Filter outliers
    cars_df = df.copy()
    cars_df = (
        cars_df.loc[cars_df.price1 < 12.5]
        .loc[cars_df.price2 < 12.5]
        .loc[cars_df.price3 < 12.5]
        .loc[cars_df.price4 < 12.5]
        .loc[cars_df.price5 < 12.5]
        .loc[cars_df.price6 < 12.5]
    )
    # Reverse features that need to be minimized & normalization
    for i in range(len(monotonous_features)):
        if features_objective[i] == "min":
            for j in range(6):
                cars_df[f"{monotonous_features[i]}{j+1}"] = -cars_df[f"{monotonous_features[i]}{j+1}"]

        min_val = np.min(cars_df[[f"{monotonous_features[i]}{j+1}" for j in range(6)]].to_numpy())
        max_val = np.max(cars_df[[f"{monotonous_features[i]}{j+1}" for j in range(6)]].to_numpy())

        for j in range(6):
            cars_df[f"{monotonous_features[i]}{j+1}"] = (
                cars_df[f"{monotonous_features[i]}{j+1}"] - min_val
            ) / (max_val - min_val)

    cars_df["choice"] = cars_df.apply(lambda row: int(row.choice[-1]), axis=1)

    # Build X (preferred alt.) and Y (non-preferred alt.) pairs
    # Build indicator of choice (regrouping pairs of preferences)
    X = []
    Y = []
    choice_ids = []
    for n_row, row in cars_df.iterrows():
        for j in range(1, 7):
            if j == row.choice:
                for _ in range(5):
                    X.append(
                        [row[f"{monotonous_features[i]}{j}"] for i in range(len(monotonous_features))]
                    )
            else:
                Y.append([row[f"{monotonous_features[i]}{j}"] for i in range(len(monotonous_features))])
                choice_ids.append(n_row)

    X = np.stack(X)
    Y = np.stack(Y)
    choice_ids = np.stack(choice_ids)

    # Remove dominance
    dominance = np.sum(X - Y >= 0, axis=1) > 7
    X = X[~dominance]
    Y = Y[~dominance]
    choice_ids = choice_ids[~dominance]

    return X, Y, choice_ids