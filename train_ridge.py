import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

def train_pred_one_fold(fold: int, df: pd.DataFrame, features: list, rmse: list):
    """
    This function trains and predicts on one fold of your selected model
    df is the train df, test_df is the test_df
    X features are defined in features
    y output is target
    oof score is printed and stored in the rmse list
    """
    train = df[df.kfold != fold]
    X_train = train[features]
    y_train = train["target"]
 
    valid = df[df.kfold == fold]
    X_valid = valid[features]
    y_valid = valid["target"]

    print(train.shape, valid.shape)
    
    model = Ridge(alpha=5)    
    model.fit(X_train, y_train)
    oof = model.predict(X_valid)
    print(np.sqrt(mean_squared_error(y_valid, oof)))
    rmse.append(np.sqrt(mean_squared_error(y_valid, oof)))
        
    return oof


if __name__ == "__main__":

    NUM_FOLDS = 5
    
    df = pd.read_csv("data/train.csv")
    df_num = pd.read_csv("data/train_numerical_features.csv")
    df_num = df[["id"]].merge(df_num)
    
    print("train data:", df_num.shape)
    with open("data/numerical_features.pkl", "rb") as reader:
        features = pickle.load(reader)

    print("features:", len(features))

    SEED = 1000
    rmse = []
    preds = []
    kfold = KFold(n_splits=NUM_FOLDS, random_state=SEED, shuffle=True)
    df_num["kfold"] = -1
    df["num_pred"] = -1
    for fold, (train_indices, val_indices) in enumerate(kfold.split(df_num)):
        df_num.loc[val_indices, "kfold"] = fold
        pred = train_pred_one_fold(fold, df_num, features, rmse)
        print(pred.shape)
        df.loc[val_indices, "num_pred"] = pred

    df["dif"] = (df.target - df.num_pred).abs()
    print(
        "Mean:", np.sqrt(((df.target.values - df.num_pred.values) ** 2).mean())
        )
    df.to_csv("data/train_ridge.csv", index=False)

