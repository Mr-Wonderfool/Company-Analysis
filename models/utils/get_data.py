import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def get_data(path: str, threshold: float):
    """
    :param path: path to "*.csv" data file
    :param threshold: pearson correlation coeff threshold
    :returns:
        data_X shape (m, #features)
        data_Y shape (m, )
    """
    data_all = pd.read_csv(path)
    target = "Bankrupt?"
    scaler = StandardScaler()
    data_X = data_all.drop(columns=target)
    data_Y = data_all[target]
    # remove highly correlated features
    corr_mat = data_X.corr().abs()
    # Create a mask to ignore the upper triangle (duplicates) and the diagonal
    upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    data_X = data_X.drop(columns=to_drop)
    data_X = scaler.fit_transform(data_X)
    data_Y = data_Y.values
    return data_X, data_Y
