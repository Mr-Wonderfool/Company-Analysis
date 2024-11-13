import pandas as pd
from models.utils.get_data import get_data
from models.selector import Selector
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


def main():
    data_path = "../data/data.csv"
    # for plotting
    data_all = pd.read_csv(data_path)
    target = "Bankrupt?"
    columns = data_all.drop(columns=target).columns.values

    data_X, data_Y = get_data(path=data_path, threshold=0.7)
    selector = Selector(data_X=data_X, data_Y=data_Y)
    features_f, _ = selector.select_univar(percent=30, method="mutual_info_classif")
    _select_data_X = data_X[..., features_f]
    fig, ax = selector.plot(figsize=(15, 6), columns=columns, threshold=0.2)
    ax.set_ylabel("scores")
    return _select_data_X, data_Y, fig


if __name__ == "__main__":
    _select_data_X, _data_Y, fig = main()
    # directly use select_data_X as data
    model = LogisticRegression()
    X_train, X_test, y_train, y_test = train_test_split(
        _select_data_X, _data_Y, test_size=0.2
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.tight_layout()
    plt.show()
