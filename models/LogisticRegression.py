import numpy as np
import matplotlib.pyplot as plt
from models.utils.get_data import get_data
from models.selector import Selector
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import seaborn as sns


# Sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 手写的逻辑回归类
class LogisticRegressionCustom:
    def __init__(self, lr=0.2, num_iter=100, regularization="l2", lambda_=0.1):
        self.lr = lr  # 学习率
        self.num_iter = num_iter  # 迭代次数
        self.regularization = regularization  # 正则化类型（None，'l2'）
        self.lambda_ = lambda_  # 正则化强度参数

    def fit(self, X, y):
        self.theta = np.zeros((X.shape[1], 1))  # 初始化参数
        for _ in range(self.num_iter):
            z = X.dot(self.theta)
            h = sigmoid(z)
            gradient = X.T.dot(h - y) / y.size
            if self.regularization == "l2":
                gradient += (self.lambda_ / y.size) * self.theta
            self.theta -= self.lr * gradient

    def predict(self, X):
        return (sigmoid(X.dot(self.theta)) >= 0.6).astype(int)


# 特征选择
def select_feature():
    data_path = "../data/data.csv"
    data_X, data_Y = get_data(path=data_path, threshold=0.7)
    selector = Selector(data_X=data_X, data_Y=data_Y)
    features_f, _ = selector.select_univar(percent=30, method="mutual_info_classif")
    _select_data_X = data_X[..., features_f]
    return _select_data_X, data_Y


# 参数搜索
def grid_search(X_train, y_train, X_test, y_test, lrs, lambdas):
    for lr in lrs:
        for lambda_ in lambdas:
            model = LogisticRegressionCustom(
                lr=lr, num_iter=100, regularization="l2", lambda_=lambda_
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            precision = report["1"]["precision"]
            recall = report["1"]["recall"]
            f1_score = report["1"]["f1-score"]
            print(
                f"lr: {lr:<4}, lambda_: {lambda_:<4}, precision: {precision:<4.2f}, recall: {recall:<4.2f}, f1-score: {f1_score:<4.2f}"
            )


# 绘制混淆矩阵
def plot_confusion_matrix(cm, title, labels):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    _select_data_X, _data_Y = select_feature()
    X_train, X_test, y_train, y_test = train_test_split(
        _select_data_X, _data_Y, test_size=0.2
    )

    """调用库函数的逻辑回归模型"""
    model_sklearn = LogisticRegression()
    model_sklearn.fit(X_train, y_train)
    y_pred_sklearn = model_sklearn.predict(X_test)

    # 输出库函数的分类报告
    print("\n调用库函数的分类报告:")
    print(classification_report(y_test, y_pred_sklearn))

    # 绘制库函数的混淆矩阵
    cm_sklearn = confusion_matrix(y_test, y_pred_sklearn)
    plot_confusion_matrix(
        cm_sklearn, "Confusion Matrix (Sklearn Model)", labels=["0", "1"]
    )

    """手写的逻辑回归模型"""
    y_train = y_train.reshape(-1, 1)

    # 设置待搜索的学习率和正则化强度
    lrs = [0.01, 0.05, 0.1, 0.2, 0.5]
    lambdas = [0.01, 0.05, 0.1, 0.5, 1, 5]
    grid_search(X_train, y_train, X_test, y_test, lrs, lambdas)

    # 训练最佳参数的手写模型
    model = LogisticRegressionCustom(
        lr=0.2, num_iter=100, regularization="l2", lambda_=0.1
    )
    model.fit(X_train, y_train)
    y_pred_best = model.predict(X_test)

    # 输出手写模型的分类报告
    print("\n手写函数的分类报告:")
    print(classification_report(y_test, y_pred_best))

    # 绘制手写模型的混淆矩阵
    cm_custom = confusion_matrix(y_test, y_pred_best)
    plot_confusion_matrix(
        cm_custom, "Confusion Matrix (Custom Model)", labels=["0", "1"]
    )
