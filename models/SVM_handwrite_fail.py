import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import time  # 用于计算时间


# 高斯核函数，用于特征选择
def gaussian_kernel(data, tau=1):
    n_samples = len(data)
    kernel = np.ones((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i):
            diff = data[i] - data[j]
            kernel[i][j] = np.exp(-np.sum(diff**2) / (2 * tau**2))
            kernel[j][i] = kernel[i][j]
    return kernel


def is_psd(matrix):
    """检查矩阵是否为正定"""
    eigvals = np.linalg.eigvals(matrix)
    return np.all(eigvals >= 0)


# 数据处理与加载
def get_data(path: str, threshold: float):
    data_all = pd.read_csv(path)
    target = "Bankrupt?"
    scaler = StandardScaler()
    data_X = data_all.drop(columns=target)
    data_Y = data_all[target]

    data_Y = data_Y.replace({0: -1})  # 标签转换为-1

    # 相关性分析，去除高相关性特征
    corr_mat = data_X.corr().abs()
    upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    data_X = data_X.drop(columns=to_drop)

    # 保留处理后的特征名称
    feature_names = data_X.columns

    data_X = scaler.fit_transform(data_X)  # 标准化
    return data_X, data_Y, feature_names


# 主函数
def main():
    data_path = "data/data.csv"
    data_X, data_Y, feature_names = get_data(path=data_path, threshold=0.7)

    # 使用高斯核特征映射
    tau = 0.1
    kernel_matrix = gaussian_kernel(data_X, tau)
    if not is_psd(kernel_matrix):
        print("核矩阵不是正定的，添加正则项修正...")
        kernel_matrix += np.eye(kernel_matrix.shape[0]) * 1e-5

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(kernel_matrix, data_Y, test_size=0.2, stratify=data_Y)

    # 使用高斯核的 SVM
    model = SVC(kernel='precomputed', C=1)  # 预计算核
    start_time = time.time()
    model.fit(X_train, y_train)  # 训练模型
    training_time = time.time() - start_time

    # 输出训练信息
    print(f"训练时间: {training_time:.4f} 秒")
    print(f"SVM 模型的超参数：C: {model.C}")

    # 预测
    y_pred = model.predict(X_test)

    # 使用分类报告进行全面评估
    print("\n分类报告：")
    print(classification_report(y_test, y_pred, target_names=["Not Bankrupt", "Bankrupt"]))

    # 绘制混淆矩阵
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    ax.set_title('Confusion Matrix', fontsize=16, pad=20)
    plt.tight_layout(pad=3.0)
    plt.show()


if __name__ == "__main__":
    main()
