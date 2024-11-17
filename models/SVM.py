import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import time  # 用于计算时间


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
    data_path = "../data/data.csv"
    data_X, data_Y, feature_names = get_data(path=data_path, threshold=0.7)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.2, stratify=data_Y)

    # 计算类别权重以处理数据不平衡
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = {label: weight for label, weight in zip(np.unique(y_train), class_weights)}
    print(f"类别权重: {class_weight_dict}")

    # 使用线性核的 SVM，添加类别权重
    model = LinearSVC(C=1, max_iter=5000, class_weight=class_weight_dict)
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

    # 提取特征重要性
    coef = model.coef_[0]  # 线性核的权重
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': np.abs(coef)
    }).sort_values(by='Importance', ascending=False)

    # 打印最重要的前10个特征
    print("\n特征重要性（前10个）：")
    print(feature_importance.head(10))

    # 绘制前20个特征的重要性
    top_20_features = feature_importance.head(20)

    plt.figure(figsize=(8, 6))  # 设置图形尺寸
    plt.bar(
        top_20_features['Feature'],
        top_20_features['Importance'],
        color="blue",
        label="Feature Importance"
    )
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Scores', fontsize=12)
    plt.title('Feature importances (Top 20, with score threshold = 0.2)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)  # 调整标签角度与字体大小
    plt.legend(fontsize=10, loc='upper right')  # 添加图例
    plt.tight_layout()  # 自动调整子图参数以适应画布
    plt.show()


if __name__ == "__main__":
    main()
