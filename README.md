### 数据集来源
- [Kaggle数据集](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction)

### 使用
- clone仓库
- 构建虚拟环境（或者使用conda环境）
```bash
python -m venv <venv_name>
<venv_name>/Scripts/activate
pip install -r requirements.txt
```
- 配置相应的库
```bash
cd repo
pip install -e .
```

### 文件说明
(完成配置之后所有文件可以直接运行)
- `models/LogisticRegression.py`: 手写实现Logistic回归，同时包括特征选择和调用库函数实现
- `models/GDA.ipynb`：手写实现高斯判别分析，包括和Logistic回归的对比
- `models/visualization.ipynb`：数据集可视化，包括特征选择