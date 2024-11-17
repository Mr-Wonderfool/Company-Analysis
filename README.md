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
1. 模型文件：
    (完成配置之后所有文件可以直接运行)
    - `models/LogisticRegression.py`: 手写实现Logistic回归，同时包括特征选择和调用库函数实现
    - `models/GDA.ipynb`：手写实现高斯判别分析，包括和Logistic回归的对比
    - `models/visualization.ipynb`：数据集可视化和特征选择
    - `models/company_bankruptcy_prediction.ipynb`：集成学习部分，需要额外安装库，依赖已经记载在.ipynb文件中
    - `models/SVM.py`：调用库函数的SVM分类模型
    - `models/SVM_handwritten.py`：手写实现的SVM分类模型，运行时间较长（本地测试约6min）
2. 辅助文件：
   - `utils/`包含特征选择部分的代码，包括随机森林和互信息

### 小组成员
丁子昂，南佳延，徐志铭，赵唯旭