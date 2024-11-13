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
- 检验环境是否配置成功
```bash
cd models
python example.py
```