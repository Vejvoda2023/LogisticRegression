# 逻辑回归分类鸢尾花数据集

## 📌 项目简介
本项目使用 `scikit-learn` 库实现 **逻辑回归**（Logistic Regression）来分类 **鸢尾花数据集（Iris Dataset）**，并进行了如下优化：
- **使用全部 4 个特征**（萼片长度、萼片宽度、花瓣长度、花瓣宽度）。
- **标准化数据** 以提高模型稳定性。
- **5 折交叉验证** 评估模型的泛化能力。
- **计算混淆矩阵和召回率**，分析分类效果。
- **可视化混淆矩阵** 以更直观地理解分类结果。

---

## 🚀 依赖项
```bash
pip install -r requirements.txt
```

---

## 🔧 代码运行
### 1️⃣ 下载或克隆本项目
```bash
git clone https://github.com/Vejvoda2023/LogisticRegression.git
cd LogisticRegression
```



---

## 📊 主要代码结构
```python
# 加载数据集
iris = datasets.load_iris()
X = iris.data  # 使用所有 4 个特征
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 数据标准化
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# 训练逻辑回归模型
lr = LogisticRegression(C=1000.0, solver='lbfgs', multi_class='auto')
lr.fit(X_train_std, y_train)

# 交叉验证（5 折）
cv_scores = cross_val_score(lr, X_train_std, y_train, cv=5, scoring='accuracy')
print(f'5折交叉验证准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

# 预测并计算混淆矩阵和分类报告
y_pred = lr.predict(X_test_std)
conf_matrix = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
```

---

## 📈 结果分析
示例运行结果：
```
5折交叉验证准确率: 0.9619 ± 0.0212

混淆矩阵:
[[15  0  0]
 [ 0 12  1]
 [ 0  0 17]]

分类报告:
              precision    recall  f1-score   support
     setosa       1.00      1.00      1.00        15
 versicolor       1.00      0.92      0.96        13
  virginica       0.94      1.00      0.97        17
```
- **交叉验证平均准确率**：96.19%，说明模型稳定。
- **召回率（Recall）**：模型能够很好地识别不同类别的样本。
- **混淆矩阵可视化**：
  ```python
  import seaborn as sns
  plt.figure(figsize=(6,5))
  sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=iris.target_names, yticklabels=iris.target_names)
  plt.xlabel("预测值")
  plt.ylabel("真实值")
  plt.title("混淆矩阵")
  plt.show()
  ```
  
---

## 📌 可能的改进方向
✅ **使用其他模型**（SVM、随机森林等）。  
✅ **优化超参数**（使用 `GridSearchCV` 自动调参）。  
✅ **使用特征选择**（如 `SelectKBest` 选取最佳特征）。  
✅ **处理类别不均衡**（如 `class_weight='balanced'`）。

---

## 📜 许可证
本项目遵循 **MIT License**，可自由使用、修改和分发。

---

## ❤️ 联系方式
如果你有任何问题或建议，欢迎联系我：
📧 邮箱: vejvoda_0@outlook.com  
🐙 GitHub: [Vejvoda2023](https://github.com/Vejvoda2023)  

