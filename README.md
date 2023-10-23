# comp49802023

https://colab.research.google.com/drive/1i7dZ05w-TXZCgkot4djN0y1SZYiWROn0#scrollTo=BeIOc-qoXQjm

https://colab.research.google.com/drive/1uWGaHhr8HqqWMX8N-w4sjqxD5aTlScnk#scrollTo=IzJJzWMEzahY




这段代码使用了Scikit-Learn中的 K 折交叉验证（KFold）和交叉验证评分（cross_val_score）来评估逻辑回归模型的性能。以下是代码的解释：

    from sklearn.model_selection import KFold, cross_val_score: 这里导入了Scikit-Learn库中的 KFold 和 cross_val_score 类和函数，用于执行 K 折交叉验证和计算交叉验证评分。

    cv = KFold(n_splits=4, shuffle=True): 这一行创建了一个 K 折交叉验证对象 cv。参数 n_splits 指定了将数据分成几个折叠（这里是 4 折），shuffle=True 表示在分割数据之前随机打乱数据。

    scores = cross_val_score(est, X, y, cv=cv, scoring='accuracy'): 这一行使用 cross_val_score 函数来执行交叉验证。它计算了逻辑回归模型 est 在数据 X 上的性能，使用了之前创建的交叉验证对象 cv。参数 scoring='accuracy' 表示使用准确性作为评估模型性能的指标。scores 是一个包含每个折叠的性能得分的数组。

    print("mean: ", np.mean(scores)): 这一行打印了交叉验证性能得分的均值。它表示模型在不同训练和测试数据集上的平均准确性。这是用来衡量模型整体性能的指标。

    print("SD: ", np.std(scores)): 这一行打印了交叉验证性能得分的标准差。标准差表示性能得分在不同折叠中的变化程度。较小的标准差表示性能更一致，较大的标准差表示性能变化较大。


Simple reg:
   "..# To make this.."
     X = X[:,0]: 这一行代码从数据集 X 中选择第一个特征（中位收入），并将其存储在 X 变量中。这是因为在简单线性回归中，我们只使用一个输入变量来进行预测。

    plt.scatter(X, y): 这一行代码使用散点图绘制了特征 X 和目标值 y 之间的关系。X轴表示中位收入，Y轴表示房屋价格中位数。这样的图形有助于可视化数据的分布。

    plt.xlabel(data.feature_names[0]): 这一行设置X轴的标签，即特征的名称。在这种情况下，X轴表示中位收入。

    plt.ylabel(data.target_names[0]): 这一行设置Y轴的标签，即目标值的名称。在这种情况下，Y轴表示房屋价格中位数。

    X = X.reshape(-1,1): 这一行代码对特征 X 进行了重塑，将其从一维数组（1D）转换为二维数组（2D）。Scikit-Learn的机器学习模型通常期望输入是一个二维数组，其中每行代表一个样本，每列代表一个特征。通过使用 reshape 函数，数据被调整为 n 行 x 1 列的形式，其中 n 是数据点的数量，1 列表示中位收入。这样符合Scikit-Learn的要求，以便用于训练和测试模型。
        X = X[:,0]: 这一行代码从数据集 X 中选择第一个特征（中位收入），并将其存储在 X 变量中。这是因为在简单线性回归中，我们只使用一个输入变量来进行预测。

    plt.scatter(X, y): 这一行代码使用散点图绘制了特征 X 和目标值 y 之间的关系。X轴表示中位收入，Y轴表示房屋价格中位数。这样的图形有助于可视化数据的分布。

    plt.xlabel(data.feature_names[0]): 这一行设置X轴的标签，即特征的名称。在这种情况下，X轴表示中位收入。

    plt.ylabel(data.target_names[0]): 这一行设置Y轴的标签，即目标值的名称。在这种情况下，Y轴表示房屋价格中位数。

    X = X.reshape(-1,1): 这一行代码对特征 X 进行了重塑，将其从一维数组（1D）转换为二维数组（2D）。Scikit-Learn的机器学习模型通常期望输入是一个二维数组，其中每行代表一个样本，每列代表一个特征。通过使用 reshape 函数，数据被调整为 n 行 x 1 列的形式，其中 n 是数据点的数量，1 列表示中位收入。这样符合Scikit-Learn的要求，以便用于训练和测试模型。


这段代码使用Scikit-Learn的 StandardScaler 对特征 X 进行标准化处理。标准化是一种常见的数据预处理步骤，它有助于确保不同特征具有相似的尺度，这对于许多机器学习算法的性能非常重要。以下是代码的解释：

    from sklearn.preprocessing import StandardScaler: 这一行导入了Scikit-Learn中的 StandardScaler 类，用于标准化数据。

    std = StandardScaler(): 这一行创建了一个 StandardScaler 的实例，将其存储在 std 变量中。

    X = std.fit_transform(X): 这一行使用 fit_transform 方法对特征 X 进行标准化。标准化的过程包括两个步骤：
        fit 步骤计算特征的均值和标准差，以便进行缩放。
        transform 步骤使用计算得到的均值和标准差来对特征进行缩放，使其具有均值为0、标准差为1的标准正态分布。

通过标准化，特征 X 中的值被转换为具有相似尺度的值，这有助于模型更好地理解特征之间的关系，避免某个特征的值范围过大导致模型偏向于该特征。这在许多机器学习算法中是一种良好的实践，尤其是线性模型等对特征尺度敏感的算法。

KNN AOC
https://colab.research.google.com/drive/1AUy9AVeF7BnzdaduseVvmmSe4U7LqNiw?usp=sharing

