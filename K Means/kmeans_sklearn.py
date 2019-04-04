import matplotlib.pyplot as plt 
from matplotlib import style
style.use('ggplot')
import numpy as np 
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing, cross_decomposition
import handle_data

df = pd.read_excel('titanic.xls')
df.drop(['body', 'name'], 1, inplace=True)
df.infer_objects() # Attempt to infer better dtypes for object columns.
df.fillna(0, inplace=True)

df = handle_data.convert_non_numerical(df)
df.drop(['boat'], 1, inplace=True)

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)

hits = 0
for i in  range(len(X)):
    x = np.array(X[i].astype(float))
    x = x.reshape(-1, len(x))
    pred = clf.predict(x)
    if pred[0] == y[i]:
        hits += 1

print(hits/len(X))

