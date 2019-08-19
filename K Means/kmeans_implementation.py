import numpy as np 
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt 
from matplotlib import style
style.use('ggplot')
import handle_data

class K_means:
    def __init__(self, max_iter=300, K=2):
        self.K = K 
        self.clusters = [[] for _ in range(K)]
        self.centroids = None
        self.max_iter = max_iter

    def fit(self, X):
        K = self.K
        N = X.shape[0]
        centroids = X[np.random.choice(X.shape[0], K, replace=False)]
        s = np.zeros(N)
        d = np.zeros(K)
        clusters = [[] for _ in range(K)]
        J1 = np.inf
        J2 = 0
        count = 0
        while J2 < J1 and count < self.max_iter:
            self.clusters = clusters
            self.centroids = centroids
            clusters = [[] for _ in range(K)]
            for i in range(N):
                for k in range(K):
                    d[k] = np.linalg.norm(X[i] - centroids[k])
                s[i] = np.argmin(d)
                clusters[s[i].astype(int)].append(X[i])
                #print(clusters[s[i].astype(int)])

            for k in range(K):
                centroids[k] = (1/len(clusters[k]))*sum(clusters[k])
            
            if J2 is not 0:
                J1 = J2

            J2 = 0
            for k in range(K):
                for x in clusters[k]:
                    J2 += np.linalg.norm(x - centroids[k])

            count += 1
    
    def predict(self, x):
        K = self.K
        d = np.zeros(K)

        for k in range(K):
            d[k] = np.linalg.norm(x - self.centroids[k])

        return np.argmin(d) 


df = pd.read_excel('titanic.xls')
df.drop(['body', 'name'], 1, inplace=True)
df.infer_objects() # Attempt to infer better dtypes for object columns.
df.fillna(0, inplace=True)

df = handle_data.convert_non_numerical(df)
df.drop(['boat'], 1, inplace=True)

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])
clf = K_means(300, K=2)

clf.fit(X)
hits = 0
for i in range(len(X)):
    x = np.array(X[i].astype(float))
    x = x.reshape(-1, len(x))
    pred = clf.predict(x)
    if pred == y[i]:
        hits += 1

performance = hits/len(X)
print(hits/len(X))