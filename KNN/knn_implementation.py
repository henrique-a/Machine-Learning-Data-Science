import numpy as np 
import matplotlib.pyplot as plt 
import warnings
from matplotlib import style 
from collections import Counter
import pandas as pd 
import random

style.use('fivethirtyeight')

dataset = {'black': [[1,2], [2,3], [3,1]], 'red':[[6,5], [7,7], [8,6]]}

new_feature = [5,7]

def KNN(data, predict, k=3):
    distances = []
    for group in data:
        for features in data[group]:
            euclidian_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidian_distance, group])
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

result = KNN(dataset, new_feature, k=3)

print("point: {}".format(new_feature))
print("predicted class: {}".format(result))

[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_feature[0], new_feature[1], color=result)
plt.show()