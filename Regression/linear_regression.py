from statistics import mean
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

# x = np.array([1,2,3,4,5,6])
# y = np.array([5,4,6,5,6,7])

def create_dataset(n, variance, step=2, correlation=False):
    val = 1
    y = []
    for _ in range(n):
        y_i = val + random.randrange(-variance, variance)
        y.append(y_i)
        if correlation and correlation is 'pos':
            val += step
        elif correlation and correlation is 'neg':
            val -= step

    x = [i for i in range(len(y))]

    return np.array(x, dtype=np.float64), np.array(y, dtype=np.float64)

def best_fit_regression(x, y):
    m = (mean(x) * mean(y) - mean(x*y)) / (mean(x)**2 - mean(x**2))
    b = mean(y) - m * mean(x)
    return m, b

def squared_error(y, y_hat):
    return sum((y_hat - y)**2)

def r_squared(y, y_hat):
    y_mean_line = [mean(y) for _ in y]
    se = squared_error(y, y_hat)
    se_y_mean = squared_error(y, y_mean_line)
    return 1 - (se / se_y_mean)

x, y = create_dataset(40, 40, 2, correlation='pos')

m, b = best_fit_regression(x, y)

y_hat = m * x + b

predict_x = 8
predict_y = (m * predict_x) + b

rq = r_squared(y, y_hat)
print(rq)

plt.scatter(x, y)
plt.plot(x, y_hat)
plt.scatter(predict_x, predict_y, s=100, color='g')
plt.show()