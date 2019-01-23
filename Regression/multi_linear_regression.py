import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

def beta_hat(X, y):
    return np.inv(np.transpose(X) @ X) @ np.transpose(X) @ y

