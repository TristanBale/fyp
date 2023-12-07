a = 3
b = 4
sum = a * b
print(sum)

import numpy as np
import pandas as pd

dataBP = np.load('./data/bp4features_norandom/test.npy')
data = np.load('./data/stockdataset/stockdataset/test.npy')

#print(dataBP)
print('--')
#print(data)

stock = pd.read_csv('data/BP.csv')
print(stock)
print(stock.shape)

#def get_sma(data, sma_days):