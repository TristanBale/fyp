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
#print(stock)
#print(stock.shape)

#def get_sma(data, sma_days):


array1 = np.array([1, 2, 3])
array2 = np.array([[4, 5, 6], [7, 8, 9]])
result = np.concatenate((array1, array2.flatten()))

array2f = array2.flatten()
array2list = list(array2)


# Function to combine features from two stocks in two different ways
def combine_features(stock1_data, stock2_data):
    # Assuming stock1_data and stock2_data have shape (num_samples, window_size, num_features)
    assert stock1_data.shape == stock2_data.shape, "Shapes of both stocks should be the same"

    # Create empty arrays to store combined data
    combined_data_1 = np.empty((stock1_data.shape[0], stock1_data.shape[1], stock1_data.shape[2] * 2))
    combined_data_2 = np.empty((stock1_data.shape[0], stock1_data.shape[1], stock1_data.shape[2] * 2))

    # Concatenate features: A1A2A3B1B2B3
    combined_data_1[:, :, :stock1_data.shape[2]] = stock1_data
    combined_data_1[:, :, stock1_data.shape[2]:] = stock2_data

    # Concatenate features: A1B1A2B2A3B3
    for i in range(stock1_data.shape[2]):
        combined_data_2[:, :, i * 2] = stock1_data[:, :, i] #A1_A2_A3_B1_B2_B3 (Order by stock. puts all features of stock A before stock B)
        combined_data_2[:, :, i * 2 + 1] = stock2_data[:, :, i] #A1_B1_A2_B2_A3_B3 (Order by feature. puts each feature of both stock side by side before moving on to the next feature)

    return combined_data_1, combined_data_2

import numpy as np

# Function to combine features from multiple stocks in two different ways
def combine_multiple_stocks(*args):
    num_stocks = len(args)
    assert num_stocks >= 2, "At least two stocks are required for merging"

    # Check if all arrays have the same shape
    shapes = [arg.shape for arg in args]
    assert all(shape == shapes[0] for shape in shapes), "Shapes of all stocks should be the same"

    # Create empty arrays to store combined data
    combined_data_1 = np.empty((args[0].shape[0], args[0].shape[1], args[0].shape[2] * num_stocks))
    combined_data_2 = np.empty((args[0].shape[0], args[0].shape[1], args[0].shape[2] * num_stocks))

    # Concatenate features: A1A2A3B1B2B3...
    for i, stock_data in enumerate(args):
        combined_data_1[:, :, i * stock_data.shape[2]:(i + 1) * stock_data.shape[2]] = stock_data

    # Concatenate features: A1B1A2B2A3B3...
    for i in range(args[0].shape[2]):
        for j, stock_data in enumerate(args):
            combined_data_2[:, :, i * num_stocks + j] = stock_data[:, :, i]

    return combined_data_1, combined_data_2

#(5,10,2) (5,10,2)
arrayA = np.random.rand(2,2,2)
print(arrayA)
print('--')
arrayB = np.random.rand(2,2,2)
print(arrayB)

print('--')
x, y = combine_multiple_stocks(arrayA,arrayB)
print(x)
print('--')
print(y)