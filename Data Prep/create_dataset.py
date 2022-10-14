from utils import *
import pandas as pd
from tqdm import tqdm
import sys

# data_directory = 'E:\Roger\dataset\lst\q1\\2_layer0.tif'

# matrix = load_image(data_directory)
# grided_mat = blockshaped(matrix, 4, 4)
# average_lst = average_grid(grided_mat)

# data_directory_2 = 'E:\Roger\dataset\\ndvi\q1\\1_layer0.tif'
# matrix_ndvi = load_image(data_directory_2)
# grided_mat_nd = blockshaped(matrix_ndvi, 16, 16)
# average_ndvi = average_grid(grided_mat_nd)
df = pd.DataFrame(columns = ['LST', 'NDVI'])
total_fires = 0
for j in tqdm([1, 2, 3, 4]):
    data_directory_head_1 = 'E:\Roger\dataset\\lst\q' + str(j) + '\\'
    data_directory_head_2 = 'E:\Roger\dataset\\ndvi\q' + str(j) + '\\'
    data_directory_tail = '_layer0.tif'
    for i in tqdm(range(217)):
        l = 0
        if j == 1:
            l = i*8+433
        else:
            l = i*8+1
        data_directory_1 = data_directory_head_1 + str(l) + data_directory_tail
        k = int((i+1)/2)
        if k == 109:
            k = 108
        data_directory_2 = data_directory_head_2 + str(k) + data_directory_tail
        matrix_lst = load_image(data_directory_1)
        matrix_ndvi = load_image(data_directory_2)
        grided_lst = blockshaped(matrix_lst, 4, 4)
        grided_ndvi = blockshaped(matrix_ndvi, 16, 16)
        average_lst = average_grid(grided_lst)
        average_ndvi = average_grid(grided_ndvi)
        lst = pd.Series(average_lst, name='LST')
        ndvi = pd.Series(average_ndvi, name='NDVI')
        df2 = pd.concat([lst, ndvi], axis=1)
        df = pd.concat([df, df2], axis=0)
        
df.to_csv('input.csv')
