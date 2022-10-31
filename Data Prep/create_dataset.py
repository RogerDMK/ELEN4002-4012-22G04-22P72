from utils import *
import pandas as pd
from tqdm import tqdm
import sys

window_size = 1
df = pd.DataFrame(columns = ['DAY_LST', 'NDVI', 'NIGHT_LST'])
total_fires = 0
for j in tqdm([1, 2, 3, 4]):
    data_directory_head_1 = 'E:\Roger\old dataset\\lst\q' + str(j) + '_tif\\'
    data_directory_head_2 = 'E:\Roger\old dataset\\ndvi\q' + str(j) + '_tif\\'
    data_directory_tail = '_layer0.tif'
    data_directory_tail_2 = '_layer4.tif'
    for i in tqdm(range(183)):
        l = i*8+2
        data_directory_1 = data_directory_head_1 + str(l) + data_directory_tail
        data_directory_3 = data_directory_head_1 + str(l) + data_directory_tail_2
        k = int((i)/2)
        if k == 109:
            k = 108
        data_directory_2 = data_directory_head_2 + str(k) + data_directory_tail
        matrix_lst = load_image(data_directory_1)
        matrix_ndvi = load_image(data_directory_2)
        matrix_night = load_image(data_directory_3)
        grided_lst = blockshaped(matrix_lst, 4, 4)
        grided_ndvi = blockshaped(matrix_ndvi, 16, 16)
        grided_night = blockshaped(matrix_night, 4, 4)
        average_lst = average_grid(grided_lst)
        average_ndvi = average_grid(grided_ndvi)
        average_night = average_grid(grided_night)
        lst = pd.Series(average_lst, name='DAY_LST')
        ndvi = pd.Series(average_ndvi, name='NDVI')
        night = pd.Series(average_night, name='NIGHT_LST')
        df2 = pd.concat([lst, ndvi, night], axis=1)
        df = pd.concat([df, df2], axis=0)
        
df.to_csv('input_old.csv')
