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
summer = pd.DataFrame(columns = ['DAY_LST', 'NDVI', 'NIGHT_LST'])
spring = pd.DataFrame(columns = ['DAY_LST', 'NDVI', 'NIGHT_LST'])
autumn = pd.DataFrame(columns = ['DAY_LST', 'NDVI', 'NIGHT_LST'])
winter = pd.DataFrame(columns = ['DAY_LST', 'NDVI', 'NIGHT_LST'])
total_fires = 0
for j in tqdm([1, 2, 3, 4]):
    data_directory_head_1 = 'E:\Roger\dataset\\lst\q' + str(j) + '_tif\\'
    data_directory_head_2 = 'E:\Roger\dataset\\ndvi\q' + str(j) + '_tif\\'
    data_directory_tail = '_layer0.tif'
    data_directory_tail_2 = '_layer4.tif'
    for i in tqdm(range(217)):
        l = i*8+1
        data_directory_1 = data_directory_head_1 + str(l) + data_directory_tail
        data_directory_3 = data_directory_head_1 + str(l) + data_directory_tail_2
        k = int((i+1)/2)
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
        check = i + 1
        if (0 < check < 9) or (42 < check < 54) or (88 < check < 100) or (133 < check < 146) or (179 < check < 191):
            summer = pd.concat([summer, df2], axis=0)
        elif (8 < check < 20) or (53 < check < 66) or (99 < check < 112) or (145 < check < 157) or (190 < check < 203):
            autumn = pd.concat([autumn, df2], axis=0)
        elif (19 < check < 32) or (65 < check < 77) or (111 < check < 123) or (156 < check < 169) or (202 < check < 213):
            winter = pd.concat([winter, df2], axis=0)
        elif (31 < check < 43) or (76 < check < 89) or (122 < check < 134) or (168 < check < 180) or (212 < check):
            spring = pd.concat([spring, df2], axis=0)
        else:
            print("ERROR")
        
        
summer.to_csv('summer.csv')
autumn.to_csv('autumn.csv')
spring.to_csv('spring.csv')
winter.to_csv('winter.csv')
