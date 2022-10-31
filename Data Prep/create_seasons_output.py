from utils import *
import pandas as pd
from tqdm import tqdm
import sys

summer = pd.DataFrame(columns = ['FIRE'])
spring = pd.DataFrame(columns = ['FIRE'])
autumn = pd.DataFrame(columns = ['FIRE'])
winter = pd.DataFrame(columns = ['FIRE'])
total_fires = 0
for j in tqdm([1, 2, 3, 4]):
    data_directory_head = 'E:\Roger\dataset\\thermal_anomalies\q' + str(j) + '_tif\\'
    data_directory_tail = '_layer0.tif'
    for i in tqdm(range(217)):
        data_directory = data_directory_head + str(i+1) + data_directory_tail
        matrix_ta_raw = load_image(data_directory)
        matrix_ta = ta_conversion(matrix_ta_raw)
        grided_ta = blockshaped(matrix_ta, 4, 4)
        average_ta_mat = average_ta(grided_ta)
        df2 = pd.DataFrame({'FIRE': average_ta_mat})
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
        
        
summer.to_csv('summer_out.csv')
autumn.to_csv('autumn_out.csv')
spring.to_csv('spring_out.csv')
winter.to_csv('winter_out.csv')
