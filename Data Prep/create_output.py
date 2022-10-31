from utils import *
import pandas as pd
from tqdm import tqdm
import sys

df = pd.DataFrame(columns = ['FIRE'])
total_fires = 0
for j in tqdm([1, 2, 3, 4]):
    data_directory_head = 'E:\Roger\old dataset\\thermal_anomalies\q' + str(j) + '_tif\\'
    data_directory_tail = '_layer0.tif'
    for i in tqdm(range(183)):
        data_directory = data_directory_head + str(i+1) + data_directory_tail
        matrix_ta_raw = load_image(data_directory)
        matrix_ta = ta_conversion(matrix_ta_raw)
        grided_ta = blockshaped(matrix_ta, 4, 4)
        average_ta_mat = average_ta(grided_ta)
        ta = pd.DataFrame({'FIRE': average_ta_mat})
        df = pd.concat([df, ta], axis=0)
        
        
df.to_csv('output_old.csv')
