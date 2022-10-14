from utils import *
import pandas as pd
from tqdm import tqdm
import sys

df = pd.DataFrame(columns = ['FIRE'])
total_fires = 0
print(df)
for j in tqdm([1, 2, 3, 4]):
    data_directory_head = 'E:\Roger\dataset\\thermal_anomalies\q' + str(j) + '\\'
    data_directory_tail = '_layer0.tif'
    for i in tqdm(range(217)):
        data_directory = data_directory_head + str(i) + data_directory_tail
        matrix_ta_raw = load_image(data_directory)
        matrix_ta = ta_conversion(matrix_ta_raw)
        grided_ta = blockshaped(matrix_ta, 4, 4)
        average_ta_mat = average_ta(grided_ta)
        ta = pd.DataFrame({'FIRE': average_ta_mat})
        df = pd.concat([df, ta], axis=0)
        
        
df.to_csv('output.csv')
