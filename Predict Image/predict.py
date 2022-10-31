from utils import *
import numpy as np
from tqdm import tqdm
import pickle
from matplotlib import pyplot as plt
from video_generator import VideoGenerator
import pathlib
from datetime import datetime, date, timedelta



def confidence(i):
    data_directory_tail = '_layer0.tif'
    data_directory_1 = ''
    data_directory_2 = ''
    q1 = np.zeros((300, 300))
    q2 = np.zeros((300, 300))
    q3 = np.zeros((300, 300))
    q4 = np.zeros((300, 300))
    for j in [1,2,3,4]:
        data_directory_head_1 = 'E:\Roger\dataset\\lst\q' + str(j) + '_tif\\'
        data_directory_head_2 = 'E:\Roger\dataset\\ndvi\q' + str(j) + '_tif\\'
        data_directory_tail = '_layer0.tif'
        data_directory_tail_2 = '_layer4.tif'
        time_step = 4
        l = i*time_step+1
        data_directory_1 = data_directory_head_1 + str(l) + data_directory_tail
        data_directory_3 = data_directory_head_1 + str(l) + data_directory_tail_2
        k = int((i+1)/4)
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
        lst_day = average_lst/np.abs(average_lst).max()
        ndvi = average_ndvi/np.abs(average_ndvi).max()
        lst_night = average_night/np.abs(average_night).max()
        combined = np.vstack((ndvi, lst_day, lst_night)).T
        if j == 1:
            q1 = combined
        if j == 2:
            q2 = combined
        if j == 3:
            q3 = combined
        if j == 4:
            q4 = combined
    
    filename = 'trained_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    result_q1 = loaded_model.predict_proba(q1)
    result_q2 = loaded_model.predict_proba(q2)
    result_q3 = loaded_model.predict_proba(q3)
    result_q4 = loaded_model.predict_proba(q4)
    return(combine_all_q(result_q1, result_q2, result_q3, result_q4))

video = VideoGenerator(pathlib.Path("./"))


date_1 = date(2018, 1, 3)

for i in tqdm(range(30)):
    image = confidence(i)
    current_month = date_1.month
    image_title = ''
    if current_month > 2 and current_month < 6:
        image_title = 'Season: Autumn'
    elif current_month > 5 and current_month < 9:
        image_title = 'Season: Winter'
    elif current_month > 8 and current_month < 12:
        image_title = 'Season: Spring'
    else:
        image_title = 'Season: Summer'
    title_full = 'Date: ' + date_1.strftime("%d/%m/%Y") + ' ' + image_title
    video.save_image(image, title_full)
    date_1 = date_1 + timedelta(days=4)

video.create_video('1_year_2')
video.clear_images()