from osgeo import gdal
import os
import matplotlib.pyplot as plt
import math
import numpy as np
from joblib import Parallel, delayed


# Function to convert hdf files to geotiff files
def hdf_to_geotiff():
    # LST, NDVI and THERMAL ANOMALIES
    for x in range(0, 3):
        if x == 0:
            data_type = 'lst'
        elif x == 1:
            data_type = 'ndvi'
        else:
            data_type = 'thermal_anomalies'
        # Four quadrants: q1, q2, q3, q4
        # Quadrants correspond to the location of the satellite image
        # q2 | q1
        # -------
        # q3 | q4
        for y in range(1, 5):
            # Specify the folder for instance:
            # ./dataset/lst/q1_hdf
            folder = './dataset_2003-2006/{0}/q{1}_hdf'.format(data_type, y)
            file_names = os.listdir(folder)
            for z in range(0, len(file_names)):
                hdf_file = '{0}/{1}'.format(folder, file_names[z])
                hdf_ds = gdal.Open(hdf_file, gdal.GA_ReadOnly)
                sds = hdf_ds.GetSubDatasets()
                # Inc is set to choose the correct data layer of the hdf file
                if data_type == 'lst':
                    inc = 4
                else:
                    inc = 12
                # Iterate through layers
                for i in range(0, len(sds), inc):
                    if i != 8:
                        sds_name = sds[i][0]
                        layer_ds = gdal.Open(sds_name, gdal.GA_ReadOnly)
                        out_folder = './dataset_2003-2006/{0}/q{1}_tif'.format(data_type, y)
                        # Xz in this case to keep the naming convention constant as there were hdf files missing
                        # If this isn't the case simply remove this part
                        xz = z
                        if data_type == 'lst':
                            if z > 30:
                                xz = z + 1
                            if z > 349:
                                if y == 2 or y == 3:
                                    xz = z + 7
                                if y == 1 or y == 4:
                                    xz = z + 8
                        # ----------------------------------------------
                        layer_path = os.path.join(out_folder, '{0}_layer{1}.tif'.format(xz, i))
                        if layer_ds.RasterCount > 1:
                            for j in range(1, layer_ds.RasterCount + 1):
                                layer = layer_ds.GetRasterBand(j)
                                layer_array = layer.ReadAsArray()
                        else:
                            layer_array = layer_ds.ReadAsArray()
                        out_ds = gdal.GetDriverByName('GTiff').Create(layer_path,
                                                                      layer_ds.RasterXSize,
                                                                      layer_ds.RasterYSize,
                                                                      1,
                                                                      gdal.GDT_Int16,
                                                                      ['COMPRESS=LZW', 'TILED=YES'])
                        out_ds.SetGeoTransform(layer_ds.GetGeoTransform())
                        out_ds.SetProjection(layer_ds.GetProjection())
                        out_ds.GetRasterBand(1).WriteArray(layer_array)
                        out_ds.GetRasterBand(1).SetNoDataValue(-32768)
                        out_ds = None
    return


# Function for converting a geotiff file to an array
def geotiff_to_array(file):
    ds = gdal.Open(file)
    b = ds.GetRasterBand(1)
    arr = gdal.Band.ReadAsArray(b)
    return arr


# Function to display a geotiff file
def display(file):
    ds = gdal.Open(file)
    array = ds.GetRasterBand(1).ReadAsArray()
    plt.figure()
    plt.imshow(array)
    plt.colorbar()
    plt.show()
    ds = None


# Function for determining the average value of a matrix
def avg(matrix):
    matrix_sum = 0
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            matrix_sum += matrix[i][j]
    matrix_avg = matrix_sum / len(matrix) ** 2
    return matrix_avg


# Function for converting the values of the thermal anomaly files
# Values are converted to:
# 0 - Non-fire
# 1 - Fire
# 2 - Cloud
# 3 - Unknown
def ta_conversion(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if matrix[i][j] == 3 or matrix[i][j] == 5:
                matrix[i][j] = 0
            elif matrix[i][j] >= 7:
                matrix[i][j] = 1
            elif matrix[i][j] == 4:
                matrix[i][j] = 2
            else:
                matrix[i][j] = 3
    return matrix


# Function to count the occurrences of different pixel states
# Use this function after applying ta_conversion
def ta_count(matrix):
    fire, non_fire, cloud, unknown = 0, 0, 0, 0
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if matrix[i][j] == 0:
                non_fire += 1
            elif matrix[i][j] == 1:
                fire += 1
            elif matrix[i][j] == 2:
                cloud += 1
            else:
                unknown += 1
    return [fire, non_fire, cloud, unknown]


# Function for interpolating data
def idw_spatio_temporal_interpolation(matrix1, matrix2, data_type):
    time = 0
    matrix_out = []
    if data_type == 'ndvi':
        time = 16
    else:
        time = 8
    c = (1 / time) / len(matrix1)
    # NDVI - 16 days between images
    # THERMAL ANOMALIES - 8 days between images
    for i in range(1, time):
        matrix_inner = []
        for j in range(1, len(matrix1)):
            matrix_temp = []
            for k in range(1, len(matrix1)):
                sum = 0
                # Only the values directly next to the pixel to be estimated are used
                for l in range(j - 1, j + 1):
                    for m in range(k - 1, k + 1):
                        x = j
                        y = k
                        t = i
                        xi = l
                        yi = m
                        ti1 = 0
                        ti2 = time
                        di1 = math.sqrt((xi - x) ** 2 + (yi - y) ** 2 + (c ** 2) * (ti1 - t) ** 2)
                        di2 = math.sqrt((xi - x) ** 2 + (yi - y) ** 2 + (c ** 2) * (ti2 - t) ** 2)
                        lmbd1 = (1 / di1) ** (1 / t)
                        lmbd2 = (1 / di2) ** (1 / (time - t))
                        sum += lmbd1 * matrix1[xi][yi] + lmbd2 * matrix2[xi][yi]
                matrix_temp.append(sum)
            matrix_inner.append(matrix_temp)
        matrix_out.append(matrix_inner)
    return matrix_out


# Linear function for interpolating data
def linear_interpolation_parallel(matrix1, matrix2, data_type):
    time = 0
    matrix_out = []
    if data_type == 'ndvi':
        time = 16
    else:
        time = 8
    for i in range(1, time):
        out = Parallel(n_jobs=-1, verbose=1)(delayed(parallel)(matrix1, matrix2, i, j, time)
                                             for j in range(0, len(matrix1)))
        matrix_out.append(out)
        print(i)
    return np.array(matrix_out)


# Parallel part of the linear function
def parallel(matrix1, matrix2, i, j, time):
    matrix_temp = []
    for k in range(0, len(matrix1)):
        t1 = 0
        t2 = time
        wi1 = matrix1[j][k]
        wi2 = matrix2[j][k]
        w = ((t2 - i) / (t2 - t1)) * wi1 + ((i - t1) / (t2 - t1)) * wi2
        matrix_temp.append(int(w))
    return matrix_temp


# Function for interpolating geotiff files in a folder
def interpolate():
    # LST, NDVI and THERMAL ANOMALIES
    for i in range(0, 3):
        # Four quadrants: q1, q2, q3, q4
        # Quadrants correspond to the location of the satellite image
        # q2 | q1
        # -------
        # q3 | q4
        for j in range(1, 5):
            if i == 0:
                data_type = 'ndvi'
                files = 15
            else:
                data_type = 'thermal_anomalies'
                files = 7
            # Specify the folder for instance:
            # ./dataset/lst/q1_hdf
            folder = './dataset/{0}/q{1}_tif'.format(data_type, j)
            file_names = os.listdir(folder)
            file_names = sorted(file_names, key=lambda d: os.path.getmtime(os.path.join(folder, d)))
            for k in range(1, len(file_names) - 1):
                arr1 = geotiff_to_array('{0}/{1}'.format(folder, file_names[k - 1]))
                arr2 = geotiff_to_array('{0}/{1}'.format(folder, file_names[k]))
                # Specify the interpolation method
                output_arr = linear_interpolation_parallel(arr1, arr2, data_type)
                input_arr = gdal.Open('{0}/{1}'.format(folder, file_names[k]), gdal.GA_ReadOnly)
                for x in range(1, files + 1):
                    output_path = os.path.join('{0}_interpolated'.format(folder),
                                               '{0}.tif'.format(files * (k - 1) + x + k - 1))
                    out_ds = gdal.GetDriverByName('GTiff').Create(output_path,
                                                                  input_arr.RasterXSize,
                                                                  input_arr.RasterYSize,
                                                                  1,
                                                                  gdal.GDT_Int16,
                                                                  ['COMPRESS=LZW', 'TILED=YES'])
                    out_ds.SetGeoTransform(input_arr.GetGeoTransform())
                    out_ds.SetProjection(input_arr.GetProjection())
                    out_ds.GetRasterBand(1).WriteArray(output_arr[x - 1])
                    out_ds.GetRasterBand(1).SetNoDataValue(-32768)
                    out_ds = None
    return
