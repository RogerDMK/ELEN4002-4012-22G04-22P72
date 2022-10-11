from osgeo import gdal
import os
import matplotlib.pyplot as plt


def hdf_to_geotiff():
    for x in range(0, 3):
        if x == 0:
            data_type = 'lst'
        elif x == 1:
            data_type = 'ndvi'
        else:
            data_type = 'thermal_anomalies'
        for y in range(1, 5):
            folder = './dataset/{0}/q{1}'.format(data_type, y)
            print(folder)
            file_names = os.listdir(folder)
            for z in range(0, len(file_names)):
                hdf_file = '{0}/{1}'.format(folder, file_names[z])
                hdf_ds = gdal.Open(hdf_file, gdal.GA_ReadOnly)
                sds = hdf_ds.GetSubDatasets()
                if data_type == 'lst':
                    inc = 4
                else:
                    inc = 12
                for i in range(0, len(sds), inc):
                    if i != 8:
                        sds_name = sds[i][0]
                        layer_ds = gdal.Open(sds_name, gdal.GA_ReadOnly)
                        layer_path = os.path.join(folder, '{0}_layer{1}.tif'.format(z, i))
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


def geotiff_to_array(file):
    ds = gdal.Open(file)
    b = ds.GetRasterBand(1)
    arr = b.ReadAsArray(b)
    return arr


def display(file):
    ds = gdal.Open(file)
    array = ds.GetRasterBand(1).ReadAsArray()
    plt.figure()
    plt.imshow(array)
    plt.colorbar()
    plt.show()
    ds = None


def avg(matrix):
    matrix_sum = 0
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            matrix_sum += matrix[i][j]
    matrix_avg = matrix_sum / len(matrix)**2
    return matrix_avg


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
