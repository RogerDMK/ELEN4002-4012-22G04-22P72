from tkinter import Image
import numpy as np
from PIL import Image

def test_mat(side_length):
    """
    Generate test matrix to test scripts.
    """
    sizes = (side_length, side_length)
    test_mat = np.zeros(sizes)
    for x in range(6):
        for y in range(6):
            test_mat[x,y+6] = 1
            test_mat[x+6,y] = 2
            test_mat[x+6,y+6] = 3
    return (test_mat)

def blockshaped(arr, nrows, ncols):
    """
    Creates a 3d matrix, comprising of nrows by ncols 2d-matrices ordered along the 3rd axis.
    The 3d matrix is constructed from arr by splitting the original array into nrow by ncol blocks.
    """
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def reformat(reshaped_mat):
    """
    Returns the original matrix from the segmented grids.
    """
    size = (1200,1200)
    listed_mats = [[reshaped_mat[j*3 + i,:,:] for i in range(size[0]//reshaped_mat.shape[1])] for j in range(size[1]//reshaped_mat.shape[2])]
    return(np.block(listed_mats).reshape((12,12)))

def average_grid(mat):
    """
    Gets the average value of a reshaped matrix.
    """
    sum = np.mean(mat, axis=((1,2)))
    return(sum)

def ta_conversion(matrix):
    """
    Return the thermal anomaly grided matrix.
    """
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if matrix[i][j] == 3 or matrix[i][j] == 5:
                matrix[i][j] = 0
            elif matrix[i][j] >= 7:
                matrix[i][j] = 1
            elif matrix[i][j] == 4:
                matrix[i][j] = 0
            else:
                matrix[i][j] = 0
    return matrix

def average_ta(mat):
    """
    Returns whether there is a fire or not in a block.
    """
    sum = np.mean(mat, axis=((1,2)))
    # number_of_fires = 0
    for i in range(len(sum)):
        if sum[i] > 0.4:
            sum[i] = 1
            # number_of_fires += 1
        else:
            sum[i] = 0
    # print('Image: ' + str(image_number) + ' number of fires: ' + str(number_of_fires))
    return(sum)

def load_image(image_location):
    """
    Create an array from an input tiff image, image_location is a string showing the path to the source image.
    """
    image = Image.open(image_location)
    out_array = np.array(image)
    return out_array

def reformat_pred(predictions):
    """
    Reformat the predictions in the format of the original image.
    """
    output = np.zeros((300, 300))
    for i in range(300):
        for j in range(300):
            output[i,j] = predictions[i*300+j, 1]
    return(output)

def combine_all_q(q1, q2, q3, q4):
    """
    Combine all quadrants into a single matrix.
    """
    output = np.zeros((600, 600))
    for i in range(600):
        for j in range(600):
            if i < 300 and j < 300:
                output[i,j] = q2[i*300+j, 1]
            elif i < 300 and j > 299:
                output[i,j] = q1[i*300 + j - 300, 1]
            elif i > 299 and j < 300:
                output[i,j] = q3[(i - 300)*300 + j, 1]
            else:
                output[i,j] = q4[(i - 300)*300 + j - 300, 1]
    return(output)