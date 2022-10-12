import numpy as np

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
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
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
    size = (12,12)
    listed_mats = [[reshaped_mat[j*3 + i,:,:] for i in range(size[0]//reshaped_mat.shape[1])] for j in range(size[1]//reshaped_mat.shape[2])]
    return(np.block(listed_mats).reshape((12,12)))

def average_grid(mat):
    """
    Gets the average value of a reshaped matrix.
    """
    sum = np.mean(mat, axis=((1,2)))
    return(sum)
