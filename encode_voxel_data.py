import numpy as np
import const
import compress

def slide_window_3d(matrix, window_size=(2, 2, 2)):
    """Slides a window over a 3D matrix and returns the windowed views.

    Args:
        matrix: The 3D matrix (NumPy array).
        window_size: A tuple (height, width, depth) specifying the size of the window.

    Returns:
        A list of windowed views of the matrix.
    """

    matrix_height, matrix_width, matrix_depth = matrix.shape
    window_height, window_width, window_depth = window_size

    windows = []
    for z in range(0, matrix_depth - window_depth + 1):
        for y in range(0, matrix_height - window_width + 1):
            for x in range(0, matrix_width - window_height + 1):
                window = matrix[x:x + window_height, y:y + window_width, z:z + window_depth]
                windows.append(window)

    return windows

    
def get_encoded_data_from_8bit(bit_array):
  """Gets the ASCII character from an 8-bit array.

  Args:
    bit_array: A list or tuple representing the 8-bit array.

  Returns:
    The ASCII character corresponding to the bit array.
  """
  if len(bit_array) != 8:
    raise ValueError("Input must be an 8-bit array.")

  # Convert the bit array to an integer
  bit_str = "".join(str(bit) for bit in bit_array)
  return const.ENCODING_2D[bit_str[:4]]+const.ENCODING_2D[bit_str[4:]]


def encode_voxel_data(arr_3d):
    windows = [w.flatten() for w in slide_window_3d(arr_3d, window_size=(2, 2, 2))]
    encoded_voxel = "".join([str(get_encoded_data_from_8bit(w)) for w in windows])
    compressed_image = compress.compress_string(encoded_voxel)
    return compressed_image


def read_voxel_data(f):
    data = np.genfromtxt(f, delimiter=',', skip_header=1, dtype=int)
    return np.unique(data, axis=0)


def encode_voxel_objects(objects):
    encoded_voxel_data = {}
    for o in objects:
        voxels = read_voxel_data(f'{const.VOXEL_DIRECTORY}/{o}.csv')
        expected = np.zeros((32, 32, 32), dtype=int)
        for voxel in voxels:
            if np.max(voxel) >= 32:
                continue
            expected[voxel[0]][voxel[1]][voxel[2]] = 1
        encoded_voxel_data[o] = encode_voxel_data(expected)
    return encoded_voxel_data