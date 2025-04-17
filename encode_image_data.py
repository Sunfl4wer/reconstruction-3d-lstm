from PIL import Image
import os
import const
import numpy as np
import compress

def slide_window(image, window_size, stride):
  """Slides a window over an image.

  Args:
    image: The 2D image as a NumPy array.
    window_size: A tuple (height, width) specifying the size of the window.
    stride: The number of pixels to move the window in each direction.

  Returns:
    A list of windowed views of the image.
  """

  image_height, image_width = image.shape
  window_height, window_width = window_size

  for y in range(0, image_height - window_height + 1, stride):
    for x in range(0, image_width - window_width + 1, stride):
      yield image[y:y + window_height, x:x + window_width]
      
def read_image_to_numpy(image_path):
  """Reads an image and converts it to a NumPy array.

  Args:
    image_path: The path to the image file.

  Returns:
    A NumPy array representing the image.
  """

  # Open the image using Pillow library
  image = Image.open(image_path).convert('L')

  # Convert the image to a NumPy array
  image_array = np.array(image)


  # Binarize the image using the threshold
  binary_image_matrix = (image_array > 125).astype(np.int_)

  return binary_image_matrix

def flatten_and_convert_to_string(matrix):
  """Flattens a NumPy matrix and converts it to a string.

  Args:
    matrix: The NumPy matrix.

  Returns:
    The flattened matrix as a string.
  """

  # Flatten the matrix
  flattened_matrix = matrix.flatten()

  # Convert the flattened matrix to a string
  string_representation = "".join(str(element) for element in flattened_matrix)

  return string_representation


def encode_image(image, window_size=(2,2), stride=1):
    windows = slide_window(image, window_size, stride)
    bit_str_arr = [flatten_and_convert_to_string(window) for window in windows]
    encoded_image = "".join([const.ENCODING_2D[bit_str] for bit_str in bit_str_arr])
    compressed_image = compress.compress_string(encoded_image)
    return compressed_image


def encode_image_objects(objects):
    encoded_image_data = {}
    for obj in objects:
        image_dir = os.path.join(const.IMAGE_DIRECTORY, obj)
        image_names = [f for f in os.listdir(image_dir) if f.endswith('.png') and not f.startswith('.')]
        image_paths = [os.path.join(image_dir, name) for name in image_names]
        encoded_images = {}
        for image_path in image_paths:
            image = read_image_to_numpy(image_path)
            encoded_images[image_path] = encode_image(image)
        encoded_image_data[obj] = encoded_images
    return encoded_image_data