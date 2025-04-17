import numpy as np

def compress_string(data):
    arr = np.asarray([])
    if len(data) == 0:
        return arr
    current_char = data[0]
    char_count = 1
    for i in range(1, len(data)):
        c = data[i]
        if c == current_char:
            char_count += 1
            continue
        arr = np.append(arr, current_char)
        arr = np.append(arr, char_count)
        char_count = 1
        current_char = c
    arr = np.append(arr, current_char)
    arr = np.append(arr, char_count)
    return arr