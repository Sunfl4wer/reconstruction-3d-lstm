import numpy as np

def split_data(dataset):
    num_samples = len(dataset)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    train_split = int(0.8 * num_samples)
    val_split = int(0.9 * num_samples)

    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]

    X_train = dataset[train_indices]
    X_val = dataset[val_indices]
    X_test = dataset[test_indices]

    return X_train, X_val, X_test