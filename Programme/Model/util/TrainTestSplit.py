import numpy as np

def train_test_split(X, y, test_size=0.2, shuffle=True):
    """
    Splits the dataset into training and testing sets using numpy arrays.

    :param X: Numpy array of features.
    :param y: Numpy array of labels.
    :param test_size: Float representing the proportion of the dataset to include in the test split.
    :param shuffle: Boolean indicating whether to shuffle the data before splitting (default True).
    :return: Four numpy arrays: X_train, X_test, y_train, y_test
    """
    # Combine X and y to shuffle them together
    data = np.column_stack((X, y))

    if shuffle:
        np.random.shuffle(data)  # Shuffle the combined data

    # Calculate the index at which to split the data
    split_index = int(len(data) * (1 - test_size))

    # Split the data into training and testing sets
    train_data = data[:split_index]
    test_data = data[split_index:]

    # Separate the features (X) and labels (y)
    X_train = train_data[:, :-1]  # All columns except the last
    y_train = train_data[:, -1]   # Last column
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1]

    return X_train, X_test, y_train, y_test