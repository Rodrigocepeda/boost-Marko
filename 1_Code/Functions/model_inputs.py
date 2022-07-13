from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, Flatten, MaxPooling3D, AveragePooling3D, ConvLSTM2D, Reshape, Dropout, BatchNormalization
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
import tensorflow as tf

def train_test_val_split(x, y, train_size, val_size, test_size, random_state, shuffle):
    """
    Documentar
    """
    x_train, x_rem, y_train, y_rem = train_test_split(
        x, y, test_size = (val_size + test_size), random_state = random_state, shuffle = shuffle)
    x_val, x_test, y_val, y_test = train_test_split(
        x_rem, y_rem, test_size = test_size / (val_size + test_size),
        random_state = random_state, shuffle = shuffle)
    return x_train, x_val, x_test, y_train, y_val, y_test
