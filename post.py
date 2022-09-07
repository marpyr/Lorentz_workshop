from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

import tensorflow as tf    
#tf.compat.v1.disable_v2_behavior() # <-- HERE !

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout, Activation, Reshape, Flatten, LSTM, Dense, Dropout, Embedding, Bidirectional, GRU
from tensorflow.keras import Sequential
from tensorflow.keras import initializers, regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import constraints
from tensorflow.keras.layers import Layer, InputSpec

def load_weight(weight_file, trained_model, test_X, test_y):
    model = trained_model
    model.load_weights(weight_file)
    # Evaluate the model
    val1, val2 = model.evaluate(test_X, test_y, verbose=2)   # val1 is loss, and val2 could be another metrics defined during the training
    y_pred = model.predict(test_X)
    return y_pred, val1, val2

def score(y_truth, y_pred):
    recall = recall_score(y_truth, y_pred, nbins)
    precision = precision_score(y_truth, y_pred)
    f1 = f1_score(y_truth, y_pred)
    acc = accuracy_score(y_truth, y_pred)
    BS = brier_score_loss(y_truth, y_pred)
    # BSS 
    calib_y, calib_x = calibration_curve(y_truth, y_pred, n_bins=nbins)
    return reccall, precision, f1, acc, BS, calib_y, calib_x
