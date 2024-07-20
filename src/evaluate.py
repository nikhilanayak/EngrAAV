import pandas as pd
import data_loader
import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.metrics import f1_score
from scipy.stats import pearsonr
from tensorflow.keras import backend as K


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def run(model, path="data/holdoutfinal.csv"):
    x, y = data_loader.get_data(path, augment=False, mutation=None)

    out = model.evaluate(x, y)

    preds = model.predict(x)

    return out, preds, y
    
model = keras.models.load_model("model")

out, preds, y = run(model)

for index, name in enumerate(model.metrics_names):
    metric = out[index]
    print(name, ":", metric)

cf = tf.math.confusion_matrix(y.argmax(axis=1), preds.argmax(axis=1)).numpy()


TN = cf[0][0]
TP = cf[1][1]

FP = cf[0][1]
FN = cf[1][0]

print("True Positive:", TP)
print("True Negative:", TN)

print("False Positive:", FP)
print("False Negative:", FN)

f1 = f1_score(y.argmax(axis=1), preds.argmax(axis=1))

print("F1", f1)