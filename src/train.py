#from bert import BertModelLayer
from tensorflow import keras
import pandas as pd
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from decay import WarmUpCosineDecayScheduler
from data_loader import get_data
import data_loader
import argparse
from tensorflow.keras import backend as K
import decay
import wandb

tf.compat.v1.disable_eager_execution()


parser = argparse.ArgumentParser()
parser.add_argument("--tune_hyperparams", help="runs hyperparameter tuning for transformer", action="store_true")
parser.add_argument("--epochs", type=int, default=20, help="number of train epochs")
parser.add_argument("--batch_size", type=int, default=256, help="train batch size")
parser.add_argument("--warmup_steps", type=int, default=128, help="number of warmup steps for cosine decay learning rate regularization")
parser.add_argument("--train_dataset", type=str, default="data/c1r2train.csv")
parser.add_argument("--eval_dataset", type=str, default="data/holdoutfinal.csv")

args = parser.parse_args()

wandb.init()

def repeat(l, n):
    return [l for i in range(n)]

X_train, Y_train = get_data("data/everything.csv") # Loads the train dataset without any data augmentation
X_val, Y_val = get_data("data/holdoutfinal.csv") # Loads the eval dataset without any data augmentation

STEPS = len(X_train) // args.batch_size * args.epochs

def sequence_model(hp: kt.HyperParameters = None): # Main Transformer Model

    def make_lstm(units, return_sequences, bidirectional):
        layer = keras.layers.LSTM(units, return_sequences=return_sequences)
        if bidirectional:
            layer = keras.layers.Bidirectional(layer)
        return layer

    if hp == None:
        embedding_dim = 26
        lstm_units = 80 # 96
        lr = 1e-3
        num_lstms = 4 # 1
        dropout = 0.85
        bidirectional = False
    else:
        embedding_dim = hp.Int("embedding_dim", min_value=1, max_value=len(data_loader.uniq), step=1)
        lstm_units = hp.Int("lstm_units", min_value=1, max_value=128, step=16)
        num_lstms = hp.Int("num_lstms", min_value=1, max_value=8, step=1)
        lr = hp.Choice("lr", values=[1e-10, 1e-9, 1e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 1e-4, 1e-3, 1e-2])
        dropout = hp.Float("dropout", min_value=0.1, max_value=0.9, step=0.1)
        bidirectional = hp.Choice("bidirectional", values=[False, True])

    inputs = keras.layers.Input(shape=(40,))
    x = keras.layers.Embedding(input_dim=len(data_loader.uniq), output_dim=embedding_dim, input_length=data_loader.MAX_SEQ_LEN)(inputs)

    for i in range(num_lstms - 1):
        #lstm_layer = keras.layers.LSTM(lstm_units, return_sequences=True)
        x = make_lstm(lstm_units, return_sequences=True, bidirectional=bidirectional)(x)
        x = keras.layers.Dropout(dropout)(x)
    

    x = make_lstm(lstm_units, return_sequences=False, bidirectional=bidirectional)(x)
    outputs = keras.layers.Dense(2, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    auc = tf.keras.metrics.AUC() # use auc for validation
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=[auc, "accuracy"])

    return model


def val(x, y, model, name=None):
    if not name:
        name = "unknown"
    mets = model.evaluate(x, y)
    met_names = model.metrics_names

    val_auc = mets[met_names.index("auc")]
    val_acc = mets[met_names.index("accuracy")]
    
    predy = model.predict(x)

    cf = confusion_matrix(y.argmax(axis=1), predy.argmax(axis=1))


    TN = cf[0][0]
    TP = cf[1][1]

    FP = cf[0][1]
    FN = cf[1][0]

    print(f"\nValidation Summary for {name}")
    print("\tEvaluation AUC: ", val_auc)
    print("\tEvaluation Accuracy: ", val_acc)
    print("True Positive:", TP)
    print("True Negative:", TN)
    print("False Positive:", FP)
    print("False Negative:", FN)
    print("="*20)

class EvalCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        val(X_val, Y_val, self.model, name="holdoutfinal")

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=3, mode="max")

model = sequence_model()

if args.tune_hyperparams:
    tuner = kt.Hyperband(sequence_model, objective=kt.Objective("val_auc", direction="max"), max_epochs=args.epochs, factor=3, project_name="oracle")
    tuner.search(X_train, Y_train, epochs=args.epochs * 5, validation_data=(X_train, Y_train), callbacks=[stop_early], batch_size=args.batch_size)
else:
    schedule = decay.WarmUpCosineDecayScheduler(learning_rate_base=1e-3, total_steps=STEPS, warmup_steps=10000, wandb=wandb)
    model.fit(x=X_train, y=Y_train, epochs=args.epochs, validation_data=(X_val, Y_val), callbacks=[stop_early, EvalCallback(), schedule], batch_size=args.batch_size)
    model.save("model")
