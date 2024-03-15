import json
import pickle
import numpy as np
import gc
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers.experimental import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adamax
import yaml
from keras.utils import plot_model
from tensorflow.keras import metrics
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, GlobalAveragePooling2D, Lambda, Resizing, Rescaling
from tensorflow.keras.layers import Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from sklearn.utils.class_weight import compute_class_weight
import os
import argparse
import sys
import mlflow
import yaml
from mlflow.models.signature import infer_signature
from tensorflow import keras
import keras_tuner as kt
import argparse
import sqlite3
from numba import cuda

sqliteConnection = sqlite3.connect('data.db')

physical_devices = tf.config.list_physical_devices('GPU')


"""
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)
"""

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


train_dir = os.path.join('fer2013_pics', 'train')
test_dir =  os.path.join('fer2013_pics', 'test')
tf.random.set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--base_model', type=str, required=True)
args = parser.parse_args()

expected_res = (48, 48, 3)
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(expected_res[0], expected_res[1]),
  batch_size=batch_size,
  label_mode="categorical")


val_ds = tf.keras.utils.image_dataset_from_directory(
  test_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(expected_res[0], expected_res[1]),
  batch_size=batch_size,
  label_mode="categorical")

num_classes = 7


AUTOTUNE = tf.data.AUTOTUNE
def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds


train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)


inputs = tf.keras.Input(shape=expected_res)
if args.base_model == 'RegNetY080' or args.base_model == 'RegNetX080':
    x = tf.keras.applications.regnet.preprocess_input(inputs)
    if args.base_model == 'RegNetY080':
        base_model = tf.keras.applications.regnet.RegNetY064(weights='imagenet', include_top=False, input_shape=expected_res)
    else:
        base_model = tf.keras.applications.regnet.RegNetX064(weights='imagenet', include_top=False, input_shape=expected_res)

elif args.base_model == 'EfficientNetV2B3':
    x = tf.keras.applications.efficientnet_v2.preprocess_input(inputs)
    base_model = tf.keras.applications.EfficientNetV2B3(weights='imagenet', include_top=False, input_shape=expected_res)

elif args.base_model == 'MobilenetV3-Small' or args.base_model == 'MobilenetV3-Large':
    x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
    if args.base_model == 'MobilenetV3-Small':
        base_model = tf.keras.applications.MobileNetV3Small(weights='imagenet', include_top=False, input_shape=expected_res)
    else:
        base_model = tf.keras.applications.MobileNetV3Large(weights='imagenet', include_top=False, input_shape=expected_res)


x = tf.keras.layers.RandomRotation(
                                factor=0.20,
                                seed=100)(x)
x = tf.keras.layers.RandomFlip(
                                mode="horizontal",
                                seed=100)(x)
x = base_model(x)
x = Flatten()(x)

base_model.trainable = False
def model_builder(hp):
    global x
    print(x)
    hp_activation = hp.Choice('activation_function', values=['LeakyReLU', 'ELU', 'PReLU', 'SELU', 'GELU', 'Swish'])
    hp_hidden_units = hp.Choice('hidden_units', values=[256, 512, 1024, 2024])
    hp_layer_num = hp.Int('hidden_layers_num', min_value=2, max_value=8, step=2)
    hp_lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
   


    if hp_activation == 'LeakyReLU':
        nonlinear_activation=tf.keras.layers.LeakyReLU()
    elif hp_activation == 'ELU':
        nonlinear_activation=tf.keras.layers.ELU()
    elif hp_activation == 'PReLU':
        nonlinear_activation=tf.keras.layers.PReLU()
    elif hp_activation == 'SELU':
        nonlinear_activation='selu'
    elif hp_activation == 'GELU':
        nonlinear_activation= tf.keras.activations.gelu
    elif hp_activation == 'Swish':
        nonlinear_activation= tf.keras.activations.swish


    for i in range(1, hp_layer_num+1):
        #[1, 2, 3]
        print(i)
        if i  == 1:
            hidden_layer = BatchNormalization(name=f'batch_normalization_n{i}')(x)
        else:
            hidden_layer = BatchNormalization(name=f'batch_normalization_n{i}')(hidden_layer)

        hidden_layer = Dense(hp_hidden_units, activation=nonlinear_activation, kernel_regularizer=regularizers.l2(0.001))(hidden_layer)
        hidden_layer = Dropout(0.5)(hidden_layer)

    outputs = Dense(num_classes, activation='softmax')(hidden_layer)
    model = Model(inputs, outputs)

    
    


    opt = Adam(learning_rate=hp_lr)
    
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', tf.keras.metrics.F1Score(average="macro"), tf.keras.metrics.AUC(multi_label=True, num_labels=7)])
    del hidden_layer
    gc.collect()

    return model



""""
tuner = kt.Hyperband(
    model_builder,
    objective=[kt.Objective("val_auc", direction="max"), kt.Objective('val_accuracy', direction="max"), 
               kt.Objective('val_loss', direction="min"), kt.Objective('val_f1_score', direction="max")],
    hyperband_iterations=1,
    max_epochs=max_epoch_num,
    factor=5, 
    directory='FER2013_TrainedNetworks',
    project_name=f'{args.base_model}__Models',
    overwrite=True
    )


"""
tuner = kt.BayesianOptimization(
    model_builder,
    objective=[kt.Objective("val_auc", direction="max"), kt.Objective('val_accuracy', direction="max"), 
               kt.Objective('val_loss', direction="min"), kt.Objective('val_f1_score', direction="max")],
    max_trials=10,
    directory='FER2013_TrainedNetworks',
    project_name=f'{args.base_model}__Models'

)


class_weights  = {0: 3.62304392, 1: 32.77283105, 2: 3.50366122, 
                3: 1.99617578, 4: 2.95299321, 
                5: 4.48297939, 6: 2.89521985} 


stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
print("Fitting model")
tuner.search(train_ds, validation_data=val_ds, use_multiprocessing=True, workers=8, epochs=5, class_weight=class_weights, validation_split=0.2, callbacks=[stop_early])
print("Finished")
# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
print(best_hps)

print(f"""The hyperparameter search is complete for a neural network using {str(args.base_model)}
        - The optimal number of units for each hidden layers is {str(best_hps.get('hidden_units'))} 
        - The best number of hidden layers is {str(best_hps.get('hidden_layers_num'))}
        - The best activation function is {best_hps.get('activation_function')}
        - The optimal learning rate for the optimizer is {str(best_hps.get('learning_rate'))}
        """)

cursor = sqliteConnection.cursor()
print("\nSucessfully Connected to the database...\n")



cursor.execute("INSERT INTO optimal_hyp VALUES (?, ?, ?, ?, ?);", (args.base_model, best_hps.get('activation_function'), 
                                                                   best_hps.get('hidden_units'), best_hps.get('hidden_layers_num'),
                                                                   best_hps.get('learning_rate')
                                                                   ))
 

print("\nInserted Data...\n")
sqliteConnection.commit()
 
# close the connection
sqliteConnection.close()


device = cuda.get_current_device()
device.reset()  # This will release all GPU memory