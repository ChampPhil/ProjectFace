import json
from numba import cuda
import pickle
import numpy as np
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
import sqlite3

parser = argparse.ArgumentParser()
parser.add_argument('--base_model', type=str, required=True)
args = parser.parse_args()


print(args.base_model)
#keras.mixed_precision.set_global_policy("mixed_float16")
sqliteConnection = sqlite3.connect('data.db')
cursor = sqliteConnection.cursor()

cursor.execute("SELECT * FROM optimal_hyp WHERE BaseNetwork = ?", (args.base_model,))
 
hyperparameter_set = cursor.fetchone()
print(hyperparameter_set)


dir_path = os.path.dirname(os.path.realpath(__file__))

# List available physical GPUs
physical_devices = tf.config.list_physical_devices('GPU')

# Set memory growth for all GPUs
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)



if hyperparameter_set[1] == 'GELU':
    nonlinear_activation = tf.keras.activations.gelu
elif hyperparameter_set[1] == 'LeakyReLU':
    nonlinear_activation = tf.keras.layers.LeakyReLU()
elif hyperparameter_set[1] == 'ELU':
    nonlinear_activation = tf.keras.layers.ELU()
elif hyperparameter_set[1] == 'PReLU':
    nonlinear_activation = tf.keras.layers.PReLU()
elif hyperparameter_set[1] == 'SELU':
    nonlinear_activation = 'selu'
elif hyperparameter_set[1] == 'Swish':
    nonlinear_activation = tf.keras.activations.swish


train_dir = os.path.join('fer2013_pics', 'train')
test_dir =  os.path.join('fer2013_pics', 'test')

expected_res = (48, 48, 3)
batch_size=32

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
print(train_ds)

if hyperparameter_set[0] == 'RegNetY080':
    #Speed
    base_model = tf.keras.applications.regnet.RegNetY064(weights='imagenet', include_top=False, input_shape=expected_res)
elif hyperparameter_set[0] == 'MobilenetV3-Small':
    base_model = tf.keras.applications.MobileNetV3Small(weights='imagenet', include_top=False, input_shape=expected_res)
elif  hyperparameter_set[0] == 'MobilenetV3-Large':
    base_model = tf.keras.applications.MobileNetV3Large(weights='imagenet', include_top=False, input_shape=expected_res)
elif  hyperparameter_set[0] == 'EfficientNetV2B3':
    base_model = tf.keras.applications.EfficientNetV2B3(weights='imagenet', include_top=False, input_shape=expected_res)
elif  hyperparameter_set[0] == 'RegNetX080':
    #Accuracy
    base_model = tf.keras.applications.regnet.RegNetX064(weights='imagenet', include_top=False, input_shape=expected_res)


for layer in base_model.layers:
    layer.trainable = False

base_model.trainable = False


tf.random.set_seed(42)




inputs = tf.keras.Input(shape=expected_res)



if args.base_model == 'RegNetY080' or args.base_model == 'RegNetX080':
    x = tf.keras.applications.regnet.preprocess_input(inputs)
elif args.base_model == 'VGG16':
    x = tf.keras.applications.vgg16.preprocess_input(inputs)
elif args.base_model == 'EfficientNetV2B3':
    x = tf.keras.applications.efficientnet_v2.preprocess_input(inputs)
elif args.base_model == 'MobilenetV3-Small' or args.base_model == 'MobilenetV3-Large':
    x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
elif args.base_model == 'MobilenetV2':
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

x = tf.keras.layers.RandomRotation(
                                factor=0.20,
                                seed=100)(x)
x = tf.keras.layers.RandomFlip(
                                mode="horizontal",
                                seed=100)(x)

x = base_model(x)
x = Flatten()(x)

for i in range(1, hyperparameter_set[3]+1):
    #[1, 2, 3]
    x = BatchNormalization()(x)
    x = Dense(hyperparameter_set[2], activation=nonlinear_activation, kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dropout(0.5)(x)

x = BatchNormalization()(x)
outputs = Dense(num_classes, activation='softmax')(x)


model = Model(inputs, outputs)

model.summary(show_trainable=True)
#plot_model(model, show_shapes=True, show_layer_names=True)
print("\n\n\n\n\n\n\n\n\n")

early_stopping = EarlyStopping(patience=15, restore_best_weights=False)
## Now train the model

epochs = 225
steps_per_epoch = 28709 // batch_size

base_lr = hyperparameter_set[4]

opt = Adam(learning_rate=base_lr)
#opt = SGD(learning_rate=base_lr, nesterov=True, momentum=0.9)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', tf.keras.metrics.F1Score(average="macro"), tf.keras.metrics.AUC(multi_label=True, num_labels=7)])


class_weights  ={0: 3.62304392, 1: 32.77283105, 2: 3.50366122, 
                    3: 1.99617578, 4: 2.95299321, 
                    5: 4.48297939, 6: 2.89521985} 
print("Fitting model")
model.fit(train_ds,
        batch_size = batch_size,
        validation_data = val_ds,
        epochs = epochs,
        class_weight=class_weights,
        callbacks=[early_stopping],
        use_multiprocessing = True,
        workers=8
    )

print("Evaluating Model...")
test_loss, test_accuracy, test_f1score, test_auc = model.evaluate(val_ds, verbose=1)

print("Saving Model...")


if not os.path.exists("models"):
    os.mkdir("models")

model.save(f'models/{hyperparameter_set[0]}__Classifier.keras')  # The file needs to end with the .keras extension

print(f"""The training is complete for a neural network using {str(hyperparameter_set[0])} as its base
        - The loss on the testing dataset is {str(test_loss)} 
        - The accuracy on the testing dataset is {str(test_accuracy)}
        - The AUC Curve on the testing datset is {str(test_auc)}
        - The F1Score on the testing dataset is {str(test_f1score)}
        """)
cursor.execute("INSERT INTO model_metrics VALUES (?, ?, ?, ?, ?);", (hyperparameter_set[0], test_accuracy, test_f1score, test_loss, test_auc))
 

print("\nInserted Metrics Into SQL Table...\n")
sqliteConnection.commit()
 
# close the connection
sqliteConnection.close()



print("\n\n\n\n----------------------------------------------------\nMODEL IS DONE BEING TRAINED\n\n\n\n\n\n\n\n\n\n\n-----------------------------------------------------------\n\n\n\n")


device = cuda.get_current_device()
device.reset()  # This will release all GPU memory











