import json

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

# List available physical GPUs
physical_devices = tf.config.list_physical_devices('GPU')

# Set memory growth for all GPUs
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)



parser = argparse.ArgumentParser()
parser.add_argument('--base_model', type=str, required=True)
args = parser.parse_args()


print(args.base_model)
expected_res = (48, 48, 3)

if args.base_model == 'RegNetY064':
    #Speed
    base_model = tf.keras.applications.regnet.RegNetY064(weights='imagenet', include_top=False, input_shape=expected_res)
elif args.base_model == 'MobilenetV3-Small':
    base_model = tf.keras.applications.MobileNetV3Small(weights='imagenet', include_top=False, input_shape=expected_res)
elif  args.base_model == 'MobilenetV3-Large':
    base_model = tf.keras.applications.MobileNetV3Large(weights='imagenet', include_top=False, input_shape=expected_res)
elif  args.base_model == 'EfficientNetV2B3':
    base_model = tf.keras.applications.EfficientNetV2B3(weights='imagenet', include_top=False, input_shape=expected_res)
elif  args.base_model == 'RegNetX064':
    #Accuracy
    base_model = tf.keras.applications.regnet.RegNetX064(weights='imagenet', include_top=False, input_shape=expected_res)

#keras.mixed_precision.set_global_policy("mixed_float16")
sqliteConnection = sqlite3.connect('data.db')
cursor = sqliteConnection.cursor()

cursor.execute("SELECT * FROM optimal_hyp WHERE BaseNetwork = ?", (args.base_model,))
 
hyperparameter_set = cursor.fetchone()
print(hyperparameter_set)


dir_path = os.path.dirname(os.path.realpath(__file__))



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


model_path = None


model = keras.models.load_model(os.path.join('models', f'{args.base_model}__Classifier.keras'))

print("Number of layers in the base model: ", len(model.layers))



# Freeze all the layers before the `fine_tune_at` layer
model.trainable = True

for layer in model.layers:
  layer.trainable = True


class_weights  ={0: 3.62304392, 1: 32.77283105, 2: 3.50366122, 
                    3: 1.99617578, 4: 2.95299321, 
                    5: 4.48297939, 6: 2.89521985} 

early_stopping = EarlyStopping(patience=10, restore_best_weights=False)

opt = Adam(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy', 
              optimizer=opt, 
              metrics=['accuracy', tf.keras.metrics.F1Score(average="macro"), tf.keras.metrics.AUC(multi_label=True, num_labels=7)])

model.fit(train_ds, 
          epochs=100, 
          validation_data=val_ds,
          batch_size = batch_size,
          class_weight=class_weights,
          callbacks=[early_stopping],)

print("Evaluating Model...")
test_loss, test_accuracy, test_f1score, test_auc = model.evaluate(val_ds, verbose=1)

print("Saving Model...")




model.save(f'best_network.keras')  # The file needs to end with the .keras extension

print(f"""The fine-tuning is complete for the best Neural Network (which is the one that uses {str(hyperparameter_set[0])} as its base)
        - The loss on the testing dataset is {str(test_loss)} 
        - The accuracy on the testing dataset is {str(test_accuracy)}
        - The AUC Curve on the testing datset is {str(test_auc)}
        - The F1Score on the testing dataset is {str(test_f1score)}
        """)
cursor.execute("INSERT INTO fine_tuned VALUES (?, ?, ?, ?, ?);", (hyperparameter_set[0], test_accuracy, test_f1score, test_loss, test_auc))
 

print("\nInserted Metrics Into SQL Table...\n")
sqliteConnection.commit()
 
# close the connection
sqliteConnection.close()



print("\n\n\n\n----------------------------------------------------\nFINE TUNING FINISHED\n\n\n\n\n\n\n\n\n\n\n-----------------------------------------------------------\n\n\n\n")













