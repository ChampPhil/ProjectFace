
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers.experimental import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adamax

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Conv2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from sklearn.utils.class_weight import compute_class_weight
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--base_model', type=str, required=True)
parser.add_argument('--hidden_units', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--epochs_num', type=int, default=30)


args = parser.parse_args()

print(os.listdir('dataset/processed'))
X_train = np.load('dataset/processed/X_train.npy')
# load data from y_train.npy in data/preprocessed directory
y_train = np.load('dataset/processed/y_train.npy')
# load data from X_test.npy in data/preprocessed directory
X_valid = np.load('dataset/processed/X_valid.npy')
# load data from y_test.npy in data/preprocessed directory
y_valid = np.load('dataset/processed/y_valid.npy')

img_width = X_train.shape[1]
img_height = X_train.shape[2]
img_depth = X_train.shape[3]
num_classes = y_train.shape[1] 

if args.base_model == 'VGG19':
    base_model = tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
elif args.base_model == 'VGG16':
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
elif args.base_model == 'Mobilenet':
    base_model = tf.keras.applications.mobilenet.MobileNet(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
elif args.base_model == 'MobilenetV2':
    base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
  
# Add dense layers
x = base_model.layers[-2].output
print(x.shape)
x = GlobalAveragePooling2D()(x)

# Add final classification layer
tf.random.set_seed(42)

output_layer = Dense(args.hidden_units, input_shape=(512,), activation='relu')(x)
output_layer = Dense(args.hidden_units, input_shape=(args.hidden_units,), activation='relu')(output_layer)
output_later = Dense(args.hidden_units, input_shape=(args.hidden_units, ), activation='relu')(output_layer)
output_layer = Dropout(0.5)(output_layer)
output_layer = Dense(int(args.hidden_units/2), input_shape=(int(args.hidden_units/2),), activation='relu')(output_layer)
output_layer = Dense(int(args.hidden_units/2), input_shape=(int(args.hidden_units/2),), activation='relu')(output_layer)
output_layer = Dense(num_classes, activation='softmax')(output_layer)

# Create model
model = Model(inputs=base_model.input, outputs=output_layer)

train_datagen = ImageDataGenerator(rotation_range=20,
    width_shift_range=0.20,
    height_shift_range=0.20,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest')

train_datagen.fit(X_train)

class_weights = compute_class_weight(
                                        class_weight = "balanced",
                                        classes = np.unique(y_train.argmax(axis=1)),
                                        y = y_train.argmax(axis=1)
                                    )
class_weights_dict = dict(enumerate(class_weights))

lr_scheduler = ReduceLROnPlateau(monitor = 'val_accuracy', 
                              factor = 0.25, 
                              patience = 8,
                              min_lr = 1e-6,
                              verbose = 1)

early_stopping = EarlyStopping(monitor = 'val_accuracy', 
                            min_delta = 0.00005, 
                            patience = 12,
                            verbose = 1, 
                            restore_best_weights = True)

callbacks = [lr_scheduler, early_stopping]
## Now train the model
batch_size = args.batch_size
epochs = args.epochs_num
steps_per_epoch = (int)(len(X_train) / batch_size)

opt = Adamax()

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

history = model.fit(train_datagen.flow(X_train,
                                       y_train,
                                       batch_size = batch_size),
                                       validation_data = (X_valid, y_valid),
                                       steps_per_epoch = steps_per_epoch,
                                       epochs = epochs,
                                       callbacks = callbacks,
                                       use_multiprocessing = True,
                                       class_weight=class_weights_dict)

model.save(f'models/{args.model_name}.keras')

