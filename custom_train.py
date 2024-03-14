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

#keras.mixed_precision.set_global_policy("mixed_float16")
dir_path = os.path.dirname(os.path.realpath(__file__))

# List available physical GPUs
physical_devices = tf.config.list_physical_devices('GPU')

# Set memory growth for all GPUs
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, required=True)


parser.add_argument('--base_model', type=str, default='RegNet-Accuracy')
parser.add_argument('--hidden_units', type=int, default=1024)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs_num', type=int, default=150)
parser.add_argument('--activation_function', type=str, default='elu')
parser.add_argument('--num_hidden_layers', type=int, default=5)



args = parser.parse_args()
print(args.epochs_num)
if args.activation_function == 'gelu':
    nonlinear_activation = tf.keras.activations.gelu
elif args.activation_function == 'leaky-relu':
    nonlinear_activation = tf.keras.layers.LeakyReLU()
elif args.activation_function == 'elu':
    nonlinear_activation = tf.keras.layers.ELU()
elif args.activation_function == 'p-relu':
    nonlinear_activation = tf.keras.layers.PReLU()
elif args.activation_function == 'selu':
    nonlinear_activation = 'selu'
elif args.activation_function == 'swish':
    nonlinear_activation = tf.keras.activations.swish

try:
    mlflow.create_experiment("Fer2013-Training")
    experiment = mlflow.get_experiment_by_name("Fer2013-Training")
except:
    experiment = mlflow.set_experiment('Fer2013-Training')

mlflow.keras.autolog(
    log_models=True
)

train_dir = os.path.join('fer2013_pics', 'train')
test_dir =  os.path.join('fer2013_pics', 'test')

expected_res = (48, 48, 3)

train_ds = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(expected_res[0], expected_res[1]),
  batch_size=args.batch_size,
  label_mode="categorical")


val_ds = tf.keras.utils.image_dataset_from_directory(
  test_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(expected_res[0], expected_res[1]),
  batch_size=args.batch_size,
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

print(args.base_model)
if args.base_model == 'RegNetY080':
    #Speed
    base_model = tf.keras.applications.regnet.RegNetY080(weights='imagenet', include_top=False, input_shape=expected_res)
elif args.base_model == 'VGG16':
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=expected_res)
elif args.base_model == 'MobilenetV3-Small':
    base_model = tf.keras.applications.MobileNetV3Small(weights='imagenet', include_top=False, input_shape=expected_res)
elif args.base_model == 'MobilenetV2':
    base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False, input_shape=expected_res)
elif args.base_model == 'MobilenetV3-Large':
    base_model = tf.keras.applications.MobileNetV3Large(weights='imagenet', include_top=False, input_shape=expected_res)
elif args.base_model == 'EfficientNetV2B3':
    base_model = tf.keras.applications.EfficientNetV2B3(weights='imagenet', include_top=False, input_shape=expected_res)
elif args.base_model == 'RegNetX080':
    #Accuracy
    base_model = tf.keras.applications.regnet.RegNetX080(weights='imagenet', include_top=False, input_shape=expected_res)


for layer in base_model.layers:
    layer.trainable = False

base_model.trainable = False


tf.random.set_seed(42)


with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=f'{args.model_name} ', tags={"Activation Function": args.activation_function, "Base Model": args.base_model}) as run:
    run_id = run.info.run_id

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
    x = GlobalAveragePooling2D()(x)

    for i in range(1, args.num_hidden_layers+1):
        #[1, 2, 3]
        x = BatchNormalization()(x)

        """
        if i != args.num_hidden_layers: #If its not the last hidden layer
            x = Dense(args.hidden_units, activation=nonlinear_activation, kernel_regularizer=regularizers.l2(0.1))(x)
        else: #if it is
            if args.num_hidden_layers == 1: #If we only have 1 layer, keep the inputted size
                x = Dense(args.hidden_units, activation=nonlinear_activation, kernel_regularizer=regularizers.l2(0.1))(x)
            else: #If not, half it.
                x = Dense(args.hidden_units/2, activation=nonlinear_activation, kernel_regularizer=regularizers.l2(0.1))(x)
        """

        x = Dense(args.hidden_units, activation=nonlinear_activation, kernel_regularizer=regularizers.l2(0.001))(x)
        x = Dropout(0.5)(x)
   
    x = BatchNormalization()(x)
    outputs = Dense(num_classes, activation='softmax')(x)


    model = Model(inputs, outputs)

    model.summary(show_trainable=True)
    #plot_model(model, show_shapes=True, show_layer_names=True)
    print("\n\n\n\n\n\n\n\n\n")
    
 

    print("Training Generator")
   

  
    

    
    #early_stopping = EarlyStopping(patience=15, restore_best_weights=False)

   

  
    
    ## Now train the model
    batch_size = args.batch_size
    epochs = args.epochs_num
    steps_per_epoch = 28709 // args.batch_size

    base_lr = 0.001
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        base_lr,
        decay_steps=steps_per_epoch*20,
        decay_rate=1,
        staircase=False)
   

    
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
            use_multiprocessing = True,
            workers=8
        )
    
    

    mlflow.log_metric("Batch Size", args.batch_size)
    mlflow.log_metric("Hidden Units", args.hidden_units)
    mlflow.log_metric("Num of Epoch", args.epochs_num)
   
   
   
    """
    #mlflow.keras.log_model(model, registered_model_name=f"{args.model_name}", signature=signature)

    
    print("About to save model")
    mlflow.keras.save_model(model, f'models/{args.model_name}', signature=signature)
    """
    
print("\n\n\n\n----------------------------------------------------\nNEXT MODEL IS BEING TRAINED\n\n\n\n\n\n\n\n\n\n\n-----------------------------------------------------------\n\n\n\n")

with open(os.path.join(dir_path, 'mlruns', experiment.experiment_id, 'meta.yaml'), 'r') as file:
    experiment_yaml = yaml.safe_load(file)


experiment_yaml["experiment_id"] = "Fer2013-Experiment"
experiment_yaml['artifact_location'] = 'file://' + os.path.join(dir_path, 'mlruns', "Fer2013-Experiment")

print(experiment_yaml)

os.rename(os.path.join('mlruns', experiment.experiment_id), os.path.join('mlruns', 'Fer2013-Experiment'))
with open(os.path.join(dir_path, 'mlruns', 'Fer2013-Experiment', 'meta.yaml'), 'w') as file:
    yaml.dump(experiment_yaml, file)





with open(os.path.join(dir_path, 'mlruns', "Fer2013-Experiment", run_id, 'meta.yaml'), 'r') as file:
    run_yaml = yaml.safe_load(file)

run_yaml['artifact_uri'] = 'file://' + os.path.join(dir_path, 'mlruns', "Fer2013-Experiment", f'{args.model_name} ', 'artifacts')
run_yaml['experiment_id'] = "Fer2013-Experiment"
run_yaml['run_id'] = f'{args.model_name}'

print(run_yaml)

os.rename(os.path.join('mlruns', 'Fer2013-Experiment', run_id), os.path.join('mlruns', 'Fer2013-Experiment', f'{args.model_name}'))
with open(os.path.join(dir_path, 'mlruns', 'Fer2013-Experiment', f'{args.model_name}', 'meta.yaml'), 'w') as file:
    yaml.dump(run_yaml, file)





