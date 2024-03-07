
import pandas as pd
import numpy as np   
import json
import cv2
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def read_data(p):
   
    df = pd.read_csv(p)
    return df

def standardize_data(df):
   
    emotion_counts = df.emotion.value_counts()
    emotion_counts_dict = emotion_counts.to_dict()
    with open('./dataset/emotion_counts.json', 'w') as f:
        json.dump(emotion_counts_dict, f)

    #standardize the data to make compatible with model
    img_array = df.pixels.apply(lambda x: np.array(x.split(' ')).reshape(48, 48).astype('float32'))
    img_array = np.stack(img_array, axis = 0)

    return img_array

def get_features(img_array):
    
   
    img_features = []
    for i in range(len(img_array)):
        temp = cv2.cvtColor(img_array[i], cv2.COLOR_GRAY2RGB)
        img_features.append(temp)

    img_features = np.array(img_features)

    return img_features

def encode_features(df):
    
    # One-Hot Encoding of emotion column
    labels = df['emotion'].values
    encoded_labels = to_categorical(labels, num_classes=7)
    
    return encoded_labels

def split_save_data(img_features, img_labels, df):

    
    X_train, X_valid, y_train, y_valid = train_test_split(
        img_features, 
        img_labels, 
        shuffle=True, 
        stratify=df.emotion, 
        test_size=0.15, 
        random_state=42)

    # Normalize Data before saving it
    X_train = X_train / 255.
    X_valid = X_valid / 255.
    
    # Save the arrays to disk as .npy files
    np.save('./dataset/processed/X_train.npy', X_train)
    np.save('./dataset/processed/X_valid.npy', X_valid)
    np.save('./dataset/processed/y_train.npy', y_train)
    np.save('./dataset/processed/y_valid.npy', y_valid)
    print("done.")


if __name__ == '__main__':
    import os
    df = read_data('dataset/fer2013/fer2013.csv')
    img_array = standardize_data(df)
    img_labels = encode_features(df)
    img_features = get_features(img_array)
    split_save_data(img_features, img_labels, df)



 
