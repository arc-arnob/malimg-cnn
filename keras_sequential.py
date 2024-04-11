#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 06:54:57 2024

@author: arnobchowdhury
"""
import tensorflow as tf
import cv2
import os
import numpy as np


from keras.layers import Conv2D, Activation, MaxPooling2D, BatchNormalization
from keras.initializers import RandomNormal
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from sklearn.preprocessing import LabelEncoder

# Define the input shape of your images
input_shape = (128, 128, 1)  # Assuming 1 channel images of size 128x128

# Initialize the model
model = Sequential()

# Layer 1
model.add(Conv2D(50, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

# Layer 2
model.add(Conv2D(70, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

# Layer 3
model.add(Conv2D(70, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

# Layer 4
# Add fully connected layers
model.add(Flatten())
model.add(Dropout(rate=0.5))
model.add(Dense(256, activation='relu')) # hidden_neurons


# O/p Layer
# Dropout layer
model.add(Dropout(rate=0.5))

model.add(Dense(25, activation='softmax')) # parameters["num_classes"]


# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train TEST VALIDATION DATA

train_dir = "malimg_dataset/train"
valid_dir = 'malimg_dataset/val'
test_dir = 'malimg_dataset/test'
folders = os.listdir(train_dir)
print(folders)

train_data, train_labels = [], []
valid_data, valid_labels = [], []
test_data, test_labels = [], []

def read_data(folder):
    data, labels = [], []
    for label in folders:
        path = f"{folder}/{label}/"
        folder_data = os.listdir(path)
        for image_path in folder_data:
            img = cv2.imread(path + image_path)
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(gray_image, (128, 128))
            data.append(np.array(img))
            labels.append(label)

    return data, labels

label_encoder = LabelEncoder()
print("Processing Images....")
train_data, train_labels = read_data(train_dir)
valid_data, valid_labels = read_data(valid_dir)
test_data, test_labels = read_data(test_dir)
print("Done Processing Images....")

train_data = np.array(train_data)
#train_data = np.expand_dims(train_data, axis=-1)
train_labels = np.array(train_labels)
train_labels_encoded = label_encoder.fit_transform(train_labels)



valid_data = np.array(valid_data)
#valid_data = np.expand_dims(valid_data, axis=-1)
valid_labels = np.array(valid_labels)
valid_labels_encoded = label_encoder.transform(valid_labels)



test_data = np.array(test_data)
#test_data = np.expand_dims(test_data, axis=-1)
test_labels = np.array(test_labels)
test_labels_encoded = label_encoder.transform(test_labels)

X_train_cv = np.concatenate((train_data, valid_data), axis=0)
y_train_encoded_cv = np.concatenate((train_labels_encoded, valid_labels_encoded), axis=0)


# %%
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, mean_absolute_error

kf = KFold(n_splits=10)
num_classes = 25  
conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int)
mae = []
per_fold_accs = []

print("Training....")
# Perform 10-fold cross-validation
for train_index, val_index in kf.split(X_train_cv):
    X_train_fold, X_val_fold = X_train_cv[train_index], X_train_cv[val_index]
    y_train_fold, y_val_fold = y_train_encoded_cv[train_index], y_train_encoded_cv[val_index]

    # Train the model
    history = model.fit(X_train_fold, y_train_fold, batch_size=1, epochs=10, validation_data=(X_val_fold, y_val_fold))
    print("Training done with a CV")

    # Evaluate the model on test data
    test_loss, test_accuracy = model.evaluate(test_data, test_labels_encoded)
    # Predict labels for the validation data
    y_pred = model.predict(X_val_fold)
    y_pred_labels = np.argmax(y_pred, axis=1)
    
    # Compute MAE for the fold
    mae_fold = mean_absolute_error(y_val_fold, y_pred_labels)
    mae.append(mae_fold)
    
    conf_matrix += confusion_matrix(y_val_fold, y_pred_labels, labels=np.arange(num_classes))
    print("Confusion Matrix {} is {}", train_index, conf_matrix)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)
    per_fold_accs.append(test_accuracy)
print("Training ENDED")
print(conf_matrix)
print(mae)
print(per_fold_accs)


#%% 1 fold, train valid Confusion matrix

model.fit(train_data, train_labels_encoded, epochs=10, batch_size=1, validation_data=(valid_data, valid_labels_encoded))

predictions_test = model.predict(test_data)

# Convert predictions from one-hot encoded format to class labels
predicted_labels_test = np.argmax(predictions_test, axis=1)

# Compute the confusion matrix
conf_matrix_one_fold = confusion_matrix(test_labels_encoded, predicted_labels_test)

# Print the confusion matrix
print("Confusion Matrix (Test Data):")
print(conf_matrix_one_fold)


#%% EXPORTING
import pandas as pd

conf_matrix_df = pd.DataFrame(conf_matrix)
conf_matrix_df.to_csv('confusion_matrix_10_fold.csv', index=False)

#%%
conf_matrix_df = pd.DataFrame(conf_matrix)
conf_matrix_df.to_excel('confusion_matrix.xlsx', index=False)

#%%
conf_matrix_df_2 = pd.DataFrame(conf_matrix_one_fold)
conf_matrix_df_2.to_csv('confusion_matrix_1_fold.csv', index=False)

#%%
conf_matrix_df_2 = pd.DataFrame(conf_matrix_one_fold)
conf_matrix_df_2.to_excel('confusion_matrix_1_fold.xlsx', index=False)

#%%
a = np.asarray(per_fold_accs)
np.savetxt("acc_per_fold.csv", a, delimiter=",")
#%%
'''
# Train the model
history = model.fit(train_data, train_labels_encoded, batch_size=1, epochs=10, validation_data=(valid_data, valid_labels_encoded))
print("Training End.")

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_data, test_labels_encoded)
print("Test Accuracy:", test_accuracy)
'''
#%%

'''predictions_test = model.predict(valid_data)
# Convert predictions from one-hot encoded format to class labels
predicted_labels_valid = np.argmax(predictions_test, axis=1)

# Compute the confusion matrix
conf_matrix_test = confusion_matrix(valid_labels_encoded, predicted_labels_valid)

# Print the confusion matrix
print("Confusion Matrix (Test Data):")
print(conf_matrix_test)'''

#%%
def calculate_metrics(conf_matrix):
    # Calculate precision, recall, and F1 score for each class
    precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
    recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Calculate macro-averaged precision, recall, and F1 score
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)

    return macro_f1

print(calculate_metrics(conf_matrix))


