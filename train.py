import tensorflow as tf
import cv2
import os
import numpy as np
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, BatchNormalization
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.metrics import confusion_matrix
from collections import Counter
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle
import csv
import sys


def main():
    devices = tf.config.list_physical_devices()
    print("Available physical devices:")
    for device in devices:
        print(device)
    '''
    Dataset order: 
        GRAY128 (Change the train_dir variable to the name of root directory)
            -> Names of the malwares
                -> Sample images of the malwares
    '''

    train_dir = "GRAY128"
    NUM_SAMPLES_THRESHOLD = 2
    class_data = {}
    for i in os.listdir(train_dir):
        class_dir = train_dir + '/'+ str(i) + '/'
        num_samples = len(os.listdir(class_dir))
        if num_samples > NUM_SAMPLES_THRESHOLD:
            class_data[i] = num_samples

    # Train TEST VALIDATION DATA

    folders = list(class_data.keys()) # only read those classes whose num_samples are greater than 10
    train_data, train_labels = [], []

    def read_data(folder):
        data, labels = [], []
        for label in folders:
            path = f"{folder}/{label}/"
            folder_data = os.listdir(path)
            for image_path in folder_data:
                img = cv2.imread(path + image_path)
                if img is None:
                    print("Error: Unable to load image for sample:", label, image_path)
                else:
                    # Convert RGB image to grayscale
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    img = cv2.resize(img, (128,128))
                    data.append(np.array(img))
                    labels.append(label)

        return data, labels

    label_encoder = LabelEncoder()
    print("Processing Images....")
    train_data, train_labels = read_data(train_dir)
    print("Done Processing Images....")

    train_data = np.stack(train_data, axis=0)
    train_labels = np.array(train_labels)
    y_train_encoded_cv = label_encoder.fit_transform(train_labels)

    # Reshape to create the training data
    n,h,w = train_data.shape
    X_train_cv = train_data.reshape(n, h, w, 1)

    NUM_CLASSES = len(folders)
    BATCH_SIZE = 16
    NUM_SPLITS = 10
    NUM_EPOCHS = 20

    # Comment this if you are running it without loading the cross validation indices
    file_path = "cross_validation_indices.pkl"
    with open(file_path, 'rb') as file:
        split_indices = pickle.load(file)

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

    model.add(Dense(NUM_CLASSES, activation='softmax')) # parameters["num_classes"]

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Change the value of k whichever necessary
    k = int(sys.argv[1])
    print(k)

    checkpoint_filepath = 'weights/model_checkpoint'+str(k)+'.h5'
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath,
                                        save_weights_only=True,
                                        monitor='val_accuracy',
                                        mode='max',
                                        save_best_only=True)



    num_classes = NUM_CLASSES
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int8)

    #List to store the performance of each model on validation set
    print("Training....")
    # Perform 10-fold cross-validation one at a time
    train_index, val_index = split_indices[k]
    X_train_fold, X_val_fold = X_train_cv[train_index], X_train_cv[val_index]
    y_train_fold, y_val_fold = y_train_encoded_cv[train_index], y_train_encoded_cv[val_index]

    # Train the model
    history = model.fit(X_train_fold, y_train_fold, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(X_val_fold, y_val_fold), callbacks=[checkpoint_callback])
    print("Training done with a CV")

    y_pred = model.predict(X_val_fold)
    y_pred_labels = np.argmax(y_pred, axis=1)

    y_train_pred = model.predict(X_train_fold)
    y_train_pred_labels = np.argmax(y_train_pred, axis=1)


    #Compute the metrics of validation set
    conf_matrix = confusion_matrix(y_val_fold, y_pred_labels, labels=np.arange(NUM_CLASSES))
    mae = mean_absolute_error(y_val_fold, y_pred_labels)
    accuracy = accuracy_score(y_val_fold, y_pred_labels)

    #Compute the metrics of training set
    conf_matrix_train = confusion_matrix(y_train_fold, y_train_pred_labels, labels=np.arange(NUM_CLASSES))
    mae_train = mean_absolute_error(y_train_fold, y_train_pred_labels)
    accuracy_train = accuracy_score(y_train_fold, y_train_pred_labels)

    # Define file paths
    metrics_file = 'metrics' +str(k)+ '.csv'
    confusion_matrix_file = 'confusion_matrix'+ str(k)+'.txt'

    # Metrics for validation set
    validation_metrics = {
        "mean_absolute_error": mae,
        "accuracy": accuracy
    }

    # Metrics for training set
    training_metrics = {
        "mean_absolute_error": mae_train,
        "accuracy": accuracy_train
    }

    # Write mean absolute error and accuracy to a CSV file
    with open(metrics_file, 'w', newline='') as csvfile:
        fieldnames = ['set', 'mean_absolute_error', 'accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerow({'set': 'Validation', 
                        'mean_absolute_error': validation_metrics['mean_absolute_error'], 
                        'accuracy': validation_metrics['accuracy']})
        
        writer.writerow({'set': 'Training', 
                        'mean_absolute_error': training_metrics['mean_absolute_error'], 
                        'accuracy': training_metrics['accuracy']})

    # Write confusion matrix to a text file
    with open(confusion_matrix_file, 'w') as file:
        file.write("Confusion Matrix (Validation):\n")
        np.savetxt(file, conf_matrix, fmt='%d')
        file.write("\n\n")
        file.write("Confusion Matrix (Training):\n")
        np.savetxt(file, conf_matrix_train, fmt='%d')

if __name__ == "__main__":
    main()