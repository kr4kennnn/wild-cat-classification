import os
import cv2
import numpy as np
import pandas as pd
import os.path
import itertools
import matplotlib.pyplot as plt
import tensorflow as tf

from pathlib import Path
from tensorflow import keras
from keras import Model
from keras.optimizers import Adam
from keras.layers import Dense, Dropout
from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, confusion_matrix

#define train test and validation folders
train_folder = './dataset/train'
val_folder = './dataset/valid'
test_folder = './dataset/test'

#create lists for paths and labels
train_paths = []
train_labels = []
val_paths = []
val_labels = []
test_paths = []
test_labels = []

#load images and labels from the train folder
for species_folder in os.listdir(train_folder):
    species_folder_path = os.path.join(train_folder, species_folder)

    for image_file in os.listdir(species_folder_path):
        image_path = os.path.join(species_folder_path, image_file)

        train_paths.append(image_path)
        train_labels.append(species_folder)

#load images and labels from the validation folder
for species_folder in os.listdir(val_folder):
    species_folder_path = os.path.join(val_folder, species_folder)

    for image_file in os.listdir(species_folder_path):
        image_path = os.path.join(species_folder_path, image_file)

        val_paths.append(image_path)
        val_labels.append(species_folder)

#load images and labels from the test folder
for species_folder in os.listdir(test_folder):
    species_folder_path = os.path.join(test_folder, species_folder)

    for image_file in os.listdir(species_folder_path):
        image_path = os.path.join(species_folder_path, image_file)

        test_paths.append(image_path)
        test_labels.append(species_folder)

#verify the size of each set
print("Training set size:", len(train_paths))
print("Validation set size:", len(val_paths))
print("Testing set size:", len(test_paths))

#image size 224x224 and 3 channels(RGB)
input_shape = (224, 224, 3)
#number of classes based on folder names
num_classes = len(os.listdir(train_folder))

#define the data generators for augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

#create the train, validation, and test data generators
train_generator = train_datagen.flow_from_directory(
    train_folder,
    target_size=input_shape[:2],
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)
val_generator = val_datagen.flow_from_directory(
    val_folder,
    target_size=input_shape[:2],
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)
test_generator = test_datagen.flow_from_directory(
    test_folder,
    target_size=input_shape[:2],
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

"""
#picking a base model from keras library
base_model = tf.keras.applications.xception.Xception(
    weights='imagenet',
    include_top=False,
    input_shape=input_shape,
    pooling='max'
)

#freezing the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

#adding our own top layers for classification
model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))

#compiling the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

#training the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=val_generator,
    validation_steps=len(val_generator)
)
"""
#loading our trained model
model = tf.keras.models.load_model('trained_model.h5')
results = model.evaluate(test_generator, steps=len(test_generator))

print("    Test Loss: {:.5f}".format(results[0]))
print("Test Accuracy: {:.2f}%".format(results[1] * 100))

#running predicitons
pred = model.predict(test_generator)
pred = np.argmax(pred,axis=1)
#mapping the labels
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
pred = [labels[k] for k in pred]
print(f'Predictions: {pred}')

#function for converting filepath and labels to DF
def convert_path_to_df(dataset):
    image_dir = Path(dataset)
    #get filepaths and labels
    filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpg')) + list(image_dir.glob(r'**/*.png')) + list(image_dir.glob(r'**/*.PNG'))

    labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    #merge filepaths and labels
    image_df = pd.concat([filepaths, labels], axis=1)
    return image_df

#assigning train test and validation dataframes
train_df = convert_path_to_df(train_folder)
test_df = convert_path_to_df(test_folder)
val_df = convert_path_to_df(val_folder)

#choosing random indexes for prediction visualization
random_index = np.random.randint(0, 49, 15)
#plot
fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(30, 20),
                        subplot_kw={'xticks': [], 'yticks': []})
#iterating through random images and plotting them with the predicted value
for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(test_df.Filepath.iloc[random_index[i]]))
    #comparing the results
    if test_df.Label.iloc[random_index[i]] == pred[random_index[i]]:
        color = "green"
    else:
        color = "red"
    ax.set_title(f"Label: {test_df.Label.iloc[random_index[i]]}\nPredicted: {pred[random_index[i]]}", color=color)
plt.show()
plt.tight_layout()

#classification report
y_test = list(test_df.Label[:50])
report = classification_report(y_test, pred)
print(report)

#function for creating confusion matrix 
def create_confusion_matrix(y_true, y_pred, classes=None, figsize=(15, 7), text_size=10, norm=False, savefig=False): 
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    n_classes = cm.shape[0]

    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Greens)
    fig.colorbar(cax)

    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    #label the axes
    ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         #create axis slots for each class
         xticks=np.arange(n_classes), 
         yticks=np.arange(n_classes),
         #axes will labeled with class names or ints
         xticklabels=labels, 
         yticklabels=labels)
  
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()
    
    plt.xticks(rotation=90, fontsize=text_size)
    plt.yticks(fontsize=text_size)

    threshold = (cm.max() + cm.min()) / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                horizontalalignment="center",
                color="white" if cm[i, j] > threshold else "black",
                size=text_size)
        else:
            plt.text(j, i, f"{cm[i, j]}",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)
    savefig = True
    if savefig:
      fig.savefig("confusion_matrix.png")

create_confusion_matrix(y_test, pred, list(labels.values()))

#saving the trained model as H5 file
#model.save('trained_model.h5')