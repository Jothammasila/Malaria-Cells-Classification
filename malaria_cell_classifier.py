"""
AUTHOR: Jotham Masila
TITLE: Malaria-Cell-Classifier
DESCRIPTION: This file contains code for training a convolutional neural network to classify images of malaria-infected and uninfected cells.
CREATION DATE: 2024-05-12
LAST MODIFIED: 2024-05-13
VERSION: 1.0
LANGUAGE: Python
FRAMEWORK: TensorFlow
"""

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

"""# **Data Loading and Exploration**

**Data Loading**

 The data is loaded as **train_ds** *(for training set)*, **val_ds** *(for validation set)* and **test_ds** *(for test set)*
"""

train_ds, val_ds, test_ds = tfds.load('malaria',
                                      split=['train[:70%]', 'train[70%:85%]', 'train[85%:]'],
                                      shuffle_files=True, as_supervised=True)

"""Examining the size of each set"""

train_size = tf.data.experimental.cardinality(train_ds).numpy()
val_size = tf.data.experimental.cardinality(val_ds).numpy()
test_size = tf.data.experimental.cardinality(test_ds).numpy()

print("Training set size:", train_size)
print("Validation set size:", val_size)
print("Test set size:", test_size)

"""Visualize the training set."""

num_examples = 10
fig, axes = plt.subplots(2, 5, figsize=(15, 7))

# Iterate over the dataset and plot each example
for i, (image, label) in enumerate(train_ds.take(num_examples)):
    row = i // 5
    col = i % 5

    # Plot the image
    axes[row, col].imshow(image.numpy())
    axes[row, col].set_title(f"Label: {label.numpy()}")
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()

print('# Examine the shape')
print('  ==================')

for image, label in train_ds.take(5):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())

"""> The images have different shapes as seen above.
> They need to be preprocessed before model training.

# **Preprocessing Pipeline**
"""

# This is the target image size
image_size = [128, 128]

# Convert the image data type
def convert_image(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label

# Pad the images
def padding(image,label):
    image,label = convert_image(image, label)
    image = tf.image.resize_with_crop_or_pad(image,image_size[0] ,image_size[1])
    return image,label

# The final preprocessing pipeline
def pipeline(image, label):
    image, label = padding(image, label)
    return image, label

BATCH_SIZE = 32
preprocessed_train_ds  = (
    train_ds
    .cache()
    .map(pipeline)
    .batch(BATCH_SIZE)
)

preprocessed_val_ds = (
    val_ds
    .cache()
    .map(pipeline)
    .batch(BATCH_SIZE)
)

preprocessed_test_ds = (
    val_ds
    .cache()
    .map(pipeline)
    .batch(BATCH_SIZE)
)

image_batch, label_batch = next(iter(preprocessed_train_ds))

def show_preprocessed_data(image_batch, label_batch):
    plt.figure(figsize=(10,10))
    for n in range(5):
        ax = plt.subplot(1,5,n+1)
        plt.imshow(image_batch[n])
        if label_batch[n]:
            plt.title("uninfected")
        else:
            plt.title("parasitized")
        plt.axis("off")

show_preprocessed_data(image_batch.numpy(), label_batch.numpy())

# The output images are of the same sizes, i.e (128, 128, 3)

"""# **Model Architecture**"""

# Define the CNN model
def malaria_cnn(input_shape, dropout_rate=0.2):
    model = Sequential([
        # Convolutional layers
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        # Flatten layer to feed into dense layers
        Flatten(),

        # Dense and Dropout layers
        Dense(512, activation='relu'),
        Dropout(dropout_rate),
        Dense(128, activation='relu'),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
    ])
    return model

input_shape = (128, 128, 3)  # Input shape of the images after preprocessing
learning_rate=0.001
batch_size = 64

# CNN model
model = malaria_cnn(input_shape)

# Compile the model
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='binary_crossentropy',  # Binary cross-entropy loss for binary classification
              metrics=['accuracy', 'auc'])

# Display the model summary
model.summary()

"""# **Model Training**"""

# Train the model
history = model.fit(preprocessed_train_ds, epochs=8, batch_size=batch_size, validation_data=preprocessed_val_ds)

"""# **Model Evaluation**"""

# Evaluate the model on the test set
# test_loss, test_accuracy =
eval_ = model.evaluate(preprocessed_test_ds)
print()
print("Model Evaluation")
print("================")
print(f"Model Accuracy: {eval_[1]}")
print(f"Area Under Curve (AUC):  {eval_[2]}")
print(f"Model Loss (Binary CrossEntropy Loss): {eval_[0]}")

"""**Training Curves**"""

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

"""**ROC Curve**"""

# Concatenate the labels from preprocessed_test_ds into a single array
y_true = np.concatenate([y for x, y in preprocessed_test_ds], axis=0)

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_true, predictions)
auc = roc_auc_score(y_true, predictions)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

"""**Confusion Matrix**"""

# Concatenate the labels from preprocessed_test_ds into a single array
y_true = np.concatenate([y for x, y in preprocessed_test_ds], axis=0)

# Convert predictions to binary values (0 or 1) based on a threshold (e.g., 0.5)
binary_predictions = np.where(predictions > 0.5, 1, 0)

# Compute confusion matrix
cm = confusion_matrix(y_true, binary_predictions)

# Define class labels
class_labels = ['Negative', 'Positive']

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

"""# **Model Testing**"""

# Make predictions on the test set
predictions = model.predict(preprocessed_test_ds)

# Convert predictions to binary values (0 or 1) based on the threshold (0.5)
binary_predictions = np.where(predictions > 0.5, 1, 0)

#   1. Convert predictions to binary values (0 or 1) based on a threshold (0.5)
binary_predictions = np.where(predictions > 0.5, 1, 0)

#   2. Display the predictions of the first 10 images of the test data
print(binary_predictions[0:10])

"""# **Model Saving**"""

# # Specify the file path where you want to save the model
# model_path = '/kaggle/working/malaria_cell_model.h5'


# # Save the model
# model.save(model_path)

# print("Model saved successfully at:", model_path)

"""# **Loading Saved Model**"""

# Specify the file path where the model is saved
model_path = '/kaggle/working/malaria_cell_model.h5' # Adjust the file path if needed

# Load the saved model
malaria_model = load_model(model_path)

# Now you can use the loaded model for predictions or further training

# Example: Make predictions using the loaded model
prediction = malaria_model.predict(preprocessed_test_ds)


#   1. Convert predictions to binary values (0 or 1) based on a threshold (0.5)
binary_prediction = np.where(prediction > 0.5, 1, 0)

#   2. Display the predictions of the first 10 images of the test data
print(binary_prediction[0:10])
