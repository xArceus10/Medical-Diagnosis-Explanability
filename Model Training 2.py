import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Define labels and image size
labels = ['PNEUMONIA', 'NORMAL']
img_size = 150

# Dataset paths (relative to your project directory)
train_path = './input/dataset/train'
test_path = './input/dataset/test'
val_path = './input/dataset/validation'

# Verify dataset paths
assert os.path.exists(train_path), f"Training path {train_path} does not exist."
assert os.path.exists(test_path), f"Test path {test_path} does not exist."
assert os.path.exists(val_path), f"Validation path {val_path} does not exist."

# Function to load and preprocess data
def get_training_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                if img_arr is not None:  # Ensure the image is successfully read
                    resized_arr = cv2.resize(img_arr, (img_size, img_size))
                    data.append([resized_arr, class_num])
            except Exception as e:
                print(f"Error processing image {img}: {e}")
    return np.array(data, dtype=object)

# Load data
train = get_training_data(train_path)
test = get_training_data(test_path)
val = get_training_data(val_path)

# Data separation
x_train, y_train = zip(*train)
x_test, y_test = zip(*test)
x_val, y_val = zip(*val)

x_train = np.array(x_train) / 255.0  # Normalize pixel values
x_test = np.array(x_test) / 255.0
x_val = np.array(x_val) / 255.0

x_train = x_train.reshape(-1, img_size, img_size, 1)
x_test = x_test.reshape(-1, img_size, img_size, 1)
x_val = x_val.reshape(-1, img_size, img_size, 1)

y_train = np.array(y_train)
y_test = np.array(y_test)
y_val = np.array(y_val)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

# Build the model
# Define the model
model = Sequential([
    Input(shape=(150, 150, 1)),  # Explicit input shape (150x150 grayscale images)
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    Dropout(0.2),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    Dropout(0.2),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification output
])

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(datagen.flow(x_train, y_train, batch_size=32),
                    validation_data=(x_val, y_val),
                    epochs=12)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy * 100:.2f}%")

# Plot training history
epochs = range(len(history.history['accuracy']))
plt.figure(figsize=(12, 6))
plt.plot(epochs, history.history['accuracy'], label='Training Accuracy')
plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Confusion matrix
predictions = (model.predict(x_test) > 0.5).astype('int32').flatten()
print(classification_report(y_test, predictions, target_names=labels))
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.show()


# Save the model
model.save('pneumonia_detection_model.h5')
print("Model saved as pneumonia_detection_model.h5")


model.summary()