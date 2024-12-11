import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
import ImageDataGenerator
import matplotlib.pyplot as plt

# Paths to the dataset directories
train_dir = "dataset/train/"
val_dir = "dataset/validation/"
test_dir = "dataset/test/"

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Normalize pixel values
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Generators for training, validation, and testing
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(224, 224), batch_size=32, class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(224, 224), batch_size=32, class_mode='binary'
)

# Define model with explicit layer names
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3), name="conv2d_1"),
    MaxPooling2D(pool_size=(2, 2), name="maxpool_1"),
    Conv2D(64, (3, 3), activation='relu', name="conv2d_2"),
    MaxPooling2D(pool_size=(2, 2), name="maxpool_2"),
    Conv2D(128, (3, 3), activation='relu', name="conv2d_3"),
    MaxPooling2D(pool_size=(2, 2), name="maxpool_3"),
    Flatten(name="flatten"),
    Dense(128, activation='relu', name="dense_1"),
    Dropout(0.5, name="dropout_1"),
    Dense(1, activation='sigmoid', name="output")  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=5  # Adjust epochs as needed
)

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the model
model.save("cnn_model2.h5")
print("Model saved as cnn_model2.h5")

# Plot training and validation accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()




