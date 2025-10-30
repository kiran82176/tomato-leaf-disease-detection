"""
Modern CNN Training Script for Tomato Leaf Disease Detection
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import json
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Build CNN Model
def create_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Data preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
print("Loading training data...")
training_set = train_datagen.flow_from_directory(
    'train',
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'
)

# Load validation data
print("Loading validation data...")
validation_set = val_datagen.flow_from_directory(
    'val',
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'
)

# Get class names
class_names = list(training_set.class_indices.keys())
num_classes = len(class_names)

print(f"\nFound {num_classes} classes:")
for i, class_name in enumerate(class_names):
    print(f"  {i}: {class_name}")

# Save class names to a JSON file
with open('class_names.json', 'w') as f:
    json.dump(class_names, f)
print("\nClass names saved to class_names.json")

# Create model
print("\nBuilding CNN model...")
model = create_model(num_classes)
model.summary()

# Callbacks
checkpoint = ModelCheckpoint(
    'CNN.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

# Train the model
print("\nStarting training...")
history = model.fit(
    training_set,
    epochs=50,
    validation_data=validation_set,
    callbacks=[checkpoint, early_stopping],
    verbose=1
)

# Save final model
model.save('CNN.h5')
print("\nModel training completed and saved as CNN.h5")

# Print final metrics
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
print(f"\nFinal Training Accuracy: {final_train_acc:.4f}")
print(f"Final Validation Accuracy: {final_val_acc:.4f}")
