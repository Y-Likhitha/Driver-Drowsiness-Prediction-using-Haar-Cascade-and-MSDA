import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants
IMG_SIZE = 48  # Resize images to 48x48
BATCH_SIZE = 32
EPOCHS = 10
DATASET_PATH = "dataset"  # Path to the dataset folder

# Data Preprocessing
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2  # 80% training, 20% validation
)

train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

# Vision Transformer Model
def create_vit_model(img_size, num_classes=1):
    vit_model = keras.applications.ViT(
        image_size=img_size,
        patch_size=16,
        num_layers=8,
        num_heads=8,
        hidden_dim=64,
        mlp_dim=128,
        dropout=0.1,
        representation_size=64,
        classifier="token",
        include_top=False
    )

    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = vit_model(inputs, training=True)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    return model

# Compile the model
model = create_vit_model(IMG_SIZE)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# Save the model
model.save("models/vit_eye_model.h5")
print("✅ Model training complete. Saved as 'models/vit_eye_model.h5'")
