"""
CVI620 - Assignment 2 - Question 1
Cat vs Dog Classification using Multiple Methods
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
import json

# Set random seed for reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

# Paths
TRAIN_DIR = 'train'
TEST_DIR = 'test'
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

# Hyperparameters
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 50

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 20% for validation
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load data
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")
print(f"Test samples: {test_generator.samples}")
print(f"Class indices: {train_generator.class_indices}")

# Save class indices
with open(os.path.join(MODELS_DIR, 'class_indices.json'), 'w') as f:
    json.dump(train_generator.class_indices, f)

# Callbacks
def get_callbacks(model_name):
    checkpoint = ModelCheckpoint(
        os.path.join(MODELS_DIR, f'{model_name}_best.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    return [checkpoint, early_stop, reduce_lr]


# ==================== MODEL 1: Custom CNN ====================
def create_custom_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ==================== MODEL 2: VGG16 Transfer Learning ====================
def create_vgg16_model():
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(150, 150, 3)
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ==================== MODEL 3: ResNet50 Transfer Learning ====================
def create_resnet50_model():
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(150, 150, 3)
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ==================== MODEL 4: MobileNetV2 Transfer Learning ====================
def create_mobilenet_model():
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(150, 150, 3)
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# Function to train and evaluate a model
def train_and_evaluate(model, model_name):
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}\n")
    
    model.summary()
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=get_callbacks(model_name)
    )
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"\n{model_name} Test Accuracy: {test_accuracy*100:.2f}%")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, f'{model_name}_history.png'))
    plt.close()
    
    # Save final model
    model.save(os.path.join(MODELS_DIR, f'{model_name}_final.h5'))
    
    return history, test_accuracy


if __name__ == "__main__":
    results = {}
    
    # Train all models
    models_to_train = [
        (create_custom_cnn, "Custom_CNN"),
        (create_vgg16_model, "VGG16_Transfer"),
        (create_resnet50_model, "ResNet50_Transfer"),
        (create_mobilenet_model, "MobileNetV2_Transfer")
    ]
    
    for model_func, model_name in models_to_train:
        try:
            model = model_func()
            history, test_acc = train_and_evaluate(model, model_name)
            results[model_name] = {
                'test_accuracy': float(test_acc),
                'best_val_accuracy': float(max(history.history['val_accuracy']))
            }
            # Clear memory
            del model
            tf.keras.backend.clear_session()
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    # Save results
    with open(os.path.join(MODELS_DIR, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print comparison
    print(f"\n{'='*60}")
    print("RESULTS COMPARISON")
    print(f"{'='*60}\n")
    
    for model_name, metrics in results.items():
        if 'test_accuracy' in metrics:
            print(f"{model_name}:")
            print(f"  Test Accuracy: {metrics['test_accuracy']*100:.2f}%")
            print(f"  Best Val Accuracy: {metrics['best_val_accuracy']*100:.2f}%")
        else:
            print(f"{model_name}: Error - {metrics.get('error', 'Unknown')}")
        print()
    
    print("Training completed! Models saved in 'models' directory.")
