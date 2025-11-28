"""
Fast Demo for Q1 - Trains a small model quickly to demonstrate functionality
This is NOT the full training - just a proof of concept
"""

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import json

print("="*60)
print("Q1 FAST DEMO - Cat vs Dog Classification")
print("="*60)
print("\nThis demo trains a small CNN quickly (~3-5 minutes)")
print("For full training with all 4 models, use train_catdog.py\n")

os.chdir("Q1")

# Paths
TRAIN_DIR = 'train'
TEST_DIR = 'test'
MODELS_DIR = 'models_demo'
os.makedirs(MODELS_DIR, exist_ok=True)

# Reduced parameters for faster training
IMG_SIZE = (64, 64)  # Smaller images
BATCH_SIZE = 32
EPOCHS = 10  # Fewer epochs

print(f"📂 Loading data...")
print(f"  Image size: {IMG_SIZE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")

# Minimal data augmentation for speed
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

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

print(f"\n✓ Training samples: {train_generator.samples}")
print(f"✓ Validation samples: {validation_generator.samples}")
print(f"✓ Test samples: {test_generator.samples}")
print(f"✓ Classes: {train_generator.class_indices}")

# Simple CNN model
print(f"\n🎯 Building small CNN model...")

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(f"\n📊 Model Summary:")
model.summary()

print(f"\n🚀 Training model...")
print("(This will take a few minutes...)\n")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    verbose=1
)

# Evaluate
print(f"\n📈 Evaluating on test set...")
test_loss, test_accuracy = model.evaluate(test_generator)

print(f"\n{'='*60}")
print(f"RESULTS")
print(f"{'='*60}")
print(f"\n✓ Test Accuracy: {test_accuracy*100:.2f}%")
print(f"✓ Final Training Accuracy: {history.history['accuracy'][-1]*100:.2f}%")
print(f"✓ Final Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")

# Save
model.save(os.path.join(MODELS_DIR, 'demo_cnn.h5'))

with open(os.path.join(MODELS_DIR, 'class_indices.json'), 'w') as f:
    json.dump(train_generator.class_indices, f)

results = {
    'Demo_CNN': {
        'test_accuracy': float(test_accuracy),
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1])
    }
}

with open(os.path.join(MODELS_DIR, 'demo_results.json'), 'w') as f:
    json.dump(results, f, indent=4)

print(f"\n💾 Model saved to: {MODELS_DIR}/demo_cnn.h5")
print(f"💾 Results saved to: {MODELS_DIR}/demo_results.json")

print(f"\n{'='*60}")
print(f"DEMO COMPLETE!")
print(f"{'='*60}")
print(f"\n📝 Note: This was a quick demo with:")
print(f"  - Smaller images (64x64 vs 150x150)")
print(f"  - Fewer epochs (10 vs 50)")
print(f"  - Simple model (not transfer learning)")
print(f"\nFor full training with all 4 models:")
print(f"  python train_catdog.py")
print(f"\nExpected full training results:")
print(f"  - ResNet50: 96-99% accuracy (best)")
print(f"  - VGG16: 94-98% accuracy")
print(f"  - MobileNetV2: 93-97% accuracy")
print(f"  - Custom CNN: 90-95% accuracy")
