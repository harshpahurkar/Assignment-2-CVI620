"""
CVI620 - Assignment 2 - Question 2
MNIST Handwritten Digit Classification using Multiple Methods
Target: Achieve at least 90% accuracy
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import json
import pickle

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Create directories
MODELS_DIR = 'models_mnist'
os.makedirs(MODELS_DIR, exist_ok=True)

print("Loading MNIST data...")
# Load data
train_df = pd.read_csv('mnist_train.csv')
test_df = pd.read_csv('mnist_test.csv')

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# Separate features and labels
X_train_full = train_df.iloc[:, 1:].values  # All columns except first (label)
y_train_full = train_df.iloc[:, 0].values   # First column (label)

X_test = test_df.iloc[:, 1:].values
y_test = test_df.iloc[:, 0].values

# Normalize pixel values to [0, 1]
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0

# Split training data into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

print(f"\nData split:")
print(f"  Training samples: {X_train.shape[0]}")
print(f"  Validation samples: {X_val.shape[0]}")
print(f"  Test samples: {X_test.shape[0]}")
print(f"  Features per sample: {X_train.shape[1]}")
print(f"  Classes: {len(np.unique(y_train))}")

# Visualize some samples
def plot_samples(X, y, n_samples=10):
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    
    indices = np.random.choice(len(X), n_samples, replace=False)
    
    for i, idx in enumerate(indices):
        img = X[idx].reshape(28, 28)
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Label: {y[idx]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, 'sample_images.png'))
    plt.close()
    print(f"Sample images saved to {MODELS_DIR}/sample_images.png")

plot_samples(X_train, y_train)

# Results storage
results = {}

# ==================== METHOD 1: Deep Neural Network (DNN) ====================
print("\n" + "="*60)
print("METHOD 1: Deep Neural Network (DNN)")
print("="*60)

model_dnn = Sequential([
    Dense(512, activation='relu', input_shape=(784,)),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(10, activation='softmax')
])

model_dnn.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_dnn = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
    ModelCheckpoint(os.path.join(MODELS_DIR, 'dnn_best.h5'), monitor='val_accuracy', save_best_only=True)
]

history_dnn = model_dnn.fit(
    X_train, y_train,
    epochs=50,
    batch_size=128,
    validation_data=(X_val, y_val),
    callbacks=callbacks_dnn,
    verbose=1
)

# Evaluate
y_pred_dnn = np.argmax(model_dnn.predict(X_test), axis=1)
acc_dnn = accuracy_score(y_test, y_pred_dnn)
print(f"DNN Test Accuracy: {acc_dnn*100:.2f}%")
results['DNN'] = acc_dnn

# Save model
model_dnn.save(os.path.join(MODELS_DIR, 'dnn_final.h5'))


# ==================== METHOD 2: Convolutional Neural Network (CNN) ====================
print("\n" + "="*60)
print("METHOD 2: Convolutional Neural Network (CNN)")
print("="*60)

# Reshape data for CNN (add channel dimension)
X_train_cnn = X_train.reshape(-1, 28, 28, 1)
X_val_cnn = X_val.reshape(-1, 28, 28, 1)
X_test_cnn = X_test.reshape(-1, 28, 28, 1)

model_cnn = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model_cnn.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_cnn = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
    ModelCheckpoint(os.path.join(MODELS_DIR, 'cnn_best.h5'), monitor='val_accuracy', save_best_only=True)
]

history_cnn = model_cnn.fit(
    X_train_cnn, y_train,
    epochs=50,
    batch_size=128,
    validation_data=(X_val_cnn, y_val),
    callbacks=callbacks_cnn,
    verbose=1
)

# Evaluate
y_pred_cnn = np.argmax(model_cnn.predict(X_test_cnn), axis=1)
acc_cnn = accuracy_score(y_test, y_pred_cnn)
print(f"CNN Test Accuracy: {acc_cnn*100:.2f}%")
results['CNN'] = acc_cnn

# Save model
model_cnn.save(os.path.join(MODELS_DIR, 'cnn_final.h5'))


# ==================== METHOD 3: Random Forest ====================
print("\n" + "="*60)
print("METHOD 3: Random Forest Classifier")
print("="*60)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("Training Random Forest...")
rf_model.fit(X_train, y_train)

# Evaluate
y_pred_rf = rf_model.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Test Accuracy: {acc_rf*100:.2f}%")
results['Random_Forest'] = acc_rf

# Save model
with open(os.path.join(MODELS_DIR, 'random_forest.pkl'), 'wb') as f:
    pickle.dump(rf_model, f)


# ==================== METHOD 4: Support Vector Machine (SVM) ====================
print("\n" + "="*60)
print("METHOD 4: Support Vector Machine (SVM)")
print("="*60)

# Use a subset for faster training
subset_size = 10000
X_train_svm = X_train[:subset_size]
y_train_svm = y_train[:subset_size]

svm_model = SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    random_state=42,
    verbose=True
)

print(f"Training SVM on {subset_size} samples...")
svm_model.fit(X_train_svm, y_train_svm)

# Evaluate
y_pred_svm = svm_model.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Test Accuracy: {acc_svm*100:.2f}%")
results['SVM'] = acc_svm

# Save model
with open(os.path.join(MODELS_DIR, 'svm.pkl'), 'wb') as f:
    pickle.dump(svm_model, f)


# ==================== METHOD 5: K-Nearest Neighbors ====================
print("\n" + "="*60)
print("METHOD 5: K-Nearest Neighbors (KNN)")
print("="*60)

knn_model = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',
    n_jobs=-1
)

print("Training KNN...")
knn_model.fit(X_train, y_train)

# Evaluate
y_pred_knn = knn_model.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
print(f"KNN Test Accuracy: {acc_knn*100:.2f}%")
results['KNN'] = acc_knn

# Save model
with open(os.path.join(MODELS_DIR, 'knn.pkl'), 'wb') as f:
    pickle.dump(knn_model, f)


# ==================== METHOD 6: Gradient Boosting ====================
print("\n" + "="*60)
print("METHOD 6: Gradient Boosting Classifier")
print("="*60)

gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    verbose=1
)

print("Training Gradient Boosting...")
gb_model.fit(X_train, y_train)

# Evaluate
y_pred_gb = gb_model.predict(X_test)
acc_gb = accuracy_score(y_test, y_pred_gb)
print(f"Gradient Boosting Test Accuracy: {acc_gb*100:.2f}%")
results['Gradient_Boosting'] = acc_gb

# Save model
with open(os.path.join(MODELS_DIR, 'gradient_boosting.pkl'), 'wb') as f:
    pickle.dump(gb_model, f)


# ==================== RESULTS COMPARISON ====================
print("\n" + "="*60)
print("RESULTS COMPARISON")
print("="*60)

# Sort results by accuracy
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

for i, (model_name, accuracy) in enumerate(sorted_results, 1):
    status = "✓ PASS" if accuracy >= 0.90 else "✗ FAIL"
    print(f"{i}. {model_name:25s}: {accuracy*100:6.2f}% {status}")

# Save results
with open(os.path.join(MODELS_DIR, 'results.json'), 'w') as f:
    json.dump(results, f, indent=4)

# Plot comparison
plt.figure(figsize=(12, 6))
models = list(results.keys())
accuracies = [results[m]*100 for m in models]

bars = plt.bar(models, accuracies, color=['green' if acc >= 90 else 'orange' for acc in accuracies])
plt.axhline(y=90, color='r', linestyle='--', label='90% Target')
plt.xlabel('Model', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('MNIST Classification - Model Comparison', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.ylim([80, 100])
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, 'model_comparison.png'), dpi=150)
plt.close()

# Detailed report for best model
best_model_name = sorted_results[0][0]
print(f"\n{'='*60}")
print(f"BEST MODEL: {best_model_name} ({sorted_results[0][1]*100:.2f}%)")
print(f"{'='*60}\n")

# Get predictions from best model
if best_model_name == 'DNN':
    y_pred_best = y_pred_dnn
elif best_model_name == 'CNN':
    y_pred_best = y_pred_cnn
elif best_model_name == 'Random_Forest':
    y_pred_best = y_pred_rf
elif best_model_name == 'SVM':
    y_pred_best = y_pred_svm
elif best_model_name == 'KNN':
    y_pred_best = y_pred_knn
else:
    y_pred_best = y_pred_gb

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_best))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, 'confusion_matrix_best.png'), dpi=150)
plt.close()

print(f"\nAll models and results saved in '{MODELS_DIR}' directory.")
print("\nTraining completed!")
