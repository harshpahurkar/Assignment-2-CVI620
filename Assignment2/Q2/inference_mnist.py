"""
CVI620 - Assignment 2 - Question 2
MNIST Inference Script - Test on individual samples
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import pickle
from tensorflow.keras.models import load_model

# Configuration
MODELS_DIR = 'models_mnist'

def load_best_model():
    """Load the best performing model"""
    try:
        with open(os.path.join(MODELS_DIR, 'results.json'), 'r') as f:
            results = json.load(f)
        
        # Find best model
        best_model_name = max(results, key=results.get)
        best_accuracy = results[best_model_name]
        
        print(f"Loading best model: {best_model_name}")
        print(f"Test Accuracy: {best_accuracy*100:.2f}%")
        
        # Load the appropriate model
        if best_model_name in ['DNN', 'CNN']:
            model_path = os.path.join(MODELS_DIR, f'{best_model_name.lower()}_best.h5')
            if not os.path.exists(model_path):
                model_path = os.path.join(MODELS_DIR, f'{best_model_name.lower()}_final.h5')
            model = load_model(model_path)
            model_type = 'deep_learning'
        else:
            # Load sklearn model
            model_file = f"{best_model_name.lower().replace('_', '')}.pkl"
            if best_model_name == 'Gradient_Boosting':
                model_file = 'gradient_boosting.pkl'
            elif best_model_name == 'Random_Forest':
                model_file = 'random_forest.pkl'
            
            with open(os.path.join(MODELS_DIR, model_file), 'rb') as f:
                model = pickle.load(f)
            model_type = 'sklearn'
        
        return model, best_model_name, model_type
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

def predict_single(model, model_name, model_type, sample_data):
    """Predict a single sample"""
    # Normalize if needed
    if sample_data.max() > 1:
        sample_data = sample_data / 255.0
    
    # Reshape for prediction
    if model_type == 'deep_learning':
        if model_name == 'CNN':
            input_data = sample_data.reshape(1, 28, 28, 1)
        else:
            input_data = sample_data.reshape(1, 784)
        prediction = model.predict(input_data, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]
    else:
        input_data = sample_data.reshape(1, 784)
        predicted_class = model.predict(input_data)[0]
        if hasattr(model, 'predict_proba'):
            confidence = model.predict_proba(input_data)[0][predicted_class]
        else:
            confidence = 1.0  # SVM doesn't provide probabilities easily
    
    return predicted_class, confidence

def visualize_prediction(sample_data, true_label, predicted_label, confidence):
    """Visualize a single prediction"""
    plt.figure(figsize=(6, 6))
    plt.imshow(sample_data.reshape(28, 28), cmap='gray')
    plt.axis('off')
    
    color = 'green' if true_label == predicted_label else 'red'
    plt.title(f'True: {true_label} | Predicted: {predicted_label}\nConfidence: {confidence*100:.1f}%',
              fontsize=14, fontweight='bold', color=color)
    plt.tight_layout()
    plt.show()

def test_random_samples(model, model_name, model_type, n_samples=10):
    """Test on random samples from test set"""
    print(f"\nTesting {n_samples} random samples from test set...")
    
    # Load test data
    test_df = pd.read_csv('mnist_test.csv')
    
    # Select random samples
    indices = np.random.choice(len(test_df), n_samples, replace=False)
    
    # Create subplot
    cols = 5
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.5*rows))
    axes = axes.flatten() if n_samples > 1 else [axes]
    
    correct = 0
    
    for i, idx in enumerate(indices):
        sample = test_df.iloc[idx]
        true_label = sample.iloc[0]
        sample_data = sample.iloc[1:].values / 255.0
        
        predicted_label, confidence = predict_single(model, model_name, model_type, sample_data)
        
        if predicted_label == true_label:
            correct += 1
        
        # Plot
        axes[i].imshow(sample_data.reshape(28, 28), cmap='gray')
        axes[i].axis('off')
        
        color = 'green' if predicted_label == true_label else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {predicted_label} ({confidence*100:.0f}%)',
                         fontsize=10, color=color, fontweight='bold')
    
    # Hide unused subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'{model_name} - Sample Predictions\nAccuracy: {correct}/{n_samples} ({correct/n_samples*100:.1f}%)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(MODELS_DIR, 'sample_predictions.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Results saved to: {output_path}")
    
    plt.show()
    
    return correct / n_samples

def find_misclassified(model, model_name, model_type, n_samples=20):
    """Find and visualize misclassified samples"""
    print(f"\nFinding misclassified samples...")
    
    # Load test data
    test_df = pd.read_csv('mnist_test.csv')
    X_test = test_df.iloc[:, 1:].values / 255.0
    y_test = test_df.iloc[:, 0].values
    
    # Get predictions for all test samples
    misclassified_indices = []
    misclassified_true = []
    misclassified_pred = []
    misclassified_conf = []
    
    print("Analyzing test set...")
    for i in range(len(X_test)):
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(X_test)} samples...")
        
        predicted_label, confidence = predict_single(model, model_name, model_type, X_test[i])
        
        if predicted_label != y_test[i]:
            misclassified_indices.append(i)
            misclassified_true.append(y_test[i])
            misclassified_pred.append(predicted_label)
            misclassified_conf.append(confidence)
        
        if len(misclassified_indices) >= n_samples:
            break
    
    print(f"\nFound {len(misclassified_indices)} misclassified samples")
    
    if len(misclassified_indices) == 0:
        print("No misclassified samples found!")
        return
    
    # Visualize
    n_show = min(n_samples, len(misclassified_indices))
    cols = 5
    rows = (n_show + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.5*rows))
    axes = axes.flatten() if n_show > 1 else [axes]
    
    for i in range(n_show):
        idx = misclassified_indices[i]
        sample_data = X_test[idx]
        
        axes[i].imshow(sample_data.reshape(28, 28), cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'True: {misclassified_true[i]}\nPred: {misclassified_pred[i]} ({misclassified_conf[i]*100:.0f}%)',
                         fontsize=10, color='red', fontweight='bold')
    
    # Hide unused subplots
    for i in range(n_show, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'{model_name} - Misclassified Samples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(MODELS_DIR, 'misclassified_samples.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Results saved to: {output_path}")
    
    plt.show()

def interactive_mode(model, model_name, model_type):
    """Interactive mode to test specific indices"""
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("Enter sample index from test set (or 'quit' to exit)")
    
    test_df = pd.read_csv('mnist_test.csv')
    print(f"Test set has {len(test_df)} samples (indices 0-{len(test_df)-1})")
    
    while True:
        try:
            user_input = input("\nSample index (or 'quit'): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            idx = int(user_input)
            
            if idx < 0 or idx >= len(test_df):
                print(f"Invalid index! Must be between 0 and {len(test_df)-1}")
                continue
            
            sample = test_df.iloc[idx]
            true_label = sample.iloc[0]
            sample_data = sample.iloc[1:].values / 255.0
            
            predicted_label, confidence = predict_single(model, model_name, model_type, sample_data)
            
            visualize_prediction(sample_data, true_label, predicted_label, confidence)
            
            print(f"\nTrue Label: {true_label}")
            print(f"Predicted Label: {predicted_label}")
            print(f"Confidence: {confidence*100:.2f}%")
            print(f"Result: {'✓ CORRECT' if predicted_label == true_label else '✗ INCORRECT'}")
        
        except ValueError:
            print("Invalid input! Please enter a number or 'quit'.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    print("="*60)
    print("MNIST CLASSIFIER - INFERENCE")
    print("="*60)
    
    # Load best model
    model, model_name, model_type = load_best_model()
    
    if model is None:
        print("\nFailed to load model. Please train models first.")
        exit(1)
    
    print("\nSelect mode:")
    print("1. Test random samples from test set")
    print("2. Find and visualize misclassified samples")
    print("3. Interactive mode (test specific indices)")
    print("4. Full test set evaluation")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        n = input("Number of samples to test (default 20): ").strip()
        n = int(n) if n else 20
        test_random_samples(model, model_name, model_type, n)
    
    elif choice == '2':
        n = input("Number of misclassified samples to show (default 20): ").strip()
        n = int(n) if n else 20
        find_misclassified(model, model_name, model_type, n)
    
    elif choice == '3':
        interactive_mode(model, model_name, model_type)
    
    elif choice == '4':
        print("\nEvaluating full test set...")
        test_df = pd.read_csv('mnist_test.csv')
        X_test = test_df.iloc[:, 1:].values / 255.0
        y_test = test_df.iloc[:, 0].values
        
        correct = 0
        total = len(X_test)
        
        for i in range(total):
            if i % 1000 == 0:
                print(f"  Processed {i}/{total} samples...")
            
            predicted_label, _ = predict_single(model, model_name, model_type, X_test[i])
            if predicted_label == y_test[i]:
                correct += 1
        
        accuracy = correct / total
        print(f"\nFull Test Set Accuracy: {accuracy*100:.2f}%")
        print(f"Correct: {correct}/{total}")
    
    else:
        print("Invalid choice!")
    
    print("\nInference completed!")
