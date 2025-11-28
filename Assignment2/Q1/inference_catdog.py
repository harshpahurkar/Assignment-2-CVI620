"""
CVI620 - Assignment 2 - Question 1
Inference Script for Cat vs Dog Classification
Test on new images from the internet
"""

import os
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image

# Configuration
MODELS_DIR = 'models'
IMG_SIZE = (150, 150)

# Load class indices
with open(os.path.join(MODELS_DIR, 'class_indices.json'), 'r') as f:
    class_indices = json.load(f)

# Reverse the dictionary to get index -> class name
index_to_class = {v: k for k, v in class_indices.items()}

def load_best_model():
    """Load the best performing model"""
    # Try to load results to find best model
    try:
        with open(os.path.join(MODELS_DIR, 'training_results.json'), 'r') as f:
            results = json.load(f)
        
        # Find best model by test accuracy
        best_model_name = max(results, key=lambda x: results[x].get('test_accuracy', 0))
        print(f"Loading best model: {best_model_name}")
        print(f"Test Accuracy: {results[best_model_name]['test_accuracy']*100:.2f}%")
        
        model_path = os.path.join(MODELS_DIR, f'{best_model_name}_best.h5')
        if not os.path.exists(model_path):
            model_path = os.path.join(MODELS_DIR, f'{best_model_name}_final.h5')
        
        return load_model(model_path), best_model_name
    except Exception as e:
        print(f"Error loading best model: {e}")
        print("Please specify model manually.")
        return None, None

def preprocess_image(img_path):
    """Load and preprocess image for prediction"""
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array

def predict_image(model, img_path, show_image=True):
    """Predict a single image"""
    img, img_array = preprocess_image(img_path)
    
    # Make prediction
    prediction = model.predict(img_array, verbose=0)[0][0]
    
    # Get class name
    predicted_class_idx = int(prediction > 0.5)
    predicted_class = index_to_class[predicted_class_idx]
    confidence = prediction if predicted_class_idx == 1 else 1 - prediction
    
    if show_image:
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Prediction: {predicted_class.upper()}\nConfidence: {confidence*100:.2f}%', 
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    return predicted_class, confidence

def test_multiple_images(model, image_folder, save_results=True):
    """Test model on multiple images in a folder"""
    if not os.path.exists(image_folder):
        print(f"Folder {image_folder} not found!")
        return
    
    # Get all image files
    image_files = [f for f in os.listdir(image_folder) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    if not image_files:
        print(f"No images found in {image_folder}")
        return
    
    results = []
    n_images = len(image_files)
    
    # Create subplot grid
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if n_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_images > 1 else [axes]
    
    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(image_folder, img_file)
        print(f"\nProcessing: {img_file}")
        
        try:
            img, img_array = preprocess_image(img_path)
            prediction = model.predict(img_array, verbose=0)[0][0]
            
            predicted_class_idx = int(prediction > 0.5)
            predicted_class = index_to_class[predicted_class_idx]
            confidence = prediction if predicted_class_idx == 1 else 1 - prediction
            
            print(f"  Prediction: {predicted_class.upper()}")
            print(f"  Confidence: {confidence*100:.2f}%")
            
            results.append({
                'filename': img_file,
                'prediction': predicted_class,
                'confidence': float(confidence)
            })
            
            # Plot
            axes[idx].imshow(img)
            axes[idx].axis('off')
            axes[idx].set_title(f'{img_file}\n{predicted_class.upper()}\n{confidence*100:.1f}%',
                               fontsize=10, fontweight='bold')
        
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'filename': img_file,
                'error': str(e)
            })
    
    # Hide unused subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_results:
        output_path = os.path.join(image_folder, 'predictions.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nResults saved to: {output_path}")
        
        # Save JSON results
        json_path = os.path.join(image_folder, 'predictions.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"JSON results saved to: {json_path}")
    
    plt.show()
    
    return results

def interactive_mode(model):
    """Interactive mode to test individual images"""
    print("\n" + "="*60)
    print("INTERACTIVE PREDICTION MODE")
    print("="*60)
    print("Enter image path to predict (or 'quit' to exit)")
    
    while True:
        img_path = input("\nImage path: ").strip().strip('"').strip("'")
        
        if img_path.lower() in ['quit', 'exit', 'q']:
            break
        
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            continue
        
        try:
            predicted_class, confidence = predict_image(model, img_path, show_image=True)
            print(f"\nPrediction: {predicted_class.upper()}")
            print(f"Confidence: {confidence*100:.2f}%")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    print("="*60)
    print("CAT vs DOG CLASSIFIER - INFERENCE")
    print("="*60)
    
    # Load the best model
    model, model_name = load_best_model()
    
    if model is None:
        print("\nFailed to load model. Please check that models exist in 'models' directory.")
        exit(1)
    
    print("\nSelect mode:")
    print("1. Test on folder of images")
    print("2. Test on test dataset")
    print("3. Interactive mode (single image)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        folder = input("Enter folder path: ").strip().strip('"').strip("'")
        test_multiple_images(model, folder)
    
    elif choice == '2':
        print("\nTesting on test dataset...")
        test_multiple_images(model, 'test_images_internet', save_results=True)
    
    elif choice == '3':
        interactive_mode(model)
    
    else:
        print("Invalid choice!")
    
    print("\nInference completed!")
