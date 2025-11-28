"""
Test internet images using the demo model
"""

import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
from PIL import Image

print("="*60)
print("Testing Internet Images - Cat vs Dog")
print("="*60)

# Load the demo model
model_path = 'models_demo/demo_cnn.h5'
images_folder = 'test_images_internet'

if not os.path.exists(model_path):
    print(f"\n❌ Model not found at: {model_path}")
    print("Please run: python quick_demo_q1.py first")
    exit(1)

print(f"\n📂 Loading model from: {model_path}")
model = load_model(model_path)

# Load class indices
with open('models_demo/class_indices.json', 'r') as f:
    class_indices = json.load(f)

index_to_class = {v: k for k, v in class_indices.items()}
print(f"✓ Classes: {class_indices}")

# Get all images
image_files = [f for f in os.listdir(images_folder) 
               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

print(f"\n📸 Found {len(image_files)} images in '{images_folder}/'")
print("="*60)

results = []
correct_predictions = 0

for i, img_file in enumerate(image_files, 1):
    img_path = os.path.join(images_folder, img_file)
    
    try:
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction = model.predict(img_array, verbose=0)[0][0]
        
        # Get class
        predicted_class_idx = int(prediction > 0.5)
        predicted_class = index_to_class[predicted_class_idx]
        confidence = prediction if predicted_class_idx == 1 else 1 - prediction
        
        print(f"\n{i}. {img_file}")
        print(f"   Prediction: {predicted_class.upper()}")
        print(f"   Confidence: {confidence*100:.1f}%")
        
        # Ask user to verify
        actual = input(f"   Is this actually a {predicted_class}? (y/n): ").strip().lower()
        
        if actual == 'y':
            correct_predictions += 1
            print(f"   ✓ CORRECT")
        else:
            print(f"   ✗ INCORRECT")
        
        results.append({
            'filename': img_file,
            'predicted': predicted_class,
            'confidence': float(confidence),
            'correct': actual == 'y'
        })
        
    except Exception as e:
        print(f"   ❌ Error processing {img_file}: {e}")

# Summary
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"\nTotal Images: {len(image_files)}")
print(f"Correct Predictions: {correct_predictions}")
print(f"Incorrect Predictions: {len(image_files) - correct_predictions}")
print(f"Accuracy: {correct_predictions/len(image_files)*100:.1f}%")

# Save results
with open(os.path.join(images_folder, 'test_results.json'), 'w') as f:
    json.dump({
        'total': len(image_files),
        'correct': correct_predictions,
        'accuracy': correct_predictions/len(image_files),
        'predictions': results
    }, f, indent=4)

print(f"\n💾 Results saved to: {images_folder}/test_results.json")

# Show breakdown
cats_predicted = sum(1 for r in results if r['predicted'] == 'Cat')
dogs_predicted = sum(1 for r in results if r['predicted'] == 'Dog')

print(f"\n📊 Predictions Breakdown:")
print(f"   Cats predicted: {cats_predicted}")
print(f"   Dogs predicted: {dogs_predicted}")

print("\n" + "="*60)
print("FOR YOUR REPORT:")
print("="*60)
print(f"\n\"I tested the trained model on {len(image_files)} images downloaded")
print(f"from the internet. The model correctly predicted {correct_predictions} out of")
print(f"{len(image_files)} images, achieving {correct_predictions/len(image_files)*100:.1f}% accuracy on real-world images.\"")
print("\n" + "="*60)
