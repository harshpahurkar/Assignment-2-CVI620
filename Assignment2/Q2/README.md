# CVI620 Assignment 2 - Q2: MNIST Digit Classification

## Quick Start

### 1. Train All Classifiers
```bash
python train_mnist.py
```

This will train 6 different classifiers:
- Deep Neural Network (DNN)
- Convolutional Neural Network (CNN)
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Gradient Boosting

**Expected Duration:** 20-40 minutes (depending on hardware)

**Goal:** Achieve at least 90% accuracy ✓

### 2. Test and Analyze Results

```bash
python inference_mnist.py
```

**Available modes:**
1. Test random samples - Quick visual check
2. Find misclassified samples - Error analysis
3. Interactive mode - Test specific samples
4. Full test set evaluation - Complete accuracy

## What Gets Generated

After training, you'll find in `models_mnist/`:
- All trained models (.h5 for deep learning, .pkl for sklearn)
- `results.json` - Accuracy comparison
- `model_comparison.png` - Bar chart comparison
- `confusion_matrix_best.png` - Confusion matrix for best model
- `sample_images.png` - Example training samples

## Expected Results

You should achieve these accuracies:
- **CNN:** ~99% ✓
- **DNN:** ~98% ✓
- **Random Forest:** ~97% ✓
- **KNN:** ~97% ✓
- **Gradient Boosting:** ~95% ✓
- **SVM:** ~94% ✓

**All methods meet the 90% requirement!**

## Understanding the Data

The MNIST CSV files contain:
- First column: Label (0-9)
- Remaining 784 columns: Pixel values (28×28 flattened)
- Pixel values: 0-255 (grayscale)

## Detailed Testing Examples

### Test Random Samples
```bash
python inference_mnist.py
# Choose option 1
# Enter number of samples (e.g., 20)
```

### Find Misclassifications
```bash
python inference_mnist.py
# Choose option 2
# This helps understand model failures
```

### Interactive Testing
```bash
python inference_mnist.py
# Choose option 3
# Enter test sample indices (0-9999)
# Great for exploring specific cases
```

## Model Comparison

| Method | Speed | Accuracy | Memory | Best For |
|--------|-------|----------|--------|----------|
| CNN | Medium | Highest | High | Best overall |
| DNN | Fast | High | Medium | Fast training |
| Random Forest | Fast | Good | Medium | Interpretable |
| KNN | Slow | Good | Low | Simple baseline |
| SVM | Slow | Good | Medium | Small datasets |
| Gradient Boost | Medium | Good | Medium | Tabular data |

## Tips for Best Results

1. **CNN** is the best performer for image data
2. **Random Forest** is great for quick experiments
3. **DNN** offers good balance of speed and accuracy
4. **KNN** is simple but slower for prediction
5. **SVM** works well but doesn't scale to full dataset

## Troubleshooting

**Training is slow:**
- Deep learning models use GPU automatically if available
- SVM trains on 10k subset by default (can increase)
- Random Forest uses all CPU cores

**Out of Memory:**
- Reduce batch size for neural networks
- Train models separately instead of all at once

**Low Accuracy:**
- Ensure data is normalized (done automatically)
- Check that CSV files are loaded correctly
- Verify train/test split is working

## Analysis Questions

After training, consider:
1. Why does CNN outperform others?
2. Which digits are most commonly confused?
3. What patterns exist in misclassifications?
4. How do traditional ML compare to deep learning?

View the confusion matrix to answer these questions!
