# Cats vs Dogs Image Classification using Convolutional Neural Networks

A deep learning project that builds a Convolutional Neural Network (CNN) from scratch in PyTorch for binary image classification on the Cats vs Dogs dataset. The project also explores transfer learning using pretrained ImageNet models and evaluates the impact of data augmentation techniques.

---

## Project Objectives

- Build a CNN from scratch using PyTorch
- Understand convolutional neural network architecture
- Train and evaluate a binary image classifier
- Compare a custom CNN against transfer learning models
- Investigate how data augmentation affects model performance

---

## Dataset

**Dataset:** Kaggle Dogs vs Cats

- 25,000 labelled images
- Binary classification
  - Cat
  - Dog

Dataset split:

- Training: 20,000 images
- Test: 5,000 images

---

## CNN Architecture

The custom CNN follows the architecture:

```
Input Image
      │
Conv2D
      │
ReLU
      │
Max Pool
      │
Conv2D
      │
ReLU
      │
Max Pool
      │
Flatten
      │
Fully Connected
      │
ReLU
      │
Fully Connected
      │
Output (Cat / Dog)
```

Model pipeline:

```
Conv1 → ReLU → MaxPool
      ↓
Conv2 → ReLU → MaxPool
      ↓
Flatten
      ↓
FC1 → ReLU
      ↓
FC2
```

---

## Training

Loss Function

- CrossEntropyLoss

Optimizer

- Adam

Learning Rate

- 0.001

Training includes:

- Forward propagation
- Backpropagation
- Weight updates
- Training loss tracking

---

## Evaluation

The trained model is evaluated on the held-out test dataset.

Metrics include:

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1 Score

These metrics provide insight into both the overall performance and the types of classification errors made by the model.

---

## Data Augmentation

To improve generalisation and reduce overfitting, the following augmentations are investigated:

- Random Horizontal Flip
- Random Rotation
- Color Jitter

Performance is compared before and after augmentation to measure the impact on test accuracy.

---

## Transfer Learning

The custom CNN is compared against pretrained ImageNet models:

- ResNet18
- MobileNetV2

Only the final classification layers are fine-tuned while the pretrained feature extractor is reused.

The comparison focuses on:

- Test Accuracy
- Training Time
- Convergence Speed
- Generalisation Performance

---

## Skills Practised

- PyTorch
- Convolutional Neural Networks
- Image Classification
- Data Augmentation
- Transfer Learning
- Model Evaluation
- Deep Learning Fundamentals

---

## Project Structure

```
catsdogs-cnn/
│
├── src/
│   ├── dataset.py
│   ├── eval.py
│   ├── model.py
│   ├── train.py
│   ├── transfer_learning.py
│   └── utils.py
│
├── .gitignore
├── catsdogs_model.pth
├── README.md
├── requirements.txt
├── training_loss_plot.png
```

---

## Future Improvements

- Batch Normalisation
- Dropout Regularisation
- Learning Rate Scheduling
- Early Stopping
- TensorBoard Logging
- Hyperparameter Optimisation
- Grad-CAM Visualisations

---

## Example Results

| Model | Accuracy |
|--------|----------|
| Custom CNN | TBD |
| Custom CNN + Augmentation | TBD |
| ResNet18 (Transfer Learning) | TBD |
| MobileNetV2 (Transfer Learning) | TBD |

---

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the model

```bash
python train.py
```

### Evaluate the model

```bash
python eval.py
```

---

## License

This project is intended for educational purposes and personal learning.
