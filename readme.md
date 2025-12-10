## Object Recognition using Pre-trained CNN Models
This project demonstrates image classification using a pre-trained Convolutional Neural Network (CNN).
The model is fine-tuned on a custom dataset to accurately recognize objects from images, leveraging transfer learning techniques in PyTorch.

## Project Overview
Deep learning models such as ResNet, VGG, and MobileNet come pre-trained on large datasets like ImageNet, giving them strong feature extraction capabilities.
In this project, we:

Load a pre-trained CNN model
Replace the final classification layer
Fine-tune the model on a custom image dataset
Train, validate, and evaluate the classifier
Test the model with real-world example images

## Features
✔ Pre-trained CNN backbone (ResNet18 / VGG16 / MobileNetV2)
✔ Custom image dataset loading with torchvision.datasets.ImageFolder
✔ Transformations: resizing, normalization, augmentation
✔ Training loop with loss tracking and optimizer
✔ GPU support (if available)
✔ Model saving & loading
✔ Inference pipeline for new images

## Technologies Used
Python
PyTorch
Torchvision
Pillow (PIL)
NumPy
Matplotlib (for training plots)

## Object Recognition Example
Image: sample_image.jpg
Predicted Class: 'Dog'

Image: objects/table.jpg
Predicted Class: 'Table'


