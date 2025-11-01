# -CIFAR-10-Object-Recognition-

This project performs object recognition on the CIFAR-10 dataset using Transfer Learning with a ResNet50 convolutional base (pretrained on ImageNet).
It classifies images into 10 object categories such as airplane, automobile, bird, cat, dog, frog, horse, ship, truck, and deer.

📌 Project Description

The notebook downloads the CIFAR-10 dataset from Kaggle, extracts it, preprocesses all images, and applies a ResNet50-based deep learning model for classification.
By leveraging a pretrained ResNet50 model as a feature extractor, it achieves robust performance even on limited data.

🧩 Workflow
🔹 1. Dataset Download and Extraction

Downloaded dataset from Kaggle: cifar-10 competition.

Extracted .zip and .7z archives using ZipFile and py7zr.

Verified extracted data directories:

/content/train/

trainLabels.csv

🔹 2. Data Loading

Loaded training image files and corresponding labels using Pandas.

Displayed examples and verified image-label mapping.

🔹 3. Data Preprocessing

Converted all images to NumPy arrays using PIL.

Resized images to 256×256 pixels to match ResNet50 input shape.

Normalized pixel values by dividing by 255.

Encoded categorical labels into numerical form:

{'airplane':0, 'automobile':1, 'bird':2, 'cat':3, 'deer':4,
 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}


Split dataset into training and testing sets using train_test_split.

🧠 Model Architecture
🏗️ Base Model: ResNet50
convontial_base = ResNet50(weights='imagenet', include_top=False, input_shape=(256,256,3))
convontial_base.summary()


Weights: Pretrained on ImageNet

Top Layers: Removed (include_top=False)

Purpose: Acts as a feature extractor for image embeddings

🧱 Custom Classifier Head

Added on top of ResNet50:

Flatten Layer — Converts extracted features into 1D

Dense(256, activation='relu') — Fully connected layer

Dropout(0.5) — Prevents overfitting

Dense(10, activation='softmax') — Output layer for 10 CIFAR-10 classes

⚙️ Compilation
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

🧮 Training

Trained for multiple epochs (as defined in notebook).

Used a validation split to monitor accuracy during training.

🧠 Model Summary

Base: ResNet50 (pretrained)

Input Size: 256×256×3

Output Classes: 10

Loss Function: Sparse Categorical Crossentropy

Optimizer: Adam

Metric: Accuracy

🛠️ Tech Stack

Language: Python

Libraries Used:

TensorFlow / Keras

NumPy

Pandas

Matplotlib

OpenCV

PIL (Pillow)

scikit-learn

py7zr

📊 Files Included

📁 CIFAR-10 Object Recognition.ipynb

Main Jupyter Notebook containing dataset download, extraction, preprocessing, feature extraction using ResNet50, model training, and evaluation.

💡 Key Highlights

✅ Implemented Transfer Learning with pretrained ResNet50
✅ Extracted deep features for improved accuracy on CIFAR-10
✅ Built and trained a custom classifier on top of ResNet50
✅ Included full data preprocessing, visualization, and model evaluation
✅ Demonstrates a clean end-to-end image recognition workflow
