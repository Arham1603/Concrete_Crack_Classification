
#%%
# Import packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, callbacks
import numpy as np
import matplotlib.pyplot as plt
import os, datetime
from sklearn.model_selection import train_test_split
import shutil
# %%
# Import dataset
dataset_concrete = r"C:\Users\ACER\Documents\SHRDC_AI\8) Capstone\1) CV\Dataset"

# List all classes
classes = os.listdir(dataset_concrete)

# Initialize lists to store file paths and labels
file_paths = []
labels = []

# Iterate through each class
for label, class_name in enumerate(classes):
    class_path = os.path.join(dataset_concrete, class_name)
    
    # Iterate through each file in the class
    for file_name in os.listdir(class_path):
        file_path = os.path.join(class_path, file_name)
        
        # Append the file path and label to the lists
        file_paths.append(file_path)
        labels.append(label)

# Split the data into training, validation, and test sets
train_files, test_files, train_labels, test_labels = train_test_split(
    file_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

train_files, val_files, train_labels, val_labels = train_test_split(
    train_files, train_labels, test_size=0.2, random_state=42, stratify=train_labels
)
# %%
# Define your destination directories
train_dir = r"C:\Users\ACER\Documents\SHRDC_AI\8) Capstone\1) CV\Split_Dataset\train"
validation_dir = r"C:\Users\ACER\Documents\SHRDC_AI\8) Capstone\1) CV\Split_Dataset\validation"
test_dir = r"C:\Users\ACER\Documents\SHRDC_AI\8) Capstone\1) CV\Split_Dataset\test"

# Create directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Copy files to the training directory
for src_path, label in zip(train_files, train_labels):
    class_name = classes[label]
    dst_path = os.path.join(train_dir, class_name, os.path.basename(src_path))
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy(src_path, dst_path)

# Copy files to the validation directory
for src_path, label in zip(val_files, val_labels):
    class_name = classes[label]
    dst_path = os.path.join(validation_dir, class_name, os.path.basename(src_path))
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy(src_path, dst_path)

# Copy files to the test directory
for src_path, label in zip(test_files, test_labels):
    class_name = classes[label]
    dst_path = os.path.join(test_dir, class_name, os.path.basename(src_path))
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy(src_path, dst_path)
# %%
