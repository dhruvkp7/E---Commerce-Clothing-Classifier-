import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from collections import Counter

# Load and analyze dataset distribution
data_dir = 'path/to/dataset'
classes = os.listdir(data_dir)
print(f"Classes found: {classes}")

# Check class distribution
class_counts = Counter([label for label in os.listdir(data_dir) for _ in os.listdir(os.path.join(data_dir, label))])
print(f"Class distribution: {class_counts}")

# Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# Preview augmented images
sample_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=1,
    class_mode='categorical',
    subset='training'
)

img, label = next(sample_generator)
plt.imshow(img[0])
plt.title(f"Sample Augmented Image: {classes[np.argmax(label)]}")
plt.show()
