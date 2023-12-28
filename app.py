# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np

# Load and preprocess the dataset 
# Temporary Pseudocode
X, y = data_loading() 
X = X / 255.0 # Normalize the images
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Buld the model 
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax') # Will go off the temporary assumption of a Binary Classification.
])