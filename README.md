# Concrete Strength Prediction with Neural Networks 🏗️🔬

This project predicts the strength of concrete using a deep neural network built with Keras. It is a **regression problem**, where the model learns from various concrete composition features to estimate its final strength.

## Features ✨
- 📊 **Loads dataset from a CSV file**  
- 🔄 **Splits data into features and labels**  
- 🔧 **Normalizes feature values** for better training performance  
- 🎯 **Splits data into training and testing sets**  
- 🏗️ **Builds a deep neural network using Keras**  
- ⚙️ **Compiles the model with the Adam optimizer and MSE loss**  
- 🚀 **Trains the model and evaluates it using MAE and MSE**  

## Model Architecture 🏗️  
The neural network is built using Keras with multiple dense layers and ReLU activation functions:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)  # Output layer for regression
])
