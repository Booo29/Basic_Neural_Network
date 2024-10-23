import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# 1. Load dataset MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# 2. Build nueral network model
model = Sequential([
    Flatten(input_shape=(28, 28)),  
    Dense(128, activation='relu'),   
    Dense(10, activation='softmax')  
])

# 3. Compile model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# 4. Train model
model.fit(x_train, y_train, epochs=5)

# 5. Save model
model.save("First_Model.h5") 
print("Model saved as First_Model.h5")
