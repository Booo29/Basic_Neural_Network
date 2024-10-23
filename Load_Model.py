from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.datasets import mnist

# Load the model
model = load_model("First_Model.h5")

# Load the dataset
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0  # Normalizar los datos

# Make predictions with the model and print the results for the first 5 images
predictions = model.predict(x_test[:5])
print("Prediction: ", np.argmax(predictions, axis=1))
print("Real tags ", y_test[:5]) 
