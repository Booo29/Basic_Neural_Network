from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.datasets import mnist

model = load_model("Three_Model.keras")

(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0 

predictions = model.predict(x_test[:10])
print("Prediction: ", np.argmax(predictions, axis=1))
print("Real tags ", y_test[:10]) 
