import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('First_Model.h5')

window = tk.Tk()
window.title("Clasificador de Dígitos MNIST")
window.configure(bg='#f0f0f0')

canvas_width = 300 
canvas_height = 300
canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg="white", relief="raised", bd=5)
canvas.grid(row=0, column=0, padx=20, pady=20, columnspan=3) 
image = Image.new("L", (canvas_width, canvas_height), color=255)
draw = ImageDraw.Draw(image)

def paint(event):
    x1, y1 = (event.x - 5), (event.y - 5)
    x2, y2 = (event.x + 5), (event.y + 5)
    canvas.create_oval(x1, y1, x2, y2, fill="black", width=10)
    draw.ellipse([x1, y1, x2, y2], fill=0)

canvas.bind("<B1-Motion>", paint)

def predict_digit():
    resized_image = image.resize((28, 28))
    inverted_image = ImageOps.invert(resized_image)
    img_array = np.array(inverted_image) / 255.0
    img_array = img_array.reshape(1, 28, 28)

    prediction = model.predict(img_array)
    digit = np.argmax(prediction)

    result_label.config(text=f"Predicción: {digit}")

def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_width, canvas_height], fill="white")
    result_label.config(text="Predicción: ")

predict_button = tk.Button(window, text="Predecir Dígito", command=predict_digit, bg="#4CAF50", fg="black", font=("Helvetica", 12))
predict_button.grid(row=1, column=0, padx=10, pady=10)

clear_button = tk.Button(window, text="Limpiar", command=clear_canvas, bg="#f44336", fg="black", font=("Helvetica", 12))
clear_button.grid(row=1, column=1, padx=10, pady=10)

result_label = tk.Label(window, text="Predicción: ", font=("Helvetica", 16), bg="#f0f0f0")
result_label.grid(row=1, column=2, padx=10, pady=10)

window.mainloop()
