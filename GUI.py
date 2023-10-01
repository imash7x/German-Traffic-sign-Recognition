import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('model.h5')

# Create a Tkinter GUI
root = tk.Tk()
root.title("Traffic Sign Detection")
canvas = tk.Canvas(root, width=600, height=400)
canvas.pack()

# Define a function to select an image using the file dialog and display it on the canvas
def select_image():
    file_path = filedialog.askopenfilename()
    image = Image.open(file_path)
    image = image.resize((600, 400))
    photo = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, image=photo, anchor=tk.NW)
    detect_sign(file_path)

# Define a function to detect the traffic sign in the selected image using the loaded model
def detect_sign(file_path):
    image = cv2.imread(file_path)
    image = cv2.resize(image, (32, 32))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    class_id = np.argmax(prediction)
    label.configure(text=f"Traffic sign detected: {class_id}")

# Create a button widget
button = tk.Button(root, text="Select Image", command=select_image)
button.pack()

label = tk.Label(root, text="")
label.pack()

# Start the main event loop
root.mainloop()
