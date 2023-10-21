import cv2
from tkinter import filedialog
import tkinter as tk
from PIL import Image, ImageTk
import os
import numpy as np
import pandas as pd
import tensorflow as tf

data = pd.read_csv("label_names.csv")
reconstructed_model = tf.keras.models.load_model('model.h5')

cap = None  # Global variable for video capture
video_running = False  # Flag variable to track video feed status

def classify_image(image):
    resized = cv2.resize(image, (50, 50))

    result = np.expand_dims(resized, axis=0)

    result = reconstructed_model.predict(result)
    rslt = np.argmax(result)

    matched_row = data[data.index == rslt]
    if not matched_row.empty:
        name = matched_row['SignName'].values[0]
        result_text.set(f"The Traffic Sign is: {name}, ClassId:{rslt}")
    else:
        result_text.set(f"No matching ClassId found: {rslt}")
   
    image = Image.fromarray(image)
    image.thumbnail((400, 400))
    image = ImageTk.PhotoImage(image)
    video_label.configure(image=image)
    video_label.image = image

def capture_frame():
    if video_running:
        ret, frame = cap.read()
        if ret:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            classify_image(image)

        video_label.after(5, capture_frame)

def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("PNG files", "*.png")])
    if file_path:
        if video_running:
            stop_video()
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        classify_image(image)


def capture_image():
    if video_running:
        stop_video()
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        classify_image(image)
        cv2.imwrite("captured_image.png", frame)
    cap.release()
    
def start_video():
    global cap, video_running
    if not video_running:
        cap = cv2.VideoCapture(0)
        video_running = True
        capture_frame()

def stop_video():
    global cap, video_running
    if video_running:
        cap.release()
        video_running = False

root = tk.Tk()
root.geometry("500x600")
root.title("ROAD SIGN DETECTION & RECOGNITION")

video_label = tk.Label(root)
video_label.pack(pady=10)

select_button = tk.Button(root, text="Select Image", command=select_image, width= 15)
select_button.pack(pady=5)

capture_button = tk.Button(root, text="Capture Image", command=capture_image, width= 15)
capture_button.pack(pady=5)

start_button = tk.Button(root, text="Start Video", command=start_video, width= 15)
start_button.pack(pady=5)

stop_button = tk.Button(root, text="Stop Video", command=stop_video, width= 15)
stop_button.pack(pady=5)

result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, font=('Arial', 10), wraplength=400)
result_label.pack(pady=10)

root.mainloop()
