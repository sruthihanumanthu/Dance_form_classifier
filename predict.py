import tensorflow as tf
import numpy as np
import cv2

# Load model
model = tf.keras.models.load_model("dance_model.h5")
with open("labels.txt") as f:
    class_names = f.read().splitlines()

def predict_dance_form(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return class_names[np.argmax(prediction)]

# Example
print(predict_dance_form("dataset/Kathak/img1.jpg"))


