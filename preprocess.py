import os
import cv2
import numpy as np

def load_dataset(data_dir, img_size=128):
    images, labels = [], []
    class_names = os.listdir(data_dir)
    for idx, label in enumerate(class_names):
        class_folder = os.path.join(data_dir, label)
        for file in os.listdir(class_folder):
            img_path = os.path.join(class_folder, file)
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (img_size, img_size))
                images.append(img)
                labels.append(idx)
            except:
                continue
    return np.array(images), np.array(labels), class_names
    

