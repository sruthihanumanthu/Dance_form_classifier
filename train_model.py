import tensorflow as tf
from sklearn.model_selection import train_test_split
from preprocess import load_dataset

# Load data
X, y, class_names = load_dataset("dataset/")
X = X / 255.0
y = tf.keras.utils.to_categorical(y, num_classes=len(class_names))

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save
model.save("dance_model.h5")
with open("labels.txt", "w") as f:
    f.write("\n".join(class_names))

