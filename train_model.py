import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

X, y = [], []
for folder in os.listdir("gestures"):
    for file in os.listdir(f"gestures/{folder}"):
        path = f"gestures/{folder}/{file}"
        img = cv2.imread(path)
        img = cv2.resize(img, (64, 64))
        X.append(img)
        y.append(folder)

X = np.array(X) / 255.0

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
joblib.dump(label_encoder, "labels.pkl")

y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)

model.save("sign_model.h5")
print("✅ Model trained and saved as 'sign_model.h5'")
