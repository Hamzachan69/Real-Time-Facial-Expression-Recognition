import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random

# ─── Configuration ─────────────────────────────────────────────────────────────
IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 30
DATA_DIR = "data/train"
CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
NUM_CLASSES = len(CLASSES)
OUTPUT_WEIGHTS = "best_weights.h5"

PROTO_PATH = "deploy.prototxt"
MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"

# ─── Load Face Detector ────────────────────────────────────────────────────────
dnn_net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)

def detect_face_dnn(image):
    """Uses OpenCV DNN to detect the most confident face."""
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    dnn_net.setInput(blob)
    detections = dnn_net.forward()

    h, w = image.shape[:2]
    max_conf = 0
    face_box = None
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5 and confidence > max_conf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            face_box = box.astype("int")
            max_conf = confidence

    return face_box

def preprocess_image(img_path):
    """Loads, detects, crops, resizes, and normalizes a face."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_box = detect_face_dnn(img)
    if face_box is None:
        return None
    x1, y1, x2, y2 = face_box
    face = gray[max(0, y1):y2, max(0, x1):x2]
    if face.size == 0:
        return None
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = face.astype("float32") / 255.0
    return np.expand_dims(face, -1)

def load_dataset(data_dir):
    X, y = [], []
    print("Scanning dataset...")
    for idx, emotion in enumerate(CLASSES):
        folder = os.path.join(data_dir, emotion)
        files = os.listdir(folder)
        random.shuffle(files)
        for fname in tqdm(files, desc=f"Loading '{emotion}'"):
            face = preprocess_image(os.path.join(folder, fname))
            if face is not None:
                X.append(face)
                y.append(idx)
    X = np.array(X)
    y = to_categorical(y, NUM_CLASSES)
    return X, y

# ─── Load & Split Data ─────────────────────────────────────────────────────────
print("Loading images...")
X, y = load_dataset(DATA_DIR)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

# ─── Build Model ───────────────────────────────────────────────────────────────
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ─── Train Model ───────────────────────────────────────────────────────────────
checkpoint = ModelCheckpoint(OUTPUT_WEIGHTS, monitor="val_accuracy", save_best_only=True, verbose=1)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint]
)

model.save("final_model.h5")
print("Training complete. Best weights saved to", OUTPUT_WEIGHTS)
