import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ─── Configuration ─────────────────────────────────────────────────────────────
IMG_SIZE    = 48
CLASSES     = ["angry","disgust","fear","happy","sad","surprise","neutral"]
WEIGHTS     = "best_weights.h5"
PROTO_PATH  = "deploy.prototxt"  # Ensure these files are in your directory
MODEL_PATH  = "res10_300x300_ssd_iter_140000.caffemodel"

# ─── Load Model & Face Detector ───────────────────────────────────────────────
model = load_model(WEIGHTS)
face_net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)

# ─── Start Webcam ──────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0)
    )
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            face = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            try:
                resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
            except:
                continue  # Skip if face region is too small or invalid

            normalized = resized.astype("float32") / 255.0
            normalized = np.expand_dims(normalized, axis=-1)
            normalized = np.expand_dims(normalized, axis=0)

            preds = model.predict(normalized)
            label = CLASSES[np.argmax(preds)]

            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Facial Expression Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
