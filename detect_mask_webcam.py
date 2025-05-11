import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load the trained model
model = load_model('best_model.h5')

# Load OpenCV's face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Setup for live plot
plt.ion()
fig, ax = plt.subplots()
labels = ['Mask', 'No Mask']
counts = [0, 0]
bar = ax.bar(labels, counts, color=['green', 'red'])
ax.set_ylim(0, 10)  # Adjust if needed

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Reset counts each frame
    counts = [0, 0]

    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        face = preprocess_input(np.expand_dims(face.astype("float32"), axis=0))

        pred = model.predict(face)[0][0]
        label = "Mask" if pred < 0.5 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Count for plot
        if label == "Mask":
            counts[0] += 1
        else:
            counts[1] += 1

        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Update bar chart
    for i, b in enumerate(bar):
        b.set_height(counts[i])
    ax.set_ylim(0, max(10, max(counts) + 1))
    plt.draw()
    plt.pause(0.001)

    # Show webcam frame
    cv2.imshow("Mask Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
