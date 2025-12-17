import cv2
import numpy as np
import face_recognition
import os
USE_HAAR = False        # True = Haar Cascade
USE_DNN = True          # True = DNN Face Detector
USE_RECOGNITION = True # True = Face Recognition
haar = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# DNN Face Detector
prototxt = "deploy.prototxt"
model = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)
known_encodings = []
known_names = []

if USE_RECOGNITION:
    for file in os.listdir("known_faces"):
        img = face_recognition.load_image_file(f"known_faces/{file}")
        enc = face_recognition.face_encodings(img)[0]
        known_encodings.append(enc)
        known_names.append(file.split(".")[0])
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = []
    if USE_HAAR:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haar.detectMultiScale(gray, 1.3, 5)
        faces = [(x, y, x+w, y+h) for (x,y,w,h) in faces]
    if USE_DNN:
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300,300), (104,177,123)
        )
        net.setInput(blob)
        detections = net.forward()
        faces = []

        for i in range(detections.shape[2]):
            conf = detections[0,0,i,2]
            if conf > 0.5:
                box = detections[0,0,i,3:7] * np.array([w,h,w,h])
                faces.append(box.astype("int"))
    locations = [(y1,x2,y2,x1) for (x1,y1,x2,y2) in faces]
    names = []

    if USE_RECOGNITION and locations:
        encodings = face_recognition.face_encodings(rgb, locations)
        for enc in encodings:
            matches = face_recognition.compare_faces(
                known_encodings, enc, tolerance=0.5
            )
            name = "Unknown"
            if True in matches:
                name = known_names[matches.index(True)]
            names.append(name)
    else:
        names = ["Face"] * len(faces)
    for (x1,y1,x2,y2), name in zip(faces, names):
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, name, (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0,255,0), 2)

    cv2.imshow("Face Detection & Recognition", frame)

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()

cv2.destroyAllWindows()
