import os
import face_recognition
import numpy as np
import cv2

# Directory containing the dataset
dataset_dir = 'dataset/'

# Initialize arrays to store encodings and labels
encodings = []
labels = []

# Iterate over each person in the dataset
for person_name in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person_name)

    # Iterate over each image of the person
    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
        image = face_recognition.load_image_file(image_path)

        # Detect face locations and encodings
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        # Assuming one face per image, store the encoding and the label
        if face_encodings:
            encodings.append(face_encodings[0])
            labels.append(person_name)

# Convert to numpy arrays
encodings = np.array(encodings)
labels = np.array(labels)











def recognize_face(image_path):
    # Load and process the image
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    for face_encoding in face_encodings:
        # Predict the person's name
        predictions = classifier.predict_proba([face_encoding])
        best_match_index = np.argmax(predictions[0])
        label = label_encoder.inverse_transform([best_match_index])
        print(f"Recognized as: {label[0]}")










test_image_path = 'path_to_test_image.jpg'
recognize_face(test_image_path)










# Initialize video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the image from BGR color (OpenCV default) to RGB color
    rgb_frame = frame[:, :, ::-1]

    # Find all face locations and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        predictions = classifier.predict_proba([face_encoding])
        best_match_index = np.argmax(predictions[0])
        label = label_encoder.inverse_transform([best_match_index])

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Label the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, label[0], (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
video_capture.release()
cv2.destroyAllWindows()
