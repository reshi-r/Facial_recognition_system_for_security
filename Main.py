import cv2
from cv2 import face
import os
import numpy as np

# Initialize the OpenCV face detector (Haar Cascade Classifier)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize LBPH face recognizer
recognizer = face.LBPHFaceRecognizer_create()

# Directory where known faces are stored
known_faces_dir = 'known_faces'
if not os.path.exists(known_faces_dir):
    os.makedirs(known_faces_dir)

# Create a function to capture and save faces for training
def capture_faces():
    # Initialize the webcam for face capture
    video_capture = cv2.VideoCapture(0)

    # Ensure that the webcam opened correctly
    if not video_capture.isOpened():
        print("Could not open video stream.")
        return

    # Start capturing video
    while True:
        ret, frame = video_capture.read()
        
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the video frame
        faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces_detected:
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Crop the face region from the frame for saving
            face_region = gray[y:y+h, x:x+w]

            # Save the face image (in 'known_faces' directory)
            # We use a unique filename for each capture
            face_id = len(os.listdir(known_faces_dir)) + 1
            cv2.imwrite(f'{known_faces_dir}/face_{face_id}.jpg', face_region)

            # Display message on screen
            cv2.putText(frame, f"Face {face_id} saved", (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Display the frame with face detection
        cv2.imshow("Face Capture", frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    video_capture.release()
    cv2.destroyAllWindows()

# Global dictionary to store the names and their corresponding labels
label_dict = {}

# Function to train the recognizer on saved faces
def train_face_recognizer():
    faces = []
    labels = []
    label_counter = 0

    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg"):
            # Load the image
            image_path = os.path.join(known_faces_dir, filename)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces_detected:
                # Crop the face from the image
                face_image = gray[y:y+h, x:x+w]
                
                # Extract the label (person's name) from the filename (without extension)
                person_name = filename.split('.')[0]
                
                # If this person name hasn't been seen before, add them to the label dictionary
                if person_name not in label_dict:
                    label_dict[person_name] = label_counter
                    label_counter += 1

                # Add the face and the corresponding label (as an integer)
                faces.append(face_image)
                labels.append(label_dict[person_name])

    # Convert labels to a numpy array of integers
    labels = np.array(labels, dtype=np.int32)

    # Train the recognizer with the faces and their labels
    recognizer.train(faces, labels)
    recognizer.save("face_recognizer.yml")  # Save the trained model to disk

# Capture faces from the webcam and save them
print("Press 'q' to stop capturing faces...")
capture_faces()

# Train the recognizer on the captured faces
print("Training the recognizer...")
train_face_recognizer()

# Initialize the webcam for live video capture (face recognition)
video_capture = cv2.VideoCapture(0)

# Start capturing video for recognition
while True:
    ret, frame = video_capture.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the video frame
    faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces_detected:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop the face region from the frame for recognition
        face_region = gray[y:y+h, x:x+w]

        # Predict the label (name) of the face
        label, confidence = recognizer.predict(face_region)

        # Debugging the label and confidence values
        print(f"Label: {label}, Confidence: {confidence}")

        # Display the name of the recognized face
        name = list(label_dict.keys())[list(label_dict.values()).index(label)] if confidence < 100 else "Unknown"
        
        print(f"Recognized as: {name}")

        cv2.putText(frame, name, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Display the frame with face detection and recognition
    cv2.imshow("Face Recognition", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
