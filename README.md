OVERVIEW :

This project is a facial recognition system developed for security applications such as identity verification, access control, and surveillance. It uses real-time webcam input to detect and recognize human faces using classical computer vision techniques. The system captures face images, trains a recognition model, and performs real-time identification, making it suitable for use in secure environments like offices, labs, or homes.

TECHNOLOGIES USED :

Python 3.x
OpenCV (cv2)
Haar Cascade Classifier for face detection
LBPH Face Recognizer for training and recognition
NumPy for data handling
FEATURES :

1.Real-time face detection using Haar Cascades

2.Face recognition using LBPH (Local Binary Patterns Histograms)

3.Live webcam feed for capturing and recognizing faces

4.Automatic dataset creation by saving detected faces

5.Trains a model and stores it for future recognition

6.Displays the identity of recognized individuals or labels them as "Unknown"

WORKING :

Capture Faces--> The system detects faces from webcam input and saves them to a folder.

Train Model--> It scans the saved images, assigns labels, and trains the recognizer.

Recognize Faces--> During live video feed, it identifies known faces and displays their names.

USE CASES :

Door access control
Secure room entry
Office attendance tracking
Home monitoring systems
CONCLUSION :

This facial recognition system provides a real-time solution for identifying and verifying individuals. Upon running the system, the webcam captures faces, which are then processed and saved for training. Once the model is trained, the system continuously recognizes faces from the live video feed. When a face is recognized, the system displays the corresponding name on the screen. If the face is not recognized, it labels the individual as "Unknown." The confidence level of the recognition is also shown, giving insight into how certain the system is about its predictions. This setup offers a reliable, offline facial recognition solution for security applications.
