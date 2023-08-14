# Real Time Face Recognition using OpenCV-Python 

# Import the opencv module
import cv2

# Load the pre-trained Haar Cascade classifier for face detection
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize video capture from the default webcam (camera index 0)
cap = cv2.VideoCapture(0)

# Set the width and height of the captured video frame
cap.set(3, 640)
cap.set(4, 480)

try:
    # Start an infinite loop for capturing and processing video frames
    while True:
        # Read a frame from the video capture
        ret, img = cap.read()

        # Convert the captured frame to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        # Iterate through each detected face
        for (x, y, w, h) in faces:
            # Draw a blue rectangle around the detected face
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display the processed frame with rectangles around detected faces
        cv2.imshow('video', img)

        # Wait for a key press and store the key's ASCII value
        key = cv2.waitKey(30) & 0xFF

        # If the 'Esc' key (ASCII value 27) is pressed, exit the loop
        if key == 27:
            break

finally:
    # Release the video capture and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

