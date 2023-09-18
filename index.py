import dlib
import cv2

# Load the pre-trained face detection model from dlib
detector = dlib.get_frontal_face_detector()

# Load the image
image = cv2.imread(r"C:\Users\thukk\Desktop\dhoni1.png")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = detector(gray)

# Draw rectangles around the detected faces
for face in faces:
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)

# Display the image with detected faces
cv2.imshow('Faces Detected', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
