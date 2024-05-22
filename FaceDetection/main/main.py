import threading
from cvzone.PoseModule import PoseDetector
import cv2
import cvzone
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import time

# Constants
MY_ADDRESS = 'clebsonsouza055@gmail.com'
PASSWORD = 'tijizudpnakhpzwv'
TO_ADDRESS = 'clebsonsouza55@outlook.com'  
EMAIL_INTERVAL = 60 

# Load the cascade for face detection
faceCascade = cv2.CascadeClassifier(r'C:\Users\eusoe\Downloads\OpenCV-Face-Recognition-master\OpenCV-Face-Recognition-master\FaceDetection\Cascades\haarcascade_frontalface_default.xml')

def send_email(subject, body):
    s = smtplib.SMTP(host='smtp.gmail.com', port=587)
    s.starttls()
    s.login(MY_ADDRESS, PASSWORD)

    msg = MIMEMultipart()
    msg['From'] = MY_ADDRESS
    msg['To'] = TO_ADDRESS
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    s.send_message(msg)
    s.quit()

def detect_fall(pontos):
    cabeca = pontos[0][1]
    joelho = pontos[26][1]
    diferenca = joelho-cabeca
    return diferenca <= 0

# Use the webcam instead of the video file
video = cv2.VideoCapture('output.mp4')
detector = PoseDetector()

last_email_time = 0
email_thread = None  # Initialize the email thread

while True:
    check, img = video.read()
    if not check:
        print("Failed to read video")
        break

    resultado = detector.findPose(img, draw=False)
    pontos, bbox = detector.findPosition(img, draw=False)

    # Convert color image to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    if len(pontos) >= 1:
        x = min(point[0] for point in pontos)
        y = min(point[1] for point in pontos)
        w = max(point[0] for point in pontos) - x
        h = max(point[1] for point in pontos) - y

        # Decrease the height of the bounding box by 15%
        y += int(h * 0.11)
        h -= int(h * 0.11)

        if detect_fall(pontos):
            cvzone.putTextRect(img, 'Queda!', (x, y - 60), scale=3, thickness=2, colorR=(0, 0, 255))
            
            current_time = time.time()
            if current_time - last_email_time > EMAIL_INTERVAL:

                email_thread = threading.Thread(target=send_email, args=("Fall Detected", "A fall has been detected in the video feed."))
                email_thread.start()
                last_email_time = current_time

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


    cv2.imshow('IMG', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if email_thread is not None:
    email_thread.join()

video.release()
cv2.destroyAllWindows()
