import cv2
import numpy as np
import os 
from cvzone.PoseModule import PoseDetector
import cvzone

# Face detection setup
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX
id = 0
names = ['Clebson', 'Clebson', 'Paula', 'Ilza', 'Z', 'W'] 

detector = PoseDetector()

cam = cv2.VideoCapture(0)
cam.set(3, 1280) 
cam.set(4, 720) 

minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:
    ret, img = cam.read()
    img = cv2.resize(img,(1280,720))

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "Clebson"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  

    resultado = detector.findPose(img, draw=False)
    pontos,bbox = detector.findPosition(img, draw=False)
    if len(pontos)>=1:
        x = min(point[0] for point in pontos)
        y = min(point[1] for point in pontos)
        w = max(point[0] for point in pontos) - x
        h = max(point[1] for point in pontos) - y

        # Decrease the height of the bounding box by 15%
        y += int(h * 0.11)
        h -= int(h * 0.11)

        cabeca = pontos[0][1]
        joelho = pontos[26][1]
        diferenca = joelho-cabeca

        if diferenca <=0:
            cvzone.putTextRect(img,'QUEDA DETECTADA',(x+122,y+100),scale=2,thickness=2,colorR=(0,0,255))


        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff 
    if k == 27:
        break

print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
