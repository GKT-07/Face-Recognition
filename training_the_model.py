import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
data_path = '/root/PycharmProjects/Face_ID_recogonisation/trainingdata/'
onfile = [f for f in listdir(data_path) if isfile(join(data_path, f))]

Trainingdata, Labels = [], []
for i,files in enumerate(onfile):
    imagepath = data_path + onfile[i]
    images = cv2.imread(imagepath,cv2.IMREAD_GRAYSCALE)
    Trainingdata.append(np.asarray(images,dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)
model = cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(Trainingdata),np.asarray(Labels))
print("Model Trainning Complete !!!")

faceclassify = cv2.CascadeClassifier('/root/PycharmProjects/Face_ID_recogonisation/haarcascade_frontalface_default.xml')

def facedetection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceclassify.detectMultiScale(gray,1.3,5)
    if faces is():
        return  image,[]
    global x
    global y
    for(x,y,w,z) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+z),(0,255,255),2)
        roi =image[y:y+z, x:x+w]
        roi = cv2.resize(roi, (200,200))


    return  image,roi
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    image, face = facedetection(frame)
    try:
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        result = model.predict(face)

        if result[1] < 500:
            confidence = int((1-(result[1])/300)*100)
            display_string = str(confidence) + '% Accuracy'

        cv2.putText(image,display_string,(200,50),cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)

        if confidence<83:
            cv2.putText(image, "", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 250, 0), 2)
            cv2.putText(image, "FACE", (x+10, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 250), 1)
            # cv2.namedWindow('Face ID  Verification')
            cv2.imshow("Face ID Verification",image)

        else:
            cv2.putText(image, "", (260, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 250), 2)
            cv2.putText(image, "UNLOCKED", (x - 35, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 250), 1)
            cv2.imshow("Face ID Verification", image)
    except:
        cv2.putText(image, "Face Not Found", (200, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 120, 255), 2)
        cv2.putText(image, "Locked", (260, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 250), 2)
        cv2.imshow("Face ID Verification",image)
        pass

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
