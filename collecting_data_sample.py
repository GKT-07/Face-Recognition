import cv2

lface_deffaceclassify = cv2.CascadeClassifier('/root/PycharmProjects/Face_ID_recogonisation/haarcascade_frontalface_default.xml')
def faceclassifier(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = faceclassify.detectMultiScale(gray,1.3,5)
    if faces is():
        return  None
    for(x,y,w,h) in faces:
        croppedface = image[y:y+h, x:x+w]

    return croppedface

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if faceclassifier(frame) is not None:
        count+=1
        face=cv2.resize(faceclassifier(frame),(200,200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        file_name_path = '/root/PycharmProjects/Face_ID_recogonisation/trainingdata/image' +str(count)+'.jpg'
        cv2.imwrite(file_name_path,face)
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),3)
        cv2.namedWindow('Face Cropper')
        cv2.imshow('Face cropper',face)
    else:
        print("Face Not Found")
        pass
    if cv2.waitKey(1)==13 or count==100:
        break

cap.release()
cv2.destroyAllWindows()
print("Collecting Samplle Compllted !!!")