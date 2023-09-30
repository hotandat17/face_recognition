import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'photos'
images = []
imgLabel = []
mylst = os.listdir(path)

for cl in mylst:
    curimg = cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    imgLabel.append(os.path.splitext(cl)[0])

def findEncodings(images): #Xác định một chức năng để tìm mã hóa khuôn mặt
    encodLst = []
    for img in images:
        # Chuyển đổi hình ảnh BGR sang RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Sử dụng mô hình CNN được đào tạo trước để tìm mã hóa khuôn mặt
        face_encodings = face_recognition.face_encodings(img_rgb)

        if len(face_encodings) > 0:
            encodLst.append(face_encodings[0])
        else:
            encodLst.append(None)

    return encodLst

encodlstKnowFaces = findEncodings(images)#Nhận mã hóa khuôn mặt cho các khuôn mặt đã biết

def attendance(name):
    with open('diemdanh.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tStr},{dStr}')

webcam = cv2.VideoCapture(0)
nm = "a"

while True:
    success, img = webcam.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrm = face_recognition.face_locations(imgS)
    encodeCurFrm = face_recognition.face_encodings(imgS, faceCurFrm)

    for encodFace, faceLocation in zip(encodeCurFrm, faceCurFrm):
        if encodFace is not None:
            # Compare face encodings using numpy (instead of face_recognition.compare_faces)
            face_distances = np.linalg.norm(encodlstKnowFaces - encodFace, axis=1)
            machesIndex = np.argmin(face_distances)

            if face_distances[machesIndex] < 0.5:
                name = imgLabel[machesIndex].upper()
                # print(name)
                y1, x2, y2, x1 = faceLocation
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1+6, y2-6),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                crTime = datetime.now().time()
                crDate = datetime.now().date()
                if name != nm:
                    attendance(name)
                    nm = name

    cv2.imshow('Frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
