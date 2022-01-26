from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np
import boto3
import os
from datetime import datetime
from urllib import request

app=Flask(__name__)

# Archivo de prueba local de reconocimiento facial
# Credenciales de Amazon
# Requisitos: Tener un bucket disponible, en este caso el bucket es "unida"

ACCESS_KEY = "AKIA25L3WLNND5PE2XQI"
SECRET_KEY = "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"

# Cliente s3 y listado de objetos del bucket

client = boto3.client(
    "s3",
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
)

response = client.list_objects_v2(Bucket="unida")
lista = response["Contents"]

# Obtener solo el "key" de los objects del bucket seleccionado y descargarlo en una ruta local ./img

for fichero in lista:
    classNames2 = fichero["Key"]
    # print(f"classNames2 :{classNames2} \n")
    client.download_file('unida', f'{classNames2}', f'./img/{classNames2}') 

# Listar los objetos descargados y obtener un array de cada objeto además de su nombre

path = './img'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    # print(cl)
    curImg = cv2.imread(f'{path}/{cl}')
    # print(curImg)
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
# print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # print(f"img: {img} \n")
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
        # print(encodeList)
    return encodeList



def markAttendance(name):
    with open('./Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name}, {dtString}')

encodeListKnown = findEncodings(images)
print('Encoding Complete')


def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)


        facesCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            # print(faceDis)
            matchIndex = np.argmin(faceDis)

            # if matches[matchIndex]:
            #     name = classNames[matchIndex].upper()
            #     print(name)
            #     y1,x2,y2,x1 = faceLoc
            #     x1,y1,x2,y2 = x1*4, y1*4, x2*4, y2*4
            #     cv2.rectangle(img, (x1,y1), (x2,y2), (0, 255, 0), 2)
            #     cv2.rectangle(img, (x1, y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            #     cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255),2)
            #     markAttendance(name)

            if faceDis[matchIndex] < 0.7:
                name = classNames[matchIndex].upper()
                markAttendance(name)
                y1,x2,y2,x1 = faceLoc
                x1,y1,x2,y2 = x1*10, y1*10, x2*10, y2*10
                cv2.rectangle(img, (x1,y1), (x2,y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2-35), (x2,y2), (0,255,0), cv2.FILLED)
                cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255),2)
            else: 
                name = 'Unknown'
                y1,x2,y2,x1 = faceLoc
                x1,y1,x2,y2 = x1*10, y1*10, x2*10, y2*10
                cv2.rectangle(img, (x1,y1), (x2,y2), (0, 0, 255), 2)
                cv2.rectangle(img, (x1, y2-35), (x2,y2), (0,0,255), cv2.FILLED)
                cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255),2)

                ret, buffer = cv2.imencode('.jpg', img)
                img = buffer.tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=='__main__':
    app.run(debug=True)