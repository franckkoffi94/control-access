import numpy as np
import cv2

import time
from datetime import datetime
from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage


app1= ClarifaiApp(api_key='903d4e1b000240cfa4e94f0858c268a2')
model1 = app1.models.get('my-first-application')

face_cascade = cv2.CascadeClassifier('haar-face.xml')

cap = cv2.VideoCapture(0)
print( 'camera is initialized')

while True:
   
    ret, img = cap.read()
    if ret:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            picname = datetime.now().strftime("%y-%m-%d-%H-%M")
            picname = picname+'.jpg'
            cv2.imwrite(picname,img)
            print ("Saving Photo")
            image = ClImage(file_obj=open(picname, 'rb'))
            response=model1.predict([image])
            data1 = response['outputs'][0]['data']['concepts']
            print(data1)
            for row in data1: 
                if row['name'] == 'helmet':
                    if row['value']>= 1.590712e-08:
                        x=1
                    else:
                        print ('please wear the helmet')
            time.sleep(2)
                 
            
                
    cv2.imshow('img',img)
    time.sleep(0.1)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
