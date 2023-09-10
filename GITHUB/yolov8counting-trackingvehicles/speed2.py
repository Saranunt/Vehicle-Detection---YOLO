import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import*
import time
from math import dist
model=YOLO('yolov5su.pt')



def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('5.3 4K Camera Road in Thailand No 3 Interchange.mp4')


my_file = open("GITHUB\yolov8counting-trackingvehicles\coco.txt")
data = my_file.read()
class_list = data.split("\n") 

count=0

tracker=Tracker()
#tracker = cv2.TrackerKCF_create()

cy1=322
cy2=368

offset=6

vh_up={}
counter1=[]

while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
   

    results=model.predict(frame)

    a=results[0].boxes.boxes
    px=pd.DataFrame(a).astype("float")
    list=[]
             
    for index,row in px.iterrows():
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'car' or 'motorcycle' or 'person' or 'truck' or 'bus' in c:
            list.append([x1,y1,x2,y2])
    bbox_id=tracker.update(list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        
        cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2)
        
 
        if cy2<(cy+offset) and cy2 > (cy-offset):
           vh_up[id]=time.time()
        if id in vh_up:

           if cy1<(cy+offset) and cy1 > (cy-offset):
             elapsed1_time=time.time() - vh_up[id]

             if counter1.count(id)==0:
                counter1.append(id)      

           else:
             elapsed1_time=time.time() - vh_up[id] + 1
             distance1 = 10 # meters
             a_speed_ms1 = distance1 / elapsed1_time
             a_speed_kh1 = a_speed_ms1 * 3.6
             cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
             cv2.putText(frame,str(int(a_speed_kh1))+'Km/h',(x4,y4),cv2.FONT_HERSHEY_COMPLEX,0.47,(0,255,255),2)



    cv2.imshow("RGB", frame)
    
    if cv2.waitKey(1)&0xFF==27:
        break


cap.release()
cv2.destroyAllWindows()

