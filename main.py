#!/user/bin/env python
import numpy as np
import cv2
import pickle
import RPi.GPIO as GPIO
import time
from time import sleep
import argparse
import sys
import os.path
import vlc
from glob import glob
from pygame import mixer
from threading import Timer

def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
def drawPred(classId, conf, left, top, right, bottom):
    global frame_count
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    label = '%.2f' % conf
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
        
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    label_name,label_conf = label.split(':')    #spliting into class & confidance. will compare it with person.
    if label_name == 'Helmet':
        cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
        frame_count+=1

    if(frame_count> 0):
        return frame_count
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    global frame_count_out
    frame_count_out=0
    classIds = []
    confidences = []
    boxes = []
    classIds = []               #have to find which class have hieghest confidence........=====>>><<<<=======
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    count_person=0 # for counting the classes in this loop.

    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        
        frame_count_out = drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

        my_class='Helmet'                   #======================================== mycode .....
        unknown_class = classes[classId]

        if my_class == unknown_class:
            count_person += 1
    
        return True if frame_count_out > 0 else False

    if count_person >= 1:
        path = 'test_out/'
        frame_name=os.path.basename(fn)
        cv2.imwrite(str(path)+frame_name, frame)
        #cv2.imshow('img',frame)
        cv2.waitKey(1)
        def off_motor():
    mixer.music.load('/home/pi/Music/ShutDown.mp3')
    mixer.music.play()
    time.sleep(2)
    global motorOff
    motorOff = True
    GPIO.output(27, False)
    time.sleep(3)
    GPIO.output(27, True)
    print('--------------------')


def play_music(path):
    mixer.music.load(path)
    mixer.music.play()

def setGPIO():
    print("[INFO] Initializing GPIO...")
    GPIO.setwarnings(False)  #RELAY1 PIN 17 NORMALLY OPEN
    GPIO.setmode(GPIO.BCM)   #RELAY2 PIN 27 NORMALLY CLOSED
    GPIO.setup(17, GPIO.OUT) #True inital state
    GPIO.output(17, True)    #False to turn on motor
    GPIO.setup(27, GPIO.OUT) #True inital state
    GPIO.output(27, True)    #False to turn off motor             
frame_count = 0             # used in mainloop  where we're extracting images., and then to drawPred( called by post process)
frame_count_out=0           # used in post process loop, to get the no of specified class value.
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image
classesFile = "/home/pi/obj.names";              
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
modelConfiguration = "/home/pi/yolov3-obj.cfg";    #algorithm for detecting object yolo = you only look once
modelWeights = "/home/pi/yolov3-obj_2400.weights"; 
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
conf_level = 45
fr_level = 50
r_count = 0
h_count = 0
count_helmet = 0
motorOff = False
face_recognize = False
helmet_recognize = False
helmet_detected = False
isStart = False

tmp_name = ""

timer = ""
#timer = Timer(60, off_motor) #After 60secs = 1min off ang motor (find function off_motor)
face_cascade = cv2.CascadeClassifier('/home/pi/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("/home/pi/trainner.yml")
labels = {"person_name": 1}
with open("/home/pi/labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
setGPIO()
mixer.init()

while True:
    ret, frame = cap.read()
    cv2.imwrite('/home/pi/helmet.jpg', frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        font = cv2.FONT_HERSHEY_SIMPLEX
        id_, conf = recognizer.predict(roi_gray)
        end_cord_x = x + w
        end_cord_y = y + h

        if conf >= conf_level:
            name = labels[id_] + " {}%".format(round(conf, 2))
            #First, check if face is recognized or not.
            if not face_recognize: #if not face recognize
                if tmp_name is labels[id_]: #check if tmp_name equal to the image captured in the camera
                    if not tmp_name == 'unknown': #if not 'unknown'
                        face_recognize = True if r_count is fr_level else False #then if r_count is equal to fr_level then face_recognize = True
                        cv2.putText(frame, "{}%".format(round((r_count / fr_level) * 100)), (w+x,y+h), font, 1, (0,0,255), 2, cv2.LINE_AA)
                        r_count = r_count + 1
                        cv2.putText(frame, name, (x,y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), (255, 255, 255), 3)
                else: #if tmp_name is not equal to the image captured in the camera
                    tmp_name = labels[id_] #initialize tmp_name to labels[id_] value
                    if tmp_name == "unknown": #if tmp_name is 'unknown' then r_count = 0
                        cv2.putText(frame, 'unknown', (x,y), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), (0, 0, 255), 3)
                        #break
                    r_count = 0
            else:
                print('FACE DETECTED')                                 #B   G   R
                cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), (0, 255, 0), 2)
                cv2.imwrite('/home/pi/helmet.jpg',frame)
                play_music('/home/pi/Music/FaceRecognized.mp3')
                if helmet_recognize: #if helmet is recognize
                    print('HELMET DETECTED')
                    play_music('/home/pi/Music/HelmetDetected.mp3')
                    if face_recognize and helmet_recognize: #if helmet and face is recognize
                        if not isStart: #if start is false then start motor
                            #timer.cancel() #cancel the 5mins timer
                            GPIO.output(17, False)
                            time.sleep(0.5)
                            GPIO.output(17, True)
                            isStart = True
                        else:
                            GPIO.output(17, True)
                            
                    if not tmp_name is labels[id_]:
                        print('HELMET REMOVED')
                        r_count = 0
                        #if helmet is removed, timer will start automatically
                        time.sleep(1)
                        timer = Timer(60, off_motor)
                        timer.start()
                        if motorOff:
                            timer.cancel()
                            mixer.music.play()
                            print('MOTOR OFF')
                            motorOff = False
                        face_recognize = False
                        helmet_recognize = False
                        play_music('/home/pi/Music/PleaseWearHelmet.mp3')
                        time.sleep(2)
                        play_music('/home/pi/Music/OneMinute.mp3')
                        time.sleep(4)
                    #continue
                else:
                    pic = cv2.imread('/home/pi/helmet.jpg') # read image
                    frame_count = 0
                    #check for blobs 
                    blob = cv2.dnn.blobFromImage(pic, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
                    net.setInput(blob)
                    outs = net.forward(getOutputsNames(net))
                    #if helmet is recognize then it is true
                    helmet_recognize = postprocess(pic, outs) 
                    t, _ = net.getPerfProfile()
        else:
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), (0, 0, 255), 3) 

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release() 
cv2.destroyAllWindows()
