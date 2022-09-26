import numpy as np
import cv2
import pickle5 as pickle
import RPi.GPIO as GPIO
import argparse
import os.path

from glob import glob
from pygame import mixer
from threading import Timer
from time import sleep

from PIL import Image

class VehicleAccess:
    def __init__(self, video_channel=None, classes_file=None, model_w=None, model_cfg=None, face_cascade=None, recognizer=None, helmet=None, pickles=None, music_path=None):
        self.video_channel = video_channel
        self.classes = None
        self.classes_file = classes_file
        self.model_w = model_w
        self.model_cfg = model_cfg
        self.face_cascade = cv2.CascadeClassifier(face_cascade)
        self.recognizer = recognizer
        self.helmet = helmet
        self.pickles = pickles
        self.music_path = music_path

        self.net = cv2.dnn.readNetFromDarknet(self.model_cfg, self.model_w)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.conf_threshold = 0.5
        self.nms_threshold = 0.4 
        self.inp_width = 416
        self.inp_height = 416
        self.conf_level = 45
        self.fr_level = 50
        self.classes = None
        self.frame_count = 0
        self.frame_count_out = 0
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        self.face_recognize = False
        self.helmet_recognize = False
        self.helmet_detected = False
        self.started = False
        self.motor_off = False
        
        GPIO.setwarnings(False)  
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(17, GPIO.OUT) 
        GPIO.output(17, True)   
        GPIO.setup(27, GPIO.OUT)
        GPIO.output(27, True)   

        self.labels = {"person_name": 1}
        with open(self.pickles, 'rb') as f:
            og_labels = pickle.load(f)
            self.labels = {v:k for k,v in og_labels.items()}

        mixer.init()

    def get_output_names(self, net):
        return [net.getLayerNames()[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    def draw_prediction(self, classId, conf, left, top, right, bottom, frame):
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

        label = '%.2f' % conf

        if self.classes:
            assert(classId < len(self.classes))
            label = '%s:%s' % (self.classes[classId], label)

        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        label_name, _ = label.split(':')
        if label_name == 'Helmet':
            cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
            self.frame_count+=1
        
        if self.frame_count > 0:
            return self.frame_count

    def post_process(self, frame, outs):
        h_frame = frame.shape[0]
        w_frame = frame.shape[1]

        self.frame_count_out=0

        classIds = []
        confidences = []
        boxes = []
        classIds = []   
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.conf_threshold:
                    center_x = int(detection[0] * w_frame)
                    center_y = int(detection[1] * h_frame)
                    width = int(detection[2] * w_frame)
                    height = int(detection[3] * h_frame)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

        count_person=0 

        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            
            frame_count_out = self.draw_prediction(classIds[i], confidences[i], left, top, left + width, top + height, frame)

            my_class='Helmet'

            unknown_class = self.classes[classId]

            if my_class == unknown_class:
                count_person += 1
        
            return True if frame_count_out > 0 else False

        if count_person >= 1:
            path = 'test_out/'
            frame_name=os.path.basename('fn')
            cv2.imwrite(str(path)+frame_name, frame)
            cv2.waitKey(1)

    def play_music(self, path):
        self.mixer.music.load(path)
        self.mixer.music.play()

    def off_motor(self):
        self.mixer.musicload(self.shutdown)
        self.mixer.music.play()

        sleep(2)
        self.motor_off = True
        GPIO.output(27, False)
        sleep(3)
        GPIO.output(27, True)

    def main(self):
        cap = cv2.VideoCapture(self.video_channel)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))

        tmp_name = ""
        r_count = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            cv2.imwrite(self.helmet, frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
            
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                id_, conf = self.recognizer.predict(roi_gray)
                end_cord_x = x + w
                end_cord_y = y + h

                if conf >= self.conf_level:
                    name = self.labels[id_] + f" {round(conf, 2)}%"

                    if not self.face_recognize:
                        if tmp_name is self.labels[id_]:
                            if not tmp_name == 'unknown':
                                self.face_recognize = True if r_count is self.fr_level else False
                                cv2.putText(frame, f"{round((r_count / self.fr_level) * 100)}%", (w+x,y+h), self.font, 1, (0,0,255), 2, cv2.LINE_AA)
                                r_count = r_count + 1
                                cv2.putText(frame, name, (x,y), self.font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                                cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), (255, 255, 255), 3)
                        else:
                            tmp_name = self.labels[id_]
                            if tmp_name == "unknown":
                                cv2.putText(frame, 'unknown', (x,y), self.font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                                cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), (0, 0, 255), 3)
                            r_count = 0
                    else:
                        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), (0, 255, 0), 2)
                        cv2.imwrite(self.helmet, frame)
                        self.play_music(os.path.join(self.music, 'FaceRecognized.mp3'))
                        if helmet_recognize:                            
                            self.play_music(os.path.join(self.music, 'HelmetDetected.mp3'))
                            if self.face_recognize and self.helmet_recognize:
                                if not self.started:
                                    GPIO.output(17, False)
                                    sleep(0.5)
                                    GPIO.output(17, True)
                                    self.started = True
                                else:
                                    GPIO.output(17, True)
                                    
                            if not tmp_name is self.labels[id_]:
                                r_count = 0
                                sleep(1)
                                timer = Timer(60, self.off_motor)
                                timer.start()
                                if self.motor_off:
                                    timer.cancel()
                                    self.mixer.music.play()
                                    print('MOTOR OFF')
                                    self.motor_off = False
                                    
                                self.face_recognize = False
                                self.helmet_recognize = False

                                self.play_music(os.path.join(self.music, 'PleaseWearHelmet.mp3'))
                                sleep(2)
                                self.play_music(os.path.join(self.music, 'OneMinute.mp3'))
                                sleep(4)
                        else:
                            pic = cv2.imread(self.helmet)
                            self.frame_count = 0
                            
                            blob = cv2.dnn.blobFromImage(pic, 1/255, (self.inp_widthdth, self.inp_height), [0,0,0], 1, crop=False)
                            self.net.setInput(blob)
                            outs = self.net.forward(self.getOutputsNames(self.net))
                            
                            helmet_recognize = self.post_process(pic, outs) 
                            t, _ = self.net.getPerfProfile()
                else:
                    cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), (0, 0, 255), 3)

            cv2.imshow('Face Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

class TrainDriver():
    def __init__(self, face_cascade=None):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.image_dir = os.path.join(self.BASE_DIR, 'images')
        self.face_cascade = cv2.CascadeClassifier(face_cascade)

    def train(self):
        current_id = 0
        label_ids = {}
        y_labels = []
        x_train = []

        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if file.endswith("jpg"):
                    path = os.path.join(root, file)
                    label = os.path.basename(root).replace(" ", "-").lower()
                    
                    if not label in label_ids:
                        label_ids[label] = current_id
                        current_id += 1 
                    id_ = label_ids[label]
                    
                    pil_image = Image.open(path).convert("L")
                    image_array = np.array(pil_image, "uint8")
                    faces = self.face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

                    for (x, y, w, h) in faces:
                        roi = image_array[y:y+h, x:x+w]
                        x_train.append(roi)
                        y_labels.append(id_)
                
        with open("labels.pickle", 'wb') as f:
            pickle.dump(label_ids, f)

        self.recognizer.train(x_train,np.array(y_labels))
        self.recognizer.save("trainner.yml")

class RegisterDriver():
    def __init__(self, video_channel, face_cascade):
        self.video_channel = video_channel
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.image_dir = os.path.join(self.BASE_DIR, 'images')
        self.face_cascade = cv2.CascadeClassifier(face_cascade)

        self.count = 0
        self.dataset = 250

    def assure_path_exists(self, path):
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)

    def register(self):
        face_id=input('Name: ')
        self.assure_path_exists(self.image_dir + face_id)

        cap = cv2.VideoCapture(self.video_channel)

        while(True):
            ret, frame = cap.read()

            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
                self.count += 1
                cv2.imwrite(self.image_dir + str(face_id) + "/" + str(self.count) + "" + ".jpg", gray[y:y+h,x:x+w])
                percentage = round((self.count / self.dataset) * 100)
                cv2.putText(frame, str(percentage)+"%", (x,y), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, cv2.LINE_AA)
            
            cv2.imshow('frame', frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            
            elif self.count >= self.dataset:
                print("[INFO] Successful!")
                
        cap.release()
        cv2.destroyAllWindows()