import os
import cv2
import numpy as np
import pickle

from PIL import Image

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