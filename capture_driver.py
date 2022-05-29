import cv2
import os
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
face_id=input('Name: ')
vid_cam = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('/home/pi/data/haarcascade_frontalface_default.xml')
count = 0
dataset = 250
p_path = "/home/pi/images/"
assure_path_exists(p_path + face_id)
try:
    os.mkdir(p_path + face_id)
except:
    pass
while(True):
    _, image_frame = vid_cam.read()
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1
        cv2.imwrite(p_path + str(face_id) + "/" + str(count) + "" + ".jpg", gray[y:y+h,x:x+w])
        percentage = round((count / dataset) * 100)
        cv2.putText(image_frame, str(percentage)+"%", (x,y), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imshow('frame', image_frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    elif count>=dataset:
        print("Successfully Captured")
        break
vid_cam.release()
cv2.destroyAllWindows()
