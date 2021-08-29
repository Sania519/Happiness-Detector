import cv2

eye_cascade = cv2.CascadeClassifier('C:/Users/Dell/Desktop/Computer Vision/Homework/haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('C:/Users/Dell/Desktop/Computer Vision/Homework/haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('C:/Users/Dell/Desktop/Computer Vision/Homework/haarcascade_smile.xml')

def detect(gray,frame):
    face = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in face :
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_frame = frame[y:y+h,x:x+w]
        eye = eye_cascade.detectMultiScale(roi_gray,1.1,5)
        smile = smile_cascade.detectMultiScale(roi_gray,1.3,5)
        for (x1,y1,w1,h1) in eye :
            cv2.rectangle(roi_frame,(x1,y1),(x1+w1,y1+h1),(0,255,0),2)
        for (x2,y2,w2,h2) in smile :
            cv2.rectangle(roi_frame,(x2,y2),(x2+w2,y2+h2),(0,0,255),2)
    return(frame)

video_capture = cv2.VideoCapture(0)
while(1):
    _,frame = video_capture.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas = detect(gray,frame)
    cv2.imshow('Video',canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()