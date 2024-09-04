import cv2

cap = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')

if cap.isOpened() == False:
    print("Error in opening video stream or file")
    
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = facedetect.detectMultiScale(gray,1.1,5)
    
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
        # Display the resulting frame
        cv2.imshow('Frame',frame)   
        
        # Press esc to exit
        if (cv2.waitKey(20) & 0xFF) == 27:
            break
    else:
        break
    
cap.release()
cv2.destroyAllWindows()