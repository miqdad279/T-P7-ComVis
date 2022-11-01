import cv2

capture = cv2.VideoCapture(0)

while(True):
    _ , frame = capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = face_cascade.detectMultiScale(gray_img, 1.3, 5)
    for (x_face, w_face, y_face, h_face) in face:
        cv2.rectangle(frame, (x_face, y_face), (x_face+w_face, y_face+h_face), (255, 130, 0), 2)

    ri_grayscale = gray_img[y_face:y_face+h_face, x_face:x_face+w_face]
    ri_color = frame[y_face:y_face+h_face, x_face:x_face+w_face]

    eye = eye_cascade.detectMultiScale(ri_grayscale, 1.2, 18)
    for(x_eye, y_eye, w_eye, h_eye) in eye:
        cv2.rectangle(ri_color, (x_eye, y_eye), (x_eye+w_eye, y_eye+h_eye), (0, 100, 60), 2)

    smile = smile_cascade.detectMultiScale(ri_grayscale, 2.5, 20)
    for(x_smile, y_smile, w_smile, h_smile) in smile:
        cv2.rectangle(ri_color, (x_smile, y_smile), (x_smile+w_smile, y_smile+h_smile), (255,0,130), 2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow('Face Feature', frame)

capture.release()
cv2.destroyAllWindows()
