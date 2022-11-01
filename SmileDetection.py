import cv2

capture = cv2.VideoCapture(0)

while(True) :

    _ , frame = capture.read()

    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    smile = smile_cascade.detectMultiScale(
        frame,
        scaleFactor = 2.5,
        minNeighbors = 20
    )

    for x, y, w, h in smile:
        frame = cv2.rectangle(
            frame,
            (x,y),
            (x+w, y+h),
            (0, 255, 0),
            3
        )

    cv2.imshow('Smile Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break

capture.realease()
cv2.destroyAllWindows()