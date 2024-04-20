import cv2

cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Central Program', frame)
    cv2.imwrite('shared_frame.jpg', frame) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
