import cv2

cam = cv2.VideoCapture(0)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
cv2.namedWindow("Calibration")
fps = cam.get(cv2.CAP_PROP_FPS)

img_counter = 0

while True:
    _, frame = cam.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    if ret is True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # Draw and display the corners
        if img_counter < 10:
            img_name = "opencv_frame_{}.jpg".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
        img=cv2.drawChessboardCorners(frame, (7,6), corners2,ret)
    cv2.imshow('Calibration',frame)
cam.release()

cv2.destroyAllWindows()