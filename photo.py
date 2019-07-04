import cv2
from camera import*


def take_photos():

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    img_counter = 0
    cam = VideoCaptureAsync(0)
    cam.start()
    
    while True:
        frame = cam.read()[1]
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
        k = cv2.waitKey(1)

        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        # Save images only if corners detected
        if ret is True:
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            # Save images
            if img_counter < 20:
                img_name = "opencv_frame_{}.jpg".format(img_counter)
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_name))
                img_counter += 1
            cv2.drawChessboardCorners(frame, (7,6), corners2,ret)
        cv2.imshow("PG",frame)
    cam.stop()
    cv2.destroyAllWindows()