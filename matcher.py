import numpy as np 
import cv2 as cv
from camera import VideoCaptureAsync
from threading import Thread
from matplotlib import pyplot as plt

img = cv.imread('matcher/query.png')
cam = VideoCaptureAsync(0)
cam.start()
cv.namedWindow("PG")

orb = cv.ORB_create()

#Query image features
kp1,des1 = orb.detectAndCompute(img,None)

FLAN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLAN_INDEX_KDTREE, trees = 10)
search_paramns = dict(checks=100)
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)


def homografy(frame):
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        # compute Homography
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        h, w,_ = img.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # project corners into frame
        dst = cv.perspectiveTransform(pts, M)  
        # connect them with lines
        img2 = cv.polylines(frame, [np.int32(dst)], True, 255, 3, cv.LINE_AA) 
        return img2


while True:
        frame = cam.read()[1]
        kp2,des2 = orb.detectAndCompute(frame,None)
        
        matches = bf.match(des1,des2)

        # Need to draw only good matches
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) > 15:
                # draw first 15 matches.
                img2 = homografy(frame)
                frame = cv.drawMatches(img, kp1, img2, kp2,matches[:15], 0, flags=2)
        cv.imshow("PG",frame)
        
        k = cv.waitKey(1)

        if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
cam.stop()
cv.destroyAllWindows()



       
