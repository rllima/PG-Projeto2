import numpy as np 
import cv2 as cv
from matplotlib import pyplot as plt

img1 = cv.imread('matcher/query.png',0)
cam = cv.VideoCapture(0)

cv.namedWindow("Calibration")

freak = cv.BRISK_create()
kp1,des1 = freak.detectAndCompute(img1,None)
while True:
        _, frame = cam.read()
        kp2,des2 = freak.detectAndCompute(frame,None)
        FLAN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLAN_INDEX_KDTREE, trees = 10)
        search_paramns = dict(checks=100)

        flann = cv.FlannBasedMatcher(index_params,search_paramns)
        matches = flann.knnMatch(np.asarray(des1,np.float32),np.asarray(des2,np.float32),k=2)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]

        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
                if m.distance < 0.7*n.distance:
                        matchesMask[i]=[1,0]

        draw_params = dict(matchColor = (0,255,0),
                        singlePointColor = (255,0,0),
                        matchesMask = matchesMask,
                        flags = 0)

        frame = cv.drawMatchesKnn(img1,kp1,frame,kp2,matches,None,**draw_params)

        cv.imshow("Calibration",frame),plt.show()
        k = cv.waitKey(1)

        if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
cam.release()

cv.destroyAllWindows()


