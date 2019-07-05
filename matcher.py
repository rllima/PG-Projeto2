import numpy as np 
import cv2 as cv
from camera import VideoCaptureAsync
from threading import Thread
from matplotlib import pyplot as plt


def pre_matcher():
        img = cv.imread('matcher/query.png')
        orb = cv.ORB_create()
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        return orb,bf,img



def homografy(frame,matches,img,kp1,kp2):
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
        return img2,M

def get_match(img,frame,bf,orb,kp1,des1):
        kp2,des2 = orb.detectAndCompute(frame,None)
        matches = bf.match(des1,des2)

        # Need to draw only good matches
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) > 15:
                # draw first 15 matches.
                img2 = homografy(frame,matches,img,kp1,kp2)
                frame = cv.drawMatches(img, kp1, img2, kp2,matches[:15], 0, flags=2)
        return frame




       
