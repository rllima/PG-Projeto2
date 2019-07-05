import argparse
import cv2
import numpy as np
import math
from os import path
import photo
import calibrate
import matcher
from objloader_simple import *
from camera import*

# Minimum number of matches that have to be found
# to consider the recognition valid
MIN_MATCHES = 15 

def read_cam_paramns():
        with np.load('pose/webcam_calibration_params.npz') as X:
                mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
        return mtx, dist

def prepare_env():
    if path.exists("pose/webcam_calibration_params.npz"):
        return read_cam_paramns()
    else:
        photo.take_photos()
        calibrate.calibrate()
        return read_cam_paramns()

def main():
    
    homography = None 
    # matrix of camera parameters (made up but works quite well for me) 
    camera_parameters,dist = prepare_env()
    # create ORB keypoint detector
    # create BFMatcher object based on hamming distance  
    # load the reference image
    orb,bf,model = matcher.pre_matcher()
    # Compute model keypoints and its descriptors
    kp_model, des_model = orb.detectAndCompute(model, None)
    # Load 3D model from OBJ file
    obj = OBJ('models/fox.obj', swapyz=True)  
    # init video capture
    cap = VideoCaptureAsync(0)
    cap.start()
    cv2.waitKey(1)

    while True:
        # read the current frame
        ret, frame = cap.read()
        if not ret:
            print ("Unable to capture video")
            return 
        # find and draw the keypoints of the frame
        kp_frame, des_frame = orb.detectAndCompute(frame, None)
        # match frame descriptors with model descriptors
        matches = bf.match(des_model, des_frame)
        # sort them in the order of their distance
        # the lower the distance, the better the match
        matches = sorted(matches, key=lambda x: x.distance)

        # compute Homography if enough matches are found
        if len(matches) > MIN_MATCHES:
            # differenciate between source points and destination points
            src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            # compute Homography
            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            # Draw a rectangle that marks the found model in the frame
            h, w,_ = model.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            # project corners into frame
            dst = cv2.perspectiveTransform(pts, homography)
            # connect them with lines  
            frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA) 
            axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3) 
            # if a valid homography matrix was found render cube on model plane
            if homography is not None:
                try:
                    #Pose estimation
                     # Find the rotation and translation vectors.
                    #rvecs, tvecs, inliers = cv2.solvePnPRansac(src_pts, dst_pts, camera_parameters, dist)
                    # obtain 3D projection matrix from homography matrix and camera parameters
                    #imgpts, jac = cv2.projectPoints(obj, rvecs, tvecs, mtx, dist)
                    projection = projection_matrix(camera_parameters, homography)  
                    # project cube or model
                    #frame = draw(model,dst_pts,imgpts)
                    frame = render(frame, obj, projection, model,color=False)
                   
                except:
                    pass
            # draw first 10 matches.
           
            #frame = cv2.drawMatches(model, kp_model, frame, kp_frame, matches[:10], 0, flags=2)
            # show result
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            print ("Not enough matches found - %d/%d" % (len(matches), MIN_MATCHES))

    cap.stop()
    cv2.destroyAllWindows()
    

def render(img, obj, projection, model, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h, w,_ = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img

def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)

def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))


if __name__ == '__main__':
    main()