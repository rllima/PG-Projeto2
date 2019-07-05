import os.path
import photo
import calibrate
from os import path
import numpy as np

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
    

mtx,dist = prepare_env()




        
