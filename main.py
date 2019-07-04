import os.path
import photo
import calibrate
from os import path
import numpy as np




def prepare_env():
    if path.exists("pose/webcam_calibration_params.npz"):
        with np.load('pose/webcam_calibration_params.npz') as X:
            return X
    else:
        photo.take_photos()
        calibrate.calibrate()
    

prepare_env()


        
