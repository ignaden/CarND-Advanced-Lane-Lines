import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import copy as cp
import time
import glob

def save_params(cameraCaleb):
    """ Save the calibrated camera parameters. """
    
    print ("Saving results to `cameraCaleb.pickle`")
    with open('../cameraCaleb.pickle', 'wb') as f:
        pickle.dump(cameraCaleb, f)
    
def load_params ():
    """ Load the calibrated camera parameters """
    
    print ("Loading parameters from `cameraCaleb.pickle`")    
    cameraCaleb = None
    with open('../cameraCaleb.pickle', 'rb') as f:
        cameraCaleb = pickle.load(f)

    return cameraCaleb

def calibrate_camera(debug=True):
  """ What about now? """

  #
  # chessboard size is different in this one
  #
  nx, ny = 9, 6

  objp = np.zeros((nx * ny, 3), np.float32)
  objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

  # Arrays to store object points and image points from all the images.
  objpoints = [] # 3d points in real world space
  imgpoints = [] # 2d points in image plane.

  # Make a list of calibration images
  images = glob.glob('../camera_cal/calibration*.jpg')
  #images = [ 'camera_cal/calibration2.jpg' ]

  # Step through the list and search for chessboard corners
  foundCount, img_size = 0, None

  for idx, fname in enumerate(images):
    print ("idx = %1d" % idx)
  
    img = cv2.imread(fname)
    img_size = (img.shape[1], img.shape[0])
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray_img, (nx, ny), None)

    if ret == True:
      foundCount += 1

      # Notice that we append both points
      objpoints.append(objp)
      imgpoints.append(corners)

      # Draw and display the corners
      cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

      # So we have a chance to witness it
      #time.sleep(1)

  if foundCount > 0:
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
  else:
    raise Exception ("Cannot calibrate camera as no points converged!")

  if debug:
    print ('Found chessboard in %d out of %d images' % (foundCount, len(images)))

    # Test undistortion on an image
    img = cv2.imread('../camera_cal/calibration3.jpg')
    img_size = (img.shape[1], img.shape[0])

    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    ret, corners = cv2.findChessboardCorners(undist, (nx, ny), None)
    cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)

    hor = np.hstack((img, undist))
    cv2.imwrite ('../writeup_images/undistorted_output.png', hor)

  cameraCaleb = {
    'ret' : ret
  , 'mtx' : mtx
  , 'dist': dist
  , 'rvecs' : rvecs
  , 'tvecs' : tvecs
  }

  save_params(cameraCaleb)

if __name__ == "__main__":
  
  try:
    calibrate_camera()
  except Exception as e:
    print ("Failed to calibrate the camera: [%s]" % str(e))
