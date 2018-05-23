
import cv2
import numpy as np
import pickle


class FramePreProcessor:
  """ """

  def __init__ (self, cameraCaleb, params):
    """ """
    
    self._params = params
    self._cameraCaleb = cameraCaleb
  
  @staticmethod
  def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    """ """

    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
      abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    elif orient == 'y':
      abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
    else:
      raise Exception ("Invalid orientation: " + orient)
    
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

  def translate_color (self, img):
    """ Perform color transform """
        
    img = np.copy(img)
        
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    # Sobel x
    min_thresh, max_thresh = self._params['color_trans']['sx_thresh']
    sxbinary = self.abs_sobel_thresh(l_channel, orient='x', thresh_min = min_thresh, thresh_max = max_thresh)
    sybinary = self.abs_sobel_thresh(l_channel, orient='y', thresh_min = min_thresh, thresh_max = max_thresh)
        
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= self._params['color_trans']['s_thresh'][0]) & 
              (s_channel <= self._params['color_trans']['s_thresh'][1])] = 1
                        
    combined = np.zeros_like(s_binary)
    combined[(sxbinary == 1) | (s_binary == 1)] = 1

    return combined
  
  def undistort_img(self, img):
    """ """
    return cv2.undistort(img, self._cameraCaleb['mtx'], self._cameraCaleb['dist'], None, self._cameraCaleb['mtx'])

  # Define a function that takes an image, number of x and y points, 
  # camera matrix and distortion coefficients
  @staticmethod
  def transform(img):
    """ """
    # Get the dimensions
    width, height = img.shape[1], img.shape[0]
    img_size = (width, height)

    # define the trapezoid

    src = np.float32([[605, 445], [685, 445],
                      [1063, 676], [260, 676]])
    
    dst = np.float32([[width * 0.35, 0], [width * 0.65, 0], 
                      [width * 0.65, height], [width * 0.35, height]])

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Warp the image using OpenCV warpPerspective()
    return cv2.warpPerspective(img, M, (int(width), int(height)), flags=cv2.INTER_LINEAR), M, Minv
    
