
import cv2
import numpy as np
import pickle
from os import path
from camera import load_params
import matplotlib.pyplot as plt 

class LineFit:
    """ Performs fit operations """
    ym_per_pix = 30 / 720   # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    def __init__ (self, fit, inds, allx, ally, debug=False):
        """ """

        self.line_fit = fit
        self.inds = inds
        self.allx = allx
        self.ally = ally
        
        # Actually, y_bottom would be the maximum value (720?)
        y_bottom = 720

        fit_cr = np.polyfit(self.ally * self.ym_per_pix, self.allx * self.xm_per_pix, 2)
        self.radius_of_curvature = ((1.0 + (2.0 * fit_cr[0] * y_bottom * self.ym_per_pix + fit_cr[1]) ** 2.0) ** 1.5) / np.absolute(2.0 * fit_cr[0])
        self.slope = 2 * fit_cr[0] * y_bottom * self.ym_per_pix + fit_cr[1]

        self.offset = (fit[0] * (y_bottom ** 2) + fit[1] * y_bottom + fit[2]) / 2 * self.xm_per_pix

        if debug:
          # Let's print out all of the parameters here
          print ("")
          print ("ym_per_pix = %.2f; xm_per_pix = %.2f" % (self.ym_per_pix, self.xm_per_pix))
          print ("A = %.4f" % fit_cr[0])
          print ("B = %.4f" % fit_cr[1])
          print ("C = %.3f" % fit_cr[2])
          print ("")

    @staticmethod
    def new_line_fit (binary_warped, params, isLeft=True):
        """ fitPolyLanes - fit a polynomial curve """
        # Just to make sure!
        binary_warped = binary_warped.astype(np.uint8) 

        histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis = 0)
              
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines        
        midpoint = np.int(histogram.shape[0] / 2)
        minpoint = np.int(histogram.shape[0] * params['fit']['minpoint_fract'])
        maxpoint = np.int(histogram.shape[0] * params['fit']['maxpoint_fract'])
        
        leftx_base = np.argmax(histogram[minpoint:midpoint]) + minpoint
        rightx_base = np.argmax(histogram[midpoint:maxpoint]) + midpoint

        # Choose the number of sliding windows
        nwindows = params['fit']['nwindows']
        
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)
        
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
        
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        # Set the width of the windows +/- margin
        margin = params['fit']['margin']
        
        # Set minimum number of pixels found to recenter window
        minpix = params['fit']['minpix']
        
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds, right_lane_inds = [], []

        # Step through the windows one by one

        rhist, lhist = [], []

        for window in range(nwindows):
            
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            rhist.append(good_right_inds)
            if len(rhist) > 3:
              rhist = rhist[1:]
            
            lhist.append(good_left_inds)
            if len(lhist) > 3:
              lhist = lhist[1:]

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[np.concatenate(lhist)]))
                #leftx_current = np.int(np.mean (nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                #rightx_current = np.int(np.mean (nonzerox[good_right_inds]))
                rightx_current = np.int(np.mean(nonzerox[np.concatenate(rhist)]))

        if isLeft:
            left_lane_inds = np.concatenate(left_lane_inds)

            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]

            left_fit = np.polyfit(lefty, leftx, 2)

            return LineFit(left_fit, left_lane_inds, leftx, lefty)
        
        else:
            right_lane_inds = np.concatenate(right_lane_inds)

            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]

            right_fit = np.polyfit(righty, rightx, 2)

            return LineFit(right_fit, right_lane_inds, rightx, righty)
    
    @staticmethod
    def update_fit (old_fit, binary_warped, params, isLeft=True):
        """ """
        
        binary_warped = binary_warped.astype(np.uint8)

        # Assume you now have a new warped binary image 
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        if isLeft:
            left_lane_inds = ((nonzerox > (old_fit[0]*(nonzeroy**2) + old_fit[1]*nonzeroy + 
                              old_fit[2] - params['fit']['margin'])) & (nonzerox < (old_fit[0]*(nonzeroy**2) + 
                              old_fit[1]*nonzeroy + old_fit[2] + params['fit']['margin']))) 
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds] 

            left_fit = np.polyfit(lefty, leftx, 2)

            return LineFit(left_fit, left_lane_inds, leftx, lefty)
        
        else:
            right_lane_inds = ((nonzerox > (old_fit[0]*(nonzeroy**2) + old_fit[1]*nonzeroy +
                              old_fit[2] - params['fit']['margin'])) & (nonzerox < (old_fit[0]*(nonzeroy**2) +
                              old_fit[1]*nonzeroy + old_fit[2] + params['fit']['margin'])))  

            # Again, extract left and right line pixel positions
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
        
            right_fit = np.polyfit(righty, rightx, 2)

            return LineFit(right_fit, right_lane_inds, rightx, righty)
    
    @staticmethod
    def diff_small (new, old, good_range):
        return (np.absolute(new - old) / np.absolute (new + old)) < good_range

    @staticmethod
    def is_good_fit (leftLine, rightLine):
        """ It's a good fit if the distance between the lines at the top is similar to the bottom. """
        
        max_dist = 0.10

        lf = leftLine.line_fit
        rf = rightLine.line_fit

        y_top, y_mid, y_bottom = 0, 719, 350

        left_fitx_top = lf[0]*y_top**2 + lf[1]*y_top + lf[2]
        right_fitx_top = rf[0]*y_top**2 + rf[1]*y_top + rf[2]

        left_fitx_mid = lf[0]*y_mid**2 + lf[1]*y_mid + lf[2]
        right_fitx_mid = rf[0]*y_mid**2 + rf[1]*y_mid + rf[2]

        left_fitx_bottom = lf[0]*y_bottom**2 + lf[1]*y_bottom + lf[2]
        right_fitx_bottom = rf[0]*y_bottom**2 + rf[1]*y_bottom + rf[2]

        dist_top = right_fitx_top - left_fitx_top
        dist_mid = right_fitx_mid - left_fitx_mid
        dist_bot = right_fitx_bottom - left_fitx_bottom

        return (np.absolute(dist_top - dist_bot) / dist_bot) < max_dist and (np.absolute(dist_mid - dist_bot) / dist_bot) < max_dist
    