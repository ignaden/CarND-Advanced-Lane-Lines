
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

    def __init__ (self, fit, inds, allx, ally):
        """ """

        self.line_fit = fit
        self.inds = inds
        self.allx = allx
        self.ally = ally

        #y_bottom = 719
        y_bottom = 0

        fit_cr = np.polyfit(self.ally * self.ym_per_pix, self.allx * self.xm_per_pix, 2)
        self.radius_of_curvature = ((1 + (2 * fit_cr[0] * y_bottom * self.ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])
        self.slope = 2 * fit_cr[0] * y_bottom * self.ym_per_pix + fit_cr[1]

        self.offset = (fit[0] * (y_bottom ** 2) + fit[1] * y_bottom + fit[2]) / 2 * self.xm_per_pix

    @staticmethod
    def new_line_fit (binary_warped, params, isLeft=True):
        """ fitPolyLanes - fit a polynomial curve """
        # Just to make sure!
        binary_warped = binary_warped.astype(np.uint8) 

        histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
              
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

        print ('curv: %.2f, %.2f' % (leftLine.radius_of_curvature, rightLine.radius_of_curvature))
        print ('slope: %.2f, %.2f' % (leftLine.slope, rightLine.slope))

        print ('left', leftLine.line_fit)
        print ('right', rightLine.line_fit)

        #return True
        if not LineFit.diff_small(leftLine.radius_of_curvature, rightLine.radius_of_curvature, 0.20):
          print ('New curv fit: %.2f; Old fit: %.2f' % (new_fit.radius_of_curvature, other_line.radius_of_curvature))
          return False

        #if not LineFit.diff_small(leftLine.slope, rightLine.slope, 0.33):
          #print ('New slope fit: %.2f; Old fit: %.2f' % (new_fit.radius_of_curvature, other_line.radius_of_curvature))
        #  return False

        return True
    
# Define a class to receive the characteristics of each line detection
class Line():
    LEFT, RIGHT = 1, 2
    MAX_HIST_LENGTH = 3

    """ class Line:  """
    def __init__(self, side=None):
        # Save the side - this will be used later
        self._side = side

        # was the line detected in the last iteration?
        self.detected = False
        
        # x values of the last n fits of the line
        self.recent_xfitted = []
        self.recent_params  = { 'A': [], 'B': [], 'C': []}
        
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        
        #radius of curvature of the line in some units
        self.radius_of_curvature = None

        # slope
        self.slope = None
        
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        
        #x values for detected line pixels
        self.allx = None
        
        #y values for detected line pixels
        self.ally = None

    def use_last_good_fit (self):
        return

    def good_history(self):
        return len(self.recent_xfitted) > 0

    def fit_exists(self):
        """ """

        return self.detected
    
    def apply_new_fit(self, new_fit):
        """ Update the new fit details """
        self.detected = True
        self.current_fit = new_fit.line_fit
        self.allx = new_fit.allx
        self.ally = new_fit.ally

        # All the required stats
        self.radius_of_curvature = new_fit.radius_of_curvature
        self.slope = new_fit.slope
        self.offset = new_fit.offset

        #
        # Now we need to add these points to our history!
        #
        self.best_fit = new_fit.line_fit
        
        # We should now be able to calculate the circumference!

        self.recent_xfitted.append(new_fit.allx)
        self.recent_params['A'].append(new_fit.line_fit[0])
        self.recent_params['B'].append(new_fit.line_fit[1])
        self.recent_params['C'].append(new_fit.line_fit[2])

        if len(self.recent_xfitted) > self.MAX_HIST_LENGTH:
          self.recent_xfitted = self.recent_xfitted[1:]
          self.recent_params['A'] = self.recent_params['A'][1:]
          self.recent_params['B'] = self.recent_params['B'][1:]
          self.recent_params['C'] = self.recent_params['C'][1:]
    
    def get_smooth_fit (self):
      
      return self.best_fit

      if not self.recent_xfitted:
        return self.current_fit
      
      return [np.mean(self.recent_params['A']), np.mean(self.recent_params['B']), np.mean(self.recent_params['C'])]

#      allx, ally = [], []
#      for h in self.recent_xfitted:
#        allx.extend(h[0])
#
#        ally.extend(h[1])
#      return np.polyfit(ally, allx, 2)

    def error_mode (self, on=True):
      pass

class Lines:
    """ Contains information about both of the lines """

    def __init__ (self, runId, cameraCaleb, transParams, debug=False, output_dir="."):
        """ """
        self._currframe = 0

        self._debug = debug             # Are we in debug mode?
        self._runId = runId             # Run ID of all the current
        self._params = transParams      # Color transform and fitting parameters
        self._output_dir = output_dir   # Output directory for debug images
        self._cameraCaleb = cameraCaleb # Calibration parameters for the camera
        self._leftLine, self._rightLine = Line(Line.LEFT), Line(Line.RIGHT)
        self._last_s_binary = None

    def fit_lines (self, warped, alwaysNew=True):
        """ Attempt to use existing fits, restart if not. """
        #self._debug = True
        # Update the left line
        if self._leftLine.fit_exists() and not alwaysNew:
          if self._debug:
            print ('Old left line fit exists. using it')
          leftCandidate = LineFit.update_fit(self._leftLine.best_fit, warped, self._params, isLeft=True)
        else:
          # Try to fit a new one
          if self._debug:
            print ('Fitting a left new line')
          leftCandidate = LineFit.new_line_fit(warped, self._params, isLeft=True)
        
        if self._rightLine.fit_exists() and not alwaysNew:
          if self._debug:
            print ('Old right line fit exists. using it')
          rightCandidate = LineFit.update_fit(self._rightLine.best_fit, warped, self._params, isLeft=False)
        else:
          # Try to fit a new one
          if self._debug:
            print ('Fitting a right new line')

          rightCandidate = LineFit.new_line_fit(warped, self._params, isLeft=False)

        if LineFit.is_good_fit(leftCandidate, rightCandidate):
          if self._debug:
            print ('Good fit!!!')
          # use these to update the current ones
          self._leftLine.apply_new_fit(leftCandidate)
          self._rightLine.apply_new_fit(rightCandidate)
        else:
          if self._debug:
            print('bad Fit!')
          self._leftLine.use_last_good_fit()
          self._rightLine.use_last_good_fit()

        if not (self._leftLine.fit_exists() and self._rightLine.fit_exists()):
          # Get out here
          if self._debug:
            print ('Hello')
          return None

        #self._debug = False
        
        # Create an image to draw on and an image to show the selection window
        warped      = warped.astype(np.uint8)
        out_img     = np.dstack((warped, warped, warped)) * 255
        window_img  = np.zeros_like(out_img)

        # Color in left and right line pixels
        out_img[self._leftLine.ally,  self._leftLine.allx]  = [255, 0, 0]
        out_img[self._rightLine.ally, self._rightLine.allx] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1   = np.array([np.transpose(np.vstack([self._leftLine.allx - self._params['fit']['margin'], self._leftLine.ally]))])
        left_line_window2   = np.array([np.flipud(np.transpose(np.vstack([self._leftLine.allx + self._params['fit']['margin'], self._leftLine.ally])))])
        left_line_pts       = np.hstack((left_line_window1, left_line_window2))
        
        # Draw out the right side!
        right_line_window1  = np.array([np.transpose(np.vstack([self._rightLine.allx - self._params['fit']['margin'], self._rightLine.ally]))])
        right_line_window2  = np.array([np.flipud(np.transpose(np.vstack([self._rightLine.allx + self._params['fit']['margin'], self._rightLine.ally])))])
        right_line_pts      = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]),  (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))

        return cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # def fit_lines (self, warped, alwaysNew=True):
    #     """ Attempt to use existing fits, restart if not. """
    #     #self._debug = True
    #     # Update the left line
    #     if self._leftLine.fit_exists() and not alwaysNew:
    #       if self._debug:
    #         print ('Old left line fit exists. using it')
    #       candidateFit = LineFit.update_fit(self._leftLine.best_fit, warped, self._params, isLeft=True)

    #       # Candidate fit is good?
    #       if self._leftLine.is_good_fit(candidateFit, self._rightLine):
    #         if self._debug:
    #           print ('Obtained a good new fit!')
    #         self._leftLine.apply_new_fit(candidateFit)
    #       else:
    #         if self._debug:
    #           print ('Could not obtain a good new fit!')
    #         # We tried to update the existing fit and it wasn't good
    #         if self._leftLine.good_history():
    #           if self._debug: print ('Good history exists!')
    #           # Take the last good one and use it again.
    #           self._leftLine.use_last_good_fit()
    #         else:
    #           if self._debug:
    #             print ('No good history exists!')
    #           candidateFit = LineFit.new_line_fit(warped, self._params, isLeft=True)
    #           # Well, we don't have good history, hence we need to continue
    #     else:
    #       # Try to fit a new one
    #       candidateFit = LineFit.new_line_fit(warped, self._params, isLeft=True)
    #       if self._leftLine.is_good_fit(candidateFit, self._rightLine):
    #         self._leftLine.apply_new_fit(candidateFit)
    #       else:
    #         # we have a good fit now, continue
    #         self._leftLine.error_mode(on=True)
        
    #     # Same thing for right line
    #     if self._rightLine.fit_exists() and not alwaysNew:
    #       candidateFit = LineFit.update_fit(self._rightLine.best_fit, warped, self._params, isLeft=False)
          
    #       if self._rightLine.is_good_fit(candidateFit, self._leftLine):
    #         self._rightLine.apply_new_fit(candidateFit)
    #       else:
    #         if self._rightLine.good_history():
    #           self._rightLine.use_last_good_fit()
    #         else:
    #           candidateFit = LineFit.new_line_fit(warped, self._params, isLeft=False)
    #     else:
    #       candidateFit = LineFit.new_line_fit(warped, self._params, isLeft=False)
    #       if self._rightLine.is_good_fit(candidateFit, self._leftLine):
    #         self._rightLine.apply_new_fit(candidateFit)
    #       else:
    #         self._rightLine.error_mode(on=True)
    #     ## At this point the data should've been updated!


    #     self._debug = False
    #     if not (self._leftLine.fit_exists() and self._rightLine.fit_exists()):
    #       # Get out here
    #       return None
        
    #     # Create an image to draw on and an image to show the selection window
    #     warped      = warped.astype(np.uint8)
    #     out_img     = np.dstack((warped, warped, warped)) * 255
    #     window_img  = np.zeros_like(out_img)

    #     # Color in left and right line pixels
    #     out_img[self._leftLine.ally,  self._leftLine.allx]  = [255, 0, 0]
    #     out_img[self._rightLine.ally, self._rightLine.allx] = [0, 0, 255]

    #     # Generate a polygon to illustrate the search window area
    #     # And recast the x and y points into usable format for cv2.fillPoly()
    #     left_line_window1   = np.array([np.transpose(np.vstack([self._leftLine.allx - self._params['fit']['margin'], self._leftLine.ally]))])
    #     left_line_window2   = np.array([np.flipud(np.transpose(np.vstack([self._leftLine.allx + self._params['fit']['margin'], self._leftLine.ally])))])
    #     left_line_pts       = np.hstack((left_line_window1, left_line_window2))
        
    #     # Draw out the right side!
    #     right_line_window1  = np.array([np.transpose(np.vstack([self._rightLine.allx - self._params['fit']['margin'], self._rightLine.ally]))])
    #     right_line_window2  = np.array([np.flipud(np.transpose(np.vstack([self._rightLine.allx + self._params['fit']['margin'], self._rightLine.ally])))])
    #     right_line_pts      = np.hstack((right_line_window1, right_line_window2))

    #     # Draw the lane onto the warped blank image
    #     cv2.fillPoly(window_img, np.int_([left_line_pts]),  (0, 255, 0))
    #     cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))

    #     return cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    @staticmethod
    def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
        # Convert to grayscale
        #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
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

    def overlay_green_zone (self, undist, warped, Minv):
        """ overlay_green_zone - draw out the green zone here. """

        if not self._leftLine.fit_exists() or not self._rightLine.fit_exists():
            # We can't really do anything here, hence, need to get out
            return undist
        
        leftLaneFit = self._leftLine.get_smooth_fit()
        rightLaneFit = self._rightLine.get_smooth_fit()
        
        # Get the x,y of the fitted lines
        ploty = np.linspace(0, undist.shape[0]-1, undist.shape[0])
        left_fitx = leftLaneFit[0]*ploty**2 + leftLaneFit[1]*ploty + leftLaneFit[2]
        right_fitx = rightLaneFit[0]*ploty**2 + rightLaneFit[1]*ploty + rightLaneFit[2]
        
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
        
        # Combine the result with the original image
        return cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    def make_out_path (self, img_type):
        return path.join(self._output_dir, "%d_%s.jpg" % (self._runId, img_type))

    def overlay_data (self, img_greenzone, gray_bin, warped, lines):
        # Now draw out the current curvature, center of the lane, etc.
        height, width  = img_greenzone.shape[0], img_greenzone.shape[1]
        # Create the thumbmail images
        thumb_h, thumb_w = int(height * self._params['thumb_ratio']), int(width * self._params['thumb_ratio'])
        thumb_size = (thumb_w, thumb_h)

        # Calculate thumbnail images
        gray_bin = np.dstack((gray_bin, gray_bin, gray_bin)) * 255
        thumb_gray_bin = cv2.resize(gray_bin, dsize = thumb_size)

        # Lines are already in the right format, hence no stacking needed
        if not(lines is None):
          thumb_lines = cv2.resize(lines, dsize = thumb_size)

        warped = np.dstack((warped, warped, warped)) * 255
        thumb_warped = cv2.resize(warped,    dsize = thumb_size)

        off_x, off_y = 20, 45

        # Add a semi-transparent rectangle to highlight thumbnails on the left
        mask = cv2.rectangle(img_greenzone.copy(), (0, 0), (2 * off_x + thumb_w, height), (0, 0, 0), thickness=cv2.FILLED)
        img_blend = cv2.addWeighted(src1 = mask, alpha = 0.2, src2 = img_greenzone, beta = 0.8, gamma = 0)

        # Stitch thumbnails here
        img_blend[off_y : off_y + thumb_h, off_x : off_x + thumb_w, :]                          = thumb_gray_bin
  
        if not (lines is None):
          img_blend[2 * off_y + thumb_h : 2 * (off_y + thumb_h), off_x : off_x + thumb_w, :]      = thumb_lines
        img_blend[3 * off_y + 2 * thumb_h : 3 * (off_y + thumb_h), off_x : off_x + thumb_w, :]  = thumb_warped
        
        if not (self._leftLine.radius_of_curvature is None or self._rightLine.radius_of_curvature is None):
          mean_curvature_meter = (self._leftLine.radius_of_curvature + self._rightLine.radius_of_curvature) / 2.0

          left_bottom_x = self._leftLine.offset
          right_bottom_x = self._rightLine.offset
          offset = (img_greenzone.shape[1] / 2 * LineFit.xm_per_pix) - (left_bottom_x + right_bottom_x)


          # The net effect should be the overall offset from the center
          offset_meter = self._leftLine.offset + self._rightLine.offset

          font = cv2.FONT_HERSHEY_SIMPLEX
          # Write out the curvatures - left, right and avg
          cv2.putText(img_blend, 'Curvature left : %10.1f' % self._leftLine.radius_of_curvature, (400, 60), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
          cv2.putText(img_blend, 'Curvature right: %10.1f' % self._rightLine.radius_of_curvature, (400, 90), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
          cv2.putText(img_blend, 'Curvature avg  : %10.1f' % mean_curvature_meter, (400, 120), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

          cv2.putText(img_blend, 'Offset from center: %.2f' % offset_meter, (860, 130), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        else:
          font = cv2.FONT_HERSHEY_SIMPLEX
          cv2.putText(img_blend, 'Curvature not available', (860, 60), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        return img_blend

    def process_frame (self, img, overlay=False):

        self._currframe += 1
        if self._debug: print ('Processing frame %.1d' % self._currframe)

        # Step 1: Undistort the image
        try:
          undist = self.undistort_img(img, self._cameraCaleb)
          if self._debug: print ('Undistorted.')
        except Exception as e:
          raise Exception ('Failed to undistort [%s]' % str(e))

        if self._debug:
          # Print out the image for this run
          outpath = self.make_out_path("undistorted")
          print ('Writing out undistorted file to ' + outpath)
          plt.clf()
          plt.imshow(undist)
          plt.savefig(outpath)

        # Step 2: Color and gradient calculations
        try:
          gray_bin = self.translate_color (undist)
          if self._debug: print ('Color and gradient calculations')
        except Exception as e:
          raise Exception ('Failed to run color pipeline [%s]' % str(e))
        if self._debug:
          outpath = self.make_out_path("gray")
          print ('Writing out gray file to ' + outpath)
          plt.clf()
          plt.imshow(gray_bin)
          plt.savefig(outpath)

        # Step 3: Transformation
        try:
          warped, M, Minv = self.transform(gray_bin)
          if self._debug: print ('Transformation')
        except Exception as e:
          raise Exception ('Failed to transform [%s]' % str(e))
        if self._debug:
          outpath = self.make_out_path("warped")
          print ('Writing out warped file to ' + outpath)
          plt.clf()
          plt.imshow(warped)
          plt.savefig(outpath)

        # Step 4: Update lane information
        try:
          lines = self.fit_lines (warped, alwaysNew=False)
          if self._debug: print ('Updating fits')
        except Exception as e:
          raise Exception ('Failed to update line fits [%s]' % str(e))
        
        # Step 5: Overlay the green zone
        try:
          img_greenzone = self.overlay_green_zone (undist, warped, Minv)
          if self._debug: print ('Greenzone')
        except Exception as e:
          raise Exception ('Failed to overlay green zone [%s]' % str(e))
        
        if self._debug:
          outpath = self.make_out_path("greenzone")
          print ('Writing out greenzone file to ' + outpath)
          plt.clf()
          plt.imshow(img_greenzone)
          plt.savefig(outpath)

        if overlay:
            img_greenzone = self.overlay_data (img_greenzone, gray_bin, warped, lines)
        return img_greenzone
    
    @staticmethod
    def undistort_img(img, cameraCaleb):
        return cv2.undistort(img, cameraCaleb['mtx'], cameraCaleb['dist'], None, cameraCaleb['mtx'])

    # Define a function that takes an image, number of x and y points, 
    # camera matrix and distortion coefficients
    @staticmethod
    def transform(img):

        # Get the dimensions
        width, height = img.shape[1], img.shape[0]
        img_size = (width, height)

        # define the trapezoid

        src = np.float32([[605, 445], [685, 445],
                          [1063, 676], [260, 676]])
        
        dst = np.float32([[width * 0.35, 0], [width * 0.65, 0], 
                          [width * 0.65, height], [ width * 0.35, height]])

        #src = np.float32( [[200, 720], [1100, 720], [595, 450], [685, 450]])

        #dst = np.float32( [[300, 720], [980, 720], [300, 0], [980, 0]])

        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)

        # Warp the image using OpenCV warpPerspective()
        return cv2.warpPerspective(img, M, (int(width), int(height)), flags=cv2.INTER_LINEAR), M, Minv
    
