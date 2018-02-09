
import cv2
import numpy as np
import pickle
from os import path
from camera import load_params
import matplotlib.pyplot as plt 


# Define a class to receive the characteristics of each line detection
class Line():

    LEFT, RIGHT = 1, 2

    """ class Line:  """
    def __init__(self, side=None):
        # Save the side - this will be used later
        self._side = side

        # was the line detected in the last iteration?
        self.detected = False
        
        # x values of the last n fits of the line
        self.recent_xfitted = []
        
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        
        #x values for detected line pixels
        self.allx = None
        
        #y values for detected line pixels
        self.ally = None
    
    def fitExists(self):
        return self.detected

    def updateFitDetails(self, new_fit, allx, ally):
        """ Update the new fit details """
        self.detected = True
        self.current_fit = new_fit
        self.allx = allx
        self.ally = ally

        if self.best_fit is None:
            self.best_fit = new_fit
        
        # We should now be able to calculate the circumference!

class Lines:
    """ Contains information about both of the lines """

    def __init__ (self, runId, cameraCaleb, transParams, debug=False, output_dir="."):
        """ """
        self._debug = debug             # Are we in debug mode?
        self._runId = runId             # Run ID of all the current
        self._params = transParams      # Color transform and fitting parameters
        self._output_dir = output_dir   # Output directory for debug images
        self._cameraCaleb = cameraCaleb # Calibration parameters for the camera
        self._leftLine, self._rightLine = Line(Line.LEFT), Line(Line.RIGHT)
        self._last_s_binary = None

    def update_line_fits (self, binary_warped, fit_left=True, fit_right=True):
        """ Re-use last fits to re-fit the line for new binary image """
        
        binary_warped = binary_warped.astype(np.uint8)

        # Assume you now have a new warped binary image 
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        margin = 100

        if fit_left:
            left_fit = self._leftLine.current_fit

            left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                              left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                              left_fit[1]*nonzeroy + left_fit[2] + margin))) 
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds] 

            left_fit = np.polyfit(lefty, leftx, 2)

            self._leftLine.updateFitDetails(left_fit, leftx, lefty)

        if fit_right:
            right_fit = self._rightLine.current_fit

            right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
                              right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
                              right_fit[1]*nonzeroy + right_fit[2] + margin)))  

            # Again, extract left and right line pixel positions
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
        
            right_fit = np.polyfit(righty, rightx, 2)

            self._rightLine.updateFitDetails(right_fit, rightx, righty)

    def fit_new_lines (self, binary_warped, fit_left=True, fit_right=True):
        """ fitPolyLanes - fit a polynomial curve """

        # Just to make sure!
        binary_warped = binary_warped.astype(np.uint8) 

        histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
              
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        
        # TODO: move the start/end into self._params!
        midpoint = np.int(histogram.shape[0] / 2)
        minpoint = np.int(histogram.shape[0] * self._params['fit']['minpoint_fract'])
        maxpoint = np.int(histogram.shape[0] * self._params['fit']['maxpoint_fract'])
        
        leftx_base = np.argmax(histogram[minpoint:midpoint]) + minpoint
        rightx_base = np.argmax(histogram[midpoint:maxpoint]) + midpoint

        # Choose the number of sliding windows
        nwindows = self._params['fit']['nwindows']
        
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        # Set the width of the windows +/- margin
        margin = self._params['fit']['margin']
        
        # Set minimum number of pixels found to recenter window
        minpix = self._params['fit']['minpix']
        
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
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
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Fit a second order polynomial to each
        if fit_left:
            left_lane_inds = np.concatenate(left_lane_inds)

            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]

            left_fit = np.polyfit(lefty, leftx, 2)
            self._leftLine.updateFitDetails(left_fit)
        
        # Do the same thing, but for the right lane
        if fit_right:
            right_lane_inds = np.concatenate(right_lane_inds)

            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]

            right_fit = np.polyfit(righty, rightx, 2)
            self._rightLine.updateFitDetails(right_fit)

    def fit_lines (self, warped):
        """ Attempt to use existing fits, restart if not. """

        # First, if there're already fit, then 
        if self._leftLine.fitExists() or self._rightLine.fitExists():
            print ('Atleast one line already exists - updating it')
            updateLeft = self._leftLine.fitExists()
            updateRight = self._rightLine.fitExists()
            self.update_line_fits(warped, updateLeft, updateRight)
        
        if not self._leftLine.fitExists() or not self._rightLine.fitExists():
            print ('Need to fit a new line!')
            updateLeft = not self._leftLine.fitExists()
            updateRight = not self._rightLine.fitExists()
            self.fit_new_lines(warped, updateLeft, updateRight)
        
        # Draw the lines here!

        if not (self._leftLine.fitExists() and self._rightLine.fitExists()):
          # Get out here
          return None
        
        # Create an image to draw on and an image to show the selection window
        warped      = warped.astype(np.uint8)
        out_img     = np.dstack((warped, warped, warped)) * 255
        window_img  = np.zeros_like(out_img)

        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]]   = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # 
        left_fitx, right_fitx = self._leftLine.bestx, self._rightLine.besty

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1   = np.array([np.transpose(np.vstack([left_fitx - self._params['fit']['margin'], ploty]))])
        left_line_window2   = np.array([np.flipud(np.transpose(np.vstack([left_fitx + self._params['fit']['margin'], ploty])))])
        left_line_pts       = np.hstack((left_line_window1, left_line_window2))
        
        # Draw out the right side!
        right_line_window1  = np.array([np.transpose(np.vstack([right_fitx - self._params['fit']['margin'], ploty]))])
        right_line_window2  = np.array([np.flipud(np.transpose(np.vstack([right_fitx + self._params['fit']['margin'], ploty])))])
        right_line_pts      = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]),  (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        
        return cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

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
        s_binary[(s_channel >= self._params['color_trans']['s_thresh'][0]) & (s_channel <= self._params['color_trans']['s_thresh'][1])] = 1
        
        # Stack each channel
        # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
        # be beneficial to replace this channel with something else.
        
        combined = np.zeros_like(s_binary)
        combined[(sxbinary == 1) | (s_binary == 1)] = 1
        
        #color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

        return combined

    def overlay_green_zone (self, undist, warped, Minv):
        """ overlay_green_zone - draw out the green zone here. """

        if not self._leftLine.fitExists() or not self._rightLine.fitExists():
            # We can't really do anything here, hence, need to get out
            print ('Cannot draw green zone')
            return undist

        leftLaneFit = self._leftLine.current_fit
        rightLaneFit = self._rightLine.current_fit
        
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

    def overlay_data (self, img_greenzone, gray_bin, warped, lines, curvature):
        """ """

        # Now draw out the current curvature, center of the lane, etc.
        height, weight = img_greenzone.shape[0], img_greenzone.shape[1]

        # Create the thumbmail images
        thumb_size = dsize(int(self._params['thumb_ratio'] * height), int(self._params['thumb_ratio'] * weight))
        thumb_weight = int(self._params['thumb_ratio'] * weight)


        # Calculate thumbs
        thumb_gray_bin  = cv2.resize(gray_bin,  dsize=self._params['thumb_size'])
        thumb_lines     = cv2.resize(lines,     dsize=self._params['thumb_size'])
        thumb_warped    = cv2.resize(warped,    dsize=self._params['thumb_size'])

        off_x, off_y    = 20, 45

        # add a semi-transparent rectangle to highlight thumbnails on the left
        mask            = cv2.rectangle(img_greenzone.copy(), (0, 0), (2*off_x + thumb_w, height), (0, 0, 0), thickness=cv2.FILLED)
        img_blend       = cv2.addWeighted(src1=mask, alpha=0.2, src2=img_greenzone, beta=0.8, gamma=0)

        # stitch thumbnails here
        img_blend[off_y:off_y+thumb_h, off_x:off_x+thumb_w, :]                  = thumb_gray_bin
        img_blend[2*off_y+thumb_h:2*(off_y+thumb_h), off_x:off_x+thumb_w, :]    = thumb_lines
        img_blend[3*off_y+2*thumb_h:3*(off_y+thumb_h), off_x:off_x+thumb_w, :]  = thumb_warped

        # Write out the current fit statistics
        # TODO calculate the 

        return img_blend

    def process_frame (self, img, overlay=False):
        """ The 'main' function """
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
          lines = self.fit_lines (warped)
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
            img_greenzone = Lines.overlay_data (img_greenzone, gray_bin, warped, lines, curvature)

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
    
