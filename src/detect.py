
import cv2
import numpy as np
import pickle
from camera import load_params

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

    def updateFitDetails(self, new_fit, indexes):
        """ Update the new fit details """

        self.detected = True

        self._current_fit = new_fit
        self._indexes = indexes

        if self.best_fit is None:
            self.best_fit = new_fit

        # We can now 

class Lines:
    """ Contains information about both of the lines """

    def __init__ (self, cameraCaleb):
        """ """        
        self._cameraCaleb = cameraCaleb # Calibration parameters for the camera
        self._leftLine, self._rightLine = Line(Line.LEFT), Line(Line.RIGHT)
        self._last_s_binary = None
    
    def update_left_line (self):
        """ """
        pass
    
    def new_left_line(self, binary_warped):
        """ fitPolyLanes - fit a polynomial curve """
        
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        print(binary_warped.shape)
        
        #histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
        histogram = np.sum(binary_warped, axis=0)
        
        #plt.show(histogram)
        
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        minpoint = np.int(histogram.shape[0] * 0.25)
        maxpoint = np.int(histogram.shape[0] * 0.75)
        leftx_base = np.argmax(histogram[minpoint:midpoint]) + minpoint

        # Choose the number of sliding windows
        nwindows = 9
        
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Current positions to be updated for each window
        leftx_current = leftx_base
        
        # Set the width of the windows +/- margin
        margin = 100
        
        # Set minimum number of pixels found to recenter window
        minpix = 50
        
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0,255, 0), 2) 

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)

        # Now - update the Line object
        self._leftLine.updateFitDetails(left_fit, left_lane_inds)
        
    def update_right_line (self):
        """ """
        pass
    
    def new_right_line (self, binary_warped):
        """ fitPolyLanes - fit a polynomial curve """
        
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        print(binary_warped.shape)
        
        #histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
        histogram = np.sum(binary_warped, axis=0)
        
        #plt.show(histogram)
        
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        minpoint = np.int(histogram.shape[0] * 0.25)
        maxpoint = np.int(histogram.shape[0] * 0.75)
        rightx_base = np.argmax(histogram[midpoint:maxpoint]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Current positions to be updated for each window
        rightx_current = rightx_base
        
        # Set the width of the windows +/- margin
        margin = 100
        
        # Set minimum number of pixels found to recenter window
        minpix = 50
        
        # Create empty lists to receive left and right lane pixel indices
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0,255, 0), 2) 
            
            # Identify the nonzero pixels in x and y within the window
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            right_lane_inds.append(good_right_inds)
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        right_fit = np.polyfit(righty, rightx, 2)
        
        self._rightLine.updateFitDetails(right_fit, right_lane_inds)


    def update_line_fits (self, warped):
        """ Attempt to use existing fits, restart if not. """

        if self._leftLine.fitExists():
            print ('Left line already exists - updating it')
            self.update_left_line(warped)
        
        # Note that either there was no original fit, or the update failed
        # we therefore might need to do another fit
        if not self._leftLine.fitExists():
            print ('No left line - creating a new fit')
            self.new_left_line(warped)
        
        if self._rightLine.fitExists():
            print ('Right line exists - updating it')
            self.update_right_line(warped)
        
        if not self._rightLine.fitExists():
            print ('No right line - creating a new fit')
            self.new_right_line(warped)

    def pipeline (self, img, s_thresh=(170, 255), sx_thresh=(20, 100)):
        img = np.copy(img)
        
        # Convert to HLS color space and separate the V channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        
        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        
        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
        
        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        
        # Stack each channel
        # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
        # be beneficial to replace this channel with something else.
        color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

        return s_binary

    def overlay_green_zone (self, undist, warped, Minv):
        """ overlay_green_zone - draw out the green zone here. """

        if not self._leftLine.fitExists() or not self._rightLine.fitExists():
            # We can't really do anything here, hence, need to get out
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
        
    def process_frame (self, img):
        """ The 'main' function """
        # Step 1: Undistort the image
        undist = self.undistort_img(img, self._cameraCaleb)
        print ('Undistorted')

        # Step 2: Color and gradient calculations
        gray_bin = self.pipeline (undist)
        print ('Color and gradient calculations')

        # Step 3: Transformation
        warped, M, Minv = self.transform(gray_bin)
        print ('Transformation')

        # Step 4: Update lane information
        self.update_line_fits (warped)
        print ('Updating fits')

        # Step 5: Overlay the green zone
        img_greenzone = self.overlay_green_zone (undist, warped, Minv)
        print ('Greenzone')

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

        #src = np.float32([[605, 445], [685, 445],
        #                  [1063, 676], [260, 676]])
        #
        #dst = np.float32([[width * 0.35, 0], [width * 0.65, 0], 
        #                  [width * 0.65, height], [ width * 0.35, height]])

        src = np.float32( [[200, 720], [1100, 720], [595, 450], [685, 450]])

        dst = np.float32( [[300, 720], [980, 720], [300, 0], [980, 0]])


        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)


        # Warp the image using OpenCV warpPerspective()
        return cv2.warpPerspective(img, M, (int(width), int(height)), flags=cv2.INTER_LINEAR), M, Minv
    

def fitPolyLanes(binary_warped):
    """ fitPolyLanes - fit a polynomial curve """
    
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    
    #histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    histogram = np.sum(binary_warped, axis=0)
    
    #plt.show(histogram)
    
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    minpoint = np.int(histogram.shape[0] * 0.30)
    maxpoint = np.int(histogram.shape[0] * 0.70)
    leftx_base = np.argmax(histogram[minpoint:midpoint]) + minpoint
    rightx_base = np.argmax(histogram[midpoint:maxpoint]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    
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
    margin = 100
    
    # Set minimum number of pixels found to recenter window
    minpix = 50
    
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
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0,255, 0), 2) 
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0,255, 0), 2) 
        
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

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    return left_fit, right_fit, left_lane_inds, right_lane_inds, histogram, out_img
