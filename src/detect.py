import cv2
import numpy as np
import pickle
from os import path
from camera import load_params
import matplotlib.pyplot as plt 

from linefit import LineFit
from line import Line
from process_img import PreProcessor


class Lines:
    """ Contains information about both of the lines """

    def __init__ (self, runId, cameraCaleb, transParams, debug=False, output_dir="."):
        """ """
        self._currframe = 0

        self._frameProcessor = PreProcessor(cameraCaleb, transParams)

        self._debug = debug             # Are we in debug mode?
        self._runId = runId             # Run ID of all the current
        self._params = transParams      # Color transform and fitting parameters
        self._output_dir = output_dir   # Output directory for debug images
        #self._cameraCaleb = cameraCaleb # Calibration parameters for the camera
        
        self._leftLine, self._rightLine = Line(Line.LEFT), Line(Line.RIGHT)
        self._last_s_binary = None

    def fit_lines (self, warped, alwaysNew=True):
        """ Attempt to use existing fits, restart if not. """
        #self._debug = True
        # Update the left line
        if self._leftLine.fit_exists() and not alwaysNew:
          leftCandidate = LineFit.update_fit(self._leftLine.best_fit, warped, self._params, isLeft=True)
        else:
          # Try to fit a new one
          leftCandidate = LineFit.new_line_fit(warped, self._params, isLeft=True)
        
        if self._rightLine.fit_exists() and not alwaysNew:
          rightCandidate = LineFit.update_fit(self._rightLine.best_fit, warped, self._params, isLeft=False)
        else:
          # Try to fit a new one
          rightCandidate = LineFit.new_line_fit(warped, self._params, isLeft=False)

        if LineFit.is_good_fit(leftCandidate, rightCandidate):
          # use these to update the current ones
          self._leftLine.apply_new_fit(leftCandidate)
          self._rightLine.apply_new_fit(rightCandidate)
        else:
          self._leftLine.use_last_good_fit()
          self._rightLine.use_last_good_fit()

        if not (self._leftLine.fit_exists() and self._rightLine.fit_exists()):
          return None

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

    @staticmethod
    def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
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
        """ 'overlay_data' here as well... """

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
        thumb_warped = cv2.resize(warped, dsize = thumb_size)

        # 
        off_x, off_y = 20, 45

        # Add a semi-transparent rectangle to highlight thumbnails on the left
        mask = cv2.rectangle(img_greenzone.copy(), (0, 0), (2 * off_x + thumb_w, height), (0, 0, 0), thickness=cv2.FILLED)
        img_blend = cv2.addWeighted(src1 = mask, alpha = 0.2, src2 = img_greenzone, beta = 0.8, gamma = 0)

        # Stitch thumbnails here
        img_blend[off_y : off_y + thumb_h, off_x : off_x + thumb_w, :] = thumb_gray_bin
  
        if not (lines is None):
          img_blend[2 * off_y + thumb_h : 2 * (off_y + thumb_h), off_x : off_x + thumb_w, :] = thumb_lines
        img_blend[3 * off_y + 2 * thumb_h : 3 * (off_y + thumb_h), off_x : off_x + thumb_w, :] = thumb_warped
        
        if not (self._leftLine.radius_of_curvature is None or self._rightLine.radius_of_curvature is None):
          curv = (self._leftLine.radius_of_curvature + self._rightLine.radius_of_curvature) / 2.0

          left_bottom_x = self._leftLine.offset
          right_bottom_x = self._rightLine.offset
          offset = (img_greenzone.shape[1] / 2 * LineFit.xm_per_pix) - (left_bottom_x + right_bottom_x)

          font = cv2.FONT_HERSHEY_SIMPLEX
          # Write out the curvatures - left, right and avg
          cv2.putText(img_blend, 'Curvature : %10.0fm' % curv, (400, 60), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
          cv2.putText(img_blend, 'Offset: %.2fm' % offset, (860, 60), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        else:
          
          font = cv2.FONT_HERSHEY_SIMPLEX
          cv2.putText(img_blend, 'Curvature not available', (860, 60), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        return img_blend

    def process_frame (self, img, overlay=False):
        """ Process a single frame:
          1. Undistort the image (by applying camera calibrated parameters)
          2. Color and gradient calculations
          3. Perspective transformation
          4. Update lane information
          5. Overlay the resulting `green zone'
        
        """

        self._currframe += 1
        if self._debug: print ('Processing frame %.1d' % self._currframe)

        #############################################################################
        # Step 1: Undistort the image
        #############################################################################
        try:
          #undist = self.undistort_img(img, self._cameraCaleb)
          undist = self._frameProcessor.undistort_img(img)
        except Exception as e:
          raise Exception ('Failed to undistort [%s]' % str(e))

        if self._debug:
          #
          # Print out the image for this run
          #
          outpath = self.make_out_path("undistorted")
          print ('Writing out undistorted file to ' + outpath)
          plt.clf()
          plt.imshow(undist)
          plt.savefig(outpath)
        
        #############################################################################
        # Step 2: Color and gradient calculations
        #############################################################################
        try:
          #gray_bin = self.translate_color (undist)
          gray_bin = self._frameProcessor.translate_color(undist)

          if self._debug: print ('Color and gradient calculations')
        except Exception as e:
          raise Exception ('Failed to run color pipeline [%s]' % str(e))
        
        if self._debug:
          outpath = self.make_out_path("gray")
          print ('Writing out gray file to ' + outpath)
          plt.clf()
          plt.imshow(gray_bin)
          plt.savefig(outpath)

        #############################################################################
        # Step 3: Transformation
        #############################################################################
        try:
          #warped, M, Minv = self.transform(gray_bin)
          warped, M, Minv = self._frameProcessor.transform(gray_bin)
          if self._debug: print ('Transformation')
        except Exception as e:
          raise Exception ('Failed to transform [%s]' % str(e))
        if self._debug:
          outpath = self.make_out_path("warped")
          print ('Writing out warped file to ' + outpath)
          plt.clf()
          plt.imshow(warped)
          plt.savefig(outpath)

        #############################################################################
        # Step 4: Update lane information
        #############################################################################
        try:
          lines = self.fit_lines (warped, alwaysNew=False)
          if self._debug: print ('Updating fits')
        except Exception as e:
          raise Exception ('Failed to update line fits [%s]' % str(e))
        
        #############################################################################
        # Step 5: Overlay the green zone
        #############################################################################
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

        # Now, this image should contain all of the information...
        return img_greenzone
    
