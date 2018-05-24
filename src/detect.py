import cv2
import numpy as np
import pickle
from os import path
from camera import load_params  
import matplotlib.pyplot as plt

from linefit import LineFit
from line import Line
from preprocessor import FramePreProcessor

class Lines:
    """ Contains information about both of the lines """

    def __init__ (self, runId, cameraCaleb, transParams, debug=False, output_dir="."):
      """ """
      self._currframe = 0
      
      # Will help us preprocess 
      self._preprocessor = FramePreProcessor(cameraCaleb, transParams)

      self._debug = debug             # Are we in debug mode?
      self._runId = runId             # Run ID of all the current
      self._params = transParams      # Color transform and fitting parameters
      self._output_dir = output_dir   # Output directory for debug images
      self._cameraCaleb = cameraCaleb # Calibration parameters for the camera
      self._leftLine, self._rightLine = Line(Line.LEFT), Line(Line.RIGHT)
      self._last_s_binary = None

    def fit_lines (self, warped, alwaysNew=True):
      """ Attempt to use existing fits, restart if not. """
        
      # Update the left line
      if self._leftLine.fit_exists() and not alwaysNew:
        if self._debug:
          print("Will look to update existing fit for left line!")
        leftCandidate = LineFit.update_fit(self._leftLine.best_fit, warped, self._params, isLeft=True)
      else:
        # Try to fit a new one
        if self._debug:
          print("Will look to create a new fit for left line!")
        leftCandidate = LineFit.new_line_fit(warped, self._params, isLeft=True)
        
      if self._rightLine.fit_exists() and not alwaysNew:
        if self._debug:
          print("Will look to update existing fit for right line!")
        rightCandidate = LineFit.update_fit(self._rightLine.best_fit, warped, self._params, isLeft=False)
      else:
        # Try to fit a new one
        if self._debug:
          print ("Will look to create a new fit for right line!")
        rightCandidate = LineFit.new_line_fit(warped, self._params, isLeft=False)

      # Now that we should have some fits, let's see whether we can apply them!

      # TODO: ideally, we should consider these separately!
      if LineFit.is_good_fit(leftCandidate, rightCandidate):
        # use these to update the current ones
        if self._debug:
          print ("Will apply the new fits!")
        self._leftLine.apply_new_fit(leftCandidate)
        self._rightLine.apply_new_fit(rightCandidate)

      else:
        if self._debug:
          print ("Will use last good fit :(")

        self._leftLine.use_last_good_fit()
        self._rightLine.use_last_good_fit()

      if not (self._leftLine.fit_exists() and self._rightLine.fit_exists()):
        if self._debug:
          print ("It looks like we dont have anything to go on with")
        return None

      ## At this point, we should have both of the lines - let's try to draw them here.

      # Create an image to draw on and an image to show the selection window
      warped      = warped.astype(np.uint8)
      out_img     = np.dstack((warped, warped, warped)) * 255
      window_img  = np.zeros_like(out_img)

      # Color in left and right line pixels
      out_img[self._leftLine.ally,  self._leftLine.allx]  = [255, 0, 0]
      out_img[self._rightLine.ally, self._rightLine.allx] = [0, 0, 255]

      # Generate a polygon to illustrate the search window area
      # And recast the x and y points into usable format for cv2.fillPoly()

      left_fit = self._leftLine.best_fit
      right_fit = self._rightLine.best_fit

      ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )

      left_fitx   = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
      right_fitx  = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

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

      # Add the resulting overlaid image on top of this
      result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

      # Now draw out the fitted curves!
      points = np.vstack((left_fitx, ploty)).astype(np.int32).T
      result = cv2.polylines(result, [points], False, (255,255,0), 10)
      points = np.vstack((right_fitx, ploty)).astype(np.int32).T
      result = cv2.polylines(result, [points], False, (255,255,0), 10)

      # 
      return result

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
      """ Create a path for storing images """

      return path.join(self._output_dir, "%d_%s.jpg" % (self._runId, img_type))

    def overlay_data (self, img_greenzone, gray_bin, warped, lines):
        """ Create a frame with overlaid thumbnail images """

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

        img_blend[off_y : off_y + thumb_h, off_x : off_x + thumb_w, :] = thumb_gray_bin
        img_blend[2 * off_y + thumb_h : 2 * (off_y + thumb_h), off_x : off_x + thumb_w, :] = thumb_warped

        if not (lines is None):
          img_blend[3 * off_y + 2 * thumb_h : 3 * (off_y + thumb_h), off_x : off_x + thumb_w, :] = thumb_lines
        
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

        self._currframe += 1
        if self._debug: print ('Processing frame %.1d' % self._currframe)

        # Step 1: Undistort the image
        try:
          undist = self._preprocessor.undistort_img(img)
        except Exception as e:
          raise Exception ('Failed to undistort [%s]' % str(e))

        if self._debug:
          # Print out the image for this run
          outpath = self.make_out_path("undistorted")
          print ('Writing out undistorted file to ' + outpath)
          plt.clf()

          f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
          f.tight_layout()
          ax1.imshow(img)
          ax1.set_title('Original Image', fontsize=50)
          ax2.imshow(undist)
          ax2.set_title('Undistorted', fontsize=50)
          plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
          plt.savefig(outpath)

        # Step 2: Color and gradient calculations
        try:
          gray_bin = self._preprocessor.translate_color (undist)
          if self._debug: print ('Color and gradient calculations')
        except Exception as e:
          raise Exception ('Failed to run color pipeline [%s]' % str(e))

        if self._debug:
          outpath = self.make_out_path("gray")
          print ('Writing out gray file to ' + outpath)
          plt.clf()
          f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
          f.tight_layout()
          ax1.imshow(img)
          ax1.set_title('Undistorted Original Image', fontsize=50)
          ax2.imshow(gray_bin, cmap='gray')
          ax2.set_title('After color conversion, gradient threshold', fontsize=50)
          plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
          plt.savefig(outpath)

        # Step 3: Transformation
        try:
          warped, M, Minv = self._preprocessor.transform(gray_bin)
          if self._debug: print ('Transformation')
        except Exception as e:
          raise Exception ('Failed to transform [%s]' % str(e))
        if self._debug:
          outpath = self.make_out_path("warped")
          print ('Writing out warped file to ' + outpath)
          plt.clf()
          f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
          f.tight_layout()
          ax1.imshow(gray_bin, cmap='gray')
          ax1.set_title('Thresholded grayscale', fontsize=50)
          ax2.imshow(warped, cmap='gray')
          ax2.set_title('Warped', fontsize=50)
          plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
          plt.savefig(outpath)
        
        # Step 4: Update lane information
        try:
          lines = self.fit_lines (warped, alwaysNew=False)
          if self._debug: print ('Updating fits')
        except Exception as e:
          raise Exception ('Failed to update line fits [%s]' % str(e))
        
        if self._debug and lines is not None:
          outpath = self.make_out_path("lines")
          plt.clf()
          plt.imshow(lines)
          plt.savefig(outpath)

        # Step 5: Overlay the green zone
        try:
          img_greenzone = self.overlay_green_zone (undist, warped, Minv)

        except Exception as e:
          raise Exception ('Failed to overlay green zone [%s]' % str(e))
        
        if self._debug and lines is not None:
          outpath = self.make_out_path("greenzone")

          print ('Writing out greenzone file to ' + outpath)
          plt.clf()
          plt.imshow(img_greenzone)
          plt.savefig(outpath)

        if overlay and lines is not None:
          img_greenzone = self.overlay_data (img_greenzone, gray_bin, warped, lines)

        return img_greenzone
