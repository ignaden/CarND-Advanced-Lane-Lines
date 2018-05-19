import cv2
import numpy as np
import pickle
from os import path
from camera import load_params
import matplotlib.pyplot as plt 

from linefit import LineFit
    
class Line:
    ''' 
      Define a class to receive the characteristics of each line detection
    '''

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