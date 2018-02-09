
import cv2
import glob
import numpy as np
import pickle
from camera import load_params
from detect import *
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt 

cameraCaleb = load_params()


def visualisePolyFit (outfilepath, binary_warped, left_fit, right_fit, leftLaneInds, rightLaneInds):
    """ visualisePolyFit """

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100

    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +  left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +  right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))

    # Plot the actual lines
    plt.clf()

    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

    plt.savefig(outfilepath)

def overlayGreenZone(undist, warped, Minv, leftLaneFit, rightLaneFit):
    """ Overlay a green zone on the original image """
    
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

def run_test_images(params):
    """ Run all of the test images """

    for idx, g in enumerate(glob.glob("../test_images/*.jpg")):
        print ("processing [%s]" % g)

        l = Lines(idx, cameraCaleb, params, True, "test_output")
        #img = l.process_frame(cv2.imread(g))
        img = l.process_frame (mpimg.imread(g))

        outpath = "../test_images_output/test_%d.jpg" % idx
        outpath_hist = "../test_images_output/test_hist%d.jpg" % idx

        # Save th
        plt.clf()
        plt.imshow(img)
        plt.savefig(outpath)
#        cv2.imwrite(outpath, np.as_int(warped) * 255)

lines = None

def process_image(img):
    """ """
    global lines
    return lines.process_frame(img)

def run_video(params):

    # 
    global lines
    lines = Lines(0, cameraCaleb, params, False, "video")

    # White how?
    white_output = '../output_videos/project_video.mp4'

    # video clip
    clip1 = VideoFileClip("../input_videos/project_video.mp4")
    white_clip = clip1.fl_image(process_image)

    # save the clip
    white_clip.write_videofile(white_output, audio=False)

params = {
  'color_trans' : {
    's_thresh' : (170, 255),
    'sx_thresh' : (20, 100)
  },

  'fit' : {
    'margin' : 50,
    'minpix' : 20,
    'minpoint_fract' : 0.25,
    'maxpoint_fract' : 0.80,
    'nwindows' : 9
  }
}

if __name__ == "__main__":

    # run test images
    #try:
    #    run_test_images(params)
    #except Exception as e:
    #    print ("Failed to run test images [%s]" % str(e))

    # do the video processing
    try:
      run_video(params)
    except Exception as e:
      print ('Failed to run video translation [%s]' % str(e))