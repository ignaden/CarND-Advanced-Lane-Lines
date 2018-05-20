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

def run_test_images(params):
    """ Run all of the test images """

    for idx, g in enumerate(glob.glob("../test_images/*.jpg")):
        print ("processing [%s]" % g)

        l = Lines(idx, cameraCaleb, params, True, "test_output")
        img = l.process_frame (mpimg.imread(g), True)

        outpath = "../test_images_output/test_%d.jpg" % idx
        outpath_hist = "../test_images_output/test_hist%d.jpg" % idx

        plt.clf()
        plt.imshow(img)
        plt.savefig(outpath)

lines = None

def process_image(img):
    """ """
    global lines
    return lines.process_frame(img, True)

def run_video(params):

    # Global lines object
    global lines
    lines = Lines(0, cameraCaleb, params, False, "video")

    # White how?
    white_output = '../output_videos/project_video.mp4'

    # video clip
    clip1 = VideoFileClip("../input_videos/project_video.mp4") #.subclip(0,40)

    #clip1.save_frame("frame.jpg", t=12)
    #return

    white_clip = clip1.fl_image(process_image)

    # save the clip
    white_clip.write_videofile(white_output, audio=False)


params = {
  'color_trans' : {
    's_thresh' : (100, 255),
    'sx_thresh' : (20, 100)
  },

  'fit' : {
    'margin' : 50,
    'minpix' : 15,
    'minpoint_fract' : 0.25,
    'maxpoint_fract' : 0.80,
    'nwindows' : 9
  },

  'thumb_ratio': 0.25
}

if __name__ == "__main__":
  testRun = True

  # Run test images
  if testRun:
    try:
      run_test_images(params)
    except Exception as e:
      print ("Failed to run test images [%s]" % str(e))

  else:

    # Do a run of video processing
    try:
      run_video(params)
    except Exception as e:
        print ('Failed to run video translation [%s]' % str(e))