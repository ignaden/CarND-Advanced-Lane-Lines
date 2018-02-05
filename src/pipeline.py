

# Read in the original test image
def test_image():
    #img = cv2.imread ('test_images/test5.jpg')
    img = cv2.imread('test_images/test2.jpg')
    # undistort it
    undist = undistort_img(img)

    # color and gradient
    gray_bin = pipeline(undist)

    # transform
    warped, M, Minv = transform(gray_bin)

    # fit lanes (left & right)
    leftLane, rightLane, leftLaneInds, rightLaneInds, histogram, out_img = fitPolyLanes(warped)

    # draw lane fits
    drawn = visualisePolyFit(warped, leftLane, rightLane, leftLaneInds, rightLaneInds)

    # draw the green zone for the car
    img_zone = overlayGreenZone (undist, warped, leftLane, rightLane)

    plt.imshow(histogram)


def run_test_images():
    """ Run all of the test images """

    for g in glob.glob("../test_images"):
        pass

        


def run_video():
    white_output = 'output_videos/challenge_video.mp4'
    clip1 = VideoFileClip("input_videos/challenge_video.mp4").subclip(0,10)
    white_clip = clip1.fl_image(process_image)