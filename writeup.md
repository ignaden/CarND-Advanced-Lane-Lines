## The Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

All of the source code is in the `src/` directory, including `camera.py` (lines 27 - 100) which calibrates the camera and saves the resulting parameters as a `pickle` file for video processing later.

Camera calibration follows code provided by the Udacity course materials.

The steps are as follows:
1. Create a grid of 9x6 points in 'real world space'.
2. For each provided image:
    a. Use CV2's `findChessboardCorners` to locate corresponding points in the image.
    b. Keep track of these points by appending both, the 'real world' and 'image', to corresponding lists.
3. Use CV2's `calibrateCamera` to create a transformation matrix from the two lists of points from step `2` above.

Here's an example of an image that's been undistorted by the camera effect:
![alt text][image1]

Note that for 3 (out of 20) images, CV2 could not locate the points. I presume this is a property of the CV2 algorithm.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Here's an example:

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

All of the steps prior to finding actual lanes are contained within `src/preprocessor.py` file.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for perspective transform logic is contained within `transform` method of `FramePreProcessor` class in `src/preprocessor.py` file. The lines are 70 to 90.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
    src = np.float32([[605, 445],
                      [685, 445],
                      [1063, 676],
                      [260, 676]])
    
    dst = np.float32([[width * 0.35, 0],
                      [width * 0.65, 0], 
                      [width * 0.65, height],
                      [width * 0.35, height]])
```

With height equal to `720` and width to `1280`, this resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 605, 445      | 448, 0        |
| 685, 445      | 448, 720      |
| 1063, 676     | 832, 720      |
| 260, 676      | 832, 0        |


Here's an example of the transformation:
![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

All of the code for lane/line fitting is contained within `src/linefit.py` - it's based on the code provided by the course materials.

There're two modes of lane detection:
- Create new fits (`new_line_fit`, lines 41 to 138 in `src/linefit.py`):
 - As per course material, I create a number of stacked windows for each lane
 - All of the non-zero X points within the windows are used to 
 
- Update existing ones (`update_fit`, lines 141 to 175 in `src/linefit.py`):
    - In cases where we have no prior lanes detected or the last detection did not converge (I describe criteria below).
    - The process is similar to the one above, except that we use the existing curve fits to identify the centers of the windows.


![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius and offset are calculated in the constructor of `LineFit` class in `src/linefit.py`, lines 25 to 27.

```python
fit_cr = np.polyfit(self.ally * self.ym_per_pix, self.allx * self.xm_per_pix, 2)
self.radius_of_curvature = ((1.0 + (2.0 * fit_cr[0] * y_bottom * self.ym_per_pix + fit_cr[1]) ** 2.0) ** 1.5) / np.absolute(2.0 * fit_cr[0])
self.slope = 2 * fit_cr[0] * y_bottom * self.ym_per_pix + fit_cr[1]

```
The code implements formulas provided in course materials. The basic principle is that we can directly solve for the both, curvature (which is a second derivative ) and the offset.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The biggest challenge I faced was ensuring that the noise from bright areas of the road would not interfere with curve fitting. I have encoded a small test - that the distance between points at the top is not signficantly different from the distance of between those at the bottom. If the difference between the distances is over a configurable threshold, then the fits would not be updated - i.e. the frame uses old fit data.

Here's the function that estimates this:
```python
@staticmethod
def is_good_fit (leftLine, rightLine):
    """ It's a good fit if the distance between the lines at the top is similar to the bottom. """
    
    max_dist = 0.10

    lf = leftLine.line_fit
    rf = rightLine.line_fit

    y_top, y_mid, y_bottom = 0, 350, 719

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
```



