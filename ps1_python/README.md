# Problem Set 0: Images as Functions

###1. Edge image
a0) Input image - a1) Edge image  
<img src="input/ps1-input0.png" height="180">
<img src="output/ps1-1-a-1.png" height="180">

###2. Hough Transform for Lines
a) Hough space - b) Highlighted Peaks - c) Highlighted lines  
<img src="output/ps1-2-a-1.png" height="180" width="60">
<img src="output/ps1-2-b-1.png" height="180" width="60">
<img src="output/ps1-2-c-1.png" height="180">

###3. Hough Transform for Lines on Noisy Image
a0) Noisy input Image - a1) Filtered image - b1) Edge image of noisy image - b2) Edge image of filtered image  
<img src="input/ps1-input0-noise.png" height="180">
<img src="output/ps1-3-a-1.png" height="180">
<img src="output/ps1-3-b-1.png" height="180">
<img src="output/ps1-3-b-2.png" height="180">  
c1) Hough space of edge image - c2) Noisy input image with highlighted lines  
<img src="output/ps1-3-c-1.png" height="180" width="60">
<img src="output/ps1-3-c-2.png" height="180">

###4. Hough Transform for Lines on a more Complex Image
a0) Input image - a1) Filtered input image - b1) Edge image  
<img src="input/ps1-input1.png" height="180">
<img src="output/ps1-4-a-1.png" height="180">
<img src="output/ps1-4-b-1.png" height="180">  
c1) Hough space - c2) Original input image with highlighted lines  
<img src="output/ps1-4-c-1.png" height="180" width="60">
<img src="output/ps1-4-c-2.png" height="180">

###5. Hough Transform for Circles
a0) Input image - a1) Smoothed image - a2) Edge image  
<img src="input/ps1-input1.png" height="180">
<img src="output/ps1-5-a-1.png" height="180">
<img src="output/ps1-5-a-2.png" height="180">  
a3) Detected circles for r=20px - b1) Detected circles for r in [20,50] px  
<img src="output/ps1-5-a-3.png" height="180">
<img src="output/ps1-5-b-1.png" height="180">  

###6. Hough Transform for Lines on Cluttered Image
a0) Input image - a1) Highlighted detected lines - c1) Detected Lines under constraints  
<img src="input/ps1-input2.png" height="180">
<img src="output/ps1-6-a-1.png" height="180">
<img src="output/ps1-6-c-1.png" height="180">  
c) The boundaries of the pens were extracted by selecting the lines that complied
with two constraints. In particular, the selection algorithm only kept the lines
that had at least one approximate parallel line (delta\_theta < delta\_theta\_max)
and in a distance smaller than rho\_max.

###7. Hough Transform for Circles on Cluttered Image
a0) Input image - a1) Highlighted circles  
<img src="input/ps1-input2.png" height="180">
<img src="output/ps1-7-a-1.png" height="180">

###8. Hough Transform for Lines and Circles on Distorted Cluttered Image
a0) Input image - a1) Highlighted lines and circles  
<img src="input/ps1-input3.png" height="180">
<img src="output/ps1-8-a-1.png" height="180">  
b) The circles in the distorted input image are actually ellipses and thus cannot
be accurately detected by a Hough transform for circles. This could probably be
solved by using a Hough transform for ellipses or by applying a homography transform
to partially fix the distortion and turn the ellipses back to circles.

