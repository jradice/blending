<div style="width: 800px;">
<h1>Assignment 6: Blending</h1>
<div id="introduction">
<h2 style="color: black; font-size: 130%; text-decoration: underline">Introduction</h2>
<p>In this homework assignment, we will be putting together a pyramid blending pipeline that will allow us to turn the following three images into a blended image (This assignment will require you to create a blend using your own images!). This assignment is to be done individually. You may ask others for help on Piazza, but you may NOT use other people's code outside of the code we have provided to you.</p>
<div style="width: 100%; text-align: center;"><img alt="" src="http://www.dcastro.me/cs4475/summer2013/assignment5/images/figure1.jpg" style="" /></div>
<div style="text-align: center;">
<p>This is the output of blending the above two images with the mask.</p>
<img alt="" src="http://www.dcastro.me/cs4475/summer2013/assignment5/images/output.jpg" /></div>
<p>Before we get started, <a href="http://www.dcastro.me/omscs6475/assignment6.zip" target="_blank">download this zip file</a> that contains the files you will be working with.</p>
<p>Once you have extracted the file above, you are ready to get started.</p>
<p>Above you can see the sample images which are provided for you in the images/source/sample directory. Every time you run the code, it will look inside each folder under images/, and attempt to find a folder that has images with filenames that contain 'white', 'black' and 'mask'. Once it finds a folder that contains three images with those respective names, it will apply a blending procedure to them, and save the output to images/output/. Along with the output image, it will create visualizations of the gaussian and laplacian pyramids used in the process.</p>
<p>The blending procedure takes two images and the mask and splits them into their red, green, and blue channels. It then blends each channel separately. You do not have to worry about dealing with three channels, you can assume your images take in grayscale images (as we have always done).</p>
<p>The code will construct a laplacian pyramid for the two images. It will then scale the mask image to the range [0,1] and construct a Gaussian pyramid for it. Finally, it will blend the two pyramids and collapse them to the output image.</p>
<p>Pixel values of 255 in the mask image are scaled to the value 1 in the mask pyramid, and assigned the strongest weight to the image labeled 'white' during blending. Pixel values of 0 in the mask image are scaled to the value 0 in the mask pyramid, and assigned the strongest weight to the image labeled 'black' during blending.</p>
<p>In order to facilitate this process, you will be providing six (6) key functions throughout this programming assignment that are the building blocks of blending two images.</p>
</div>
<div id="part0">
<h2 style="color: black; font-size: 130%; text-decoration: underline">Part 0: Reduce and Expand functions.</h2>
<p><strong> See Module 04-03 on Pyramids. </strong></p>
<p>As with previous assignments, running <strong> assignment6_test.py </strong> directly will apply a unit test to your code and print out helpful feedback. You can use this to debug your functions.</p>
<h3 style="color: black;">reduce</h3>
<p>This function takes an image, convolves it, and then subsamples it down to a quarter of the size (dividing the height and width by two). Note that we say subsample here. We recommend you look into numpy indexing techniques to accomplish the subsampling (essentially you want to index every other row and every other column).</p>
<p>Within the code, you are provided with a generating_kernel(a) function. This function takes a floating point number a, and returns a 5x5 generating kernel. For the reduce and expand functions, you should use <strong> a = 0.4. </strong></p>
<p>Seeing as we have already covered how convolve works in class, for this assignment we allow you to use the scipy library implementation of convolve.</p>
<p style="width: 100%; height: 40px; background-color: #DDD; border: 1px solid #222; padding: 5px;">scipy.signal.convolve2d(image, kernel, 'same')</p>
<p>This call will convolve the image and kernel, and return an array of the 'same' size as image.</p>
<h3 style="color: black;">expand</h3>
<p>This function takes an image and supersamples it to four times the size (multiplying the height and width by two (2)). After increasing the size, we have to interpolate the missing values by using the same convolution we used in reduce, and lastly we scale our output by 4.</p>
<p>Some tips in terms of how to super sample an image.</p>
<ol>
    <li>Create an image that is twice the size of the input.</li>
    <li>Review your reduce code. Look at what you did to get your output.</li>
    <li>Instead of choosing every other row / col in reduce for your output, can you assign every other row / col in expand for your output?</li>
    <li>As stressed above, look into <a href="http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html" target="_blank"> numpy indexing</a>. A key thing to note is the basic slice syntax, think about how you can use that to your advantage.</li>
</ol>
<p>For this part of the assignment, please use the generating kernel with <strong> a = 0.4 </strong> and the convolve2d function from the reduce function discussion above.</p>
</div>
<div id="part1">
<h2 style="color: black; font-size: 130%; text-decoration: underline">Part 1: Gaussian and Laplacian Pyramids</h2>
<p>In this part of the assignment, you will be implementing functions that create Gaussian and Laplacian pyramids. As usual, use <strong> assignment6_test.py </strong> to test your code. In addition, assignment6_test.py defines the functions viz_gauss_pyramid and viz_lapl_pyramid, which take a pyramid as input and return an image visualizaiton. You might want to use these functions to visualize your pyramids while debugging -- we use these at the end of the testing to output your results!</p>
<h3 style="color: black;">gaussPyramid</h3>
<p>This function takes an image and builds a pyramid out of it. The first layer of this pyramid is the original image, and each subsequent layer of the pyramid is the reduced form of the previous layer. Put simply, you are iteratively calling the reduce function on the output of the previous call, with the first call simply being the input image.</p>
<p>Please use the reduce function that you implemented in the previous part in order to write this function.</p>
<h3 style="color: black;">laplPyramid</h3>
<p>This function takes a Gaussian pyramid constructed by the previous function, and turns it into a Laplacian pyramid. The doc string contains further information about the operations you should perform for each layer.</p>
<p>Like with Gaussian pyramids, Laplacian pyramids are represented as lists of numpy arrays in the code.</p>
<p>Please use the expand function that you implemented in the previous part of the code in order to write this function.</p>
</div>
<div id="part2">
<h2 style="color: black; font-size: 130%; text-decoration: underline">Part 2: Writing the blend and collapse functions.</h2>
<p>In this part, you will be completing the pipeline by writing the actual blend function, and creating a collapse function that will allow us to convert our Laplacian pyramid into an output image.</p>
<p>As always, you can use <strong> assignment6_test.py </strong> to test your code.</p>
<h3 style="color: black;">blend</h3>
<p>This function takes three pyramids:</p>
<ul>
    <li>white - a laplacian pyramid of an image</li>
    <li>black - a laplacian pyramid of another imge.</li>
    <li>mask - a gaussian pyramid of a mask image.</li>
</ul>
<p>It should perform an alpha-blend of the two Laplacian pyramids according to the mask pyramid. So you will be blending each pair of layers together using the mask of that layer as the weight.</p>
<p>As described in the doc string, pixels where the mask is 1 should be taken from the white image, pixels where the mask is 0 should be taken from the black image. Pixels with value 0.5 in the mask should be an equal blend of the white and black images. This is mathematically described in the assignment comments.</p>
<p>You may assume that all of the provided pyramids are of the same dimensons, and have dtype float. You may further assume that the mask pyramid has values in the range [0,1].</p>
<p>Your output pyramid should be of the same dimensions as all the inputs, and dtype float.</p>
<h3 style="color: black;">collapse</h3>
<p>This function is given a laplacian pyramid and is expected to 'flatten' it to an image.</p>
<p>We need to take the bottom (smallest) layer, expand it, and then add it to the next layer. We continue this process until we reach the top of the pyramid. Once you add the second layer to the top layer, the top layer is your output (a common mistake is to re-add the top layer to the top layer so if your values look like they are twice as big as what the expected output is, that is what you are doing).</p>
<p>This function should return a numpy array of the same shape as the top (largest) layer of the pyramid, and dtype float.</p>
</div>
<div id="pdf_details">
<h2 style="color: black; font-size: 130%; text-decoration: underline">The Writeup</h2>
<p>This is what we want you to do for the PDF.</p>
<ol>
    <li>Choose a black image and a white image that you would like to blend.</li>
    <li>Choose a unique mask (don't use the one we provide) that you created and explain how you created it.</li>
    <li>Demonstrate these three images in the PDF.</li>
    <li>Explain what you did to tackle each function. Note: For the expand function, we also want you to explicitly state why you have to scale your output by 4.</li>
    <li>Anything else you wish to note, feel free to include the pyramid outputs if you found them interesting, etc (for Peer Feedback).</li>
</ol>
</div>
<div id="turnin">
<h2 style="color: black; font-size: 130%; text-decoration: underline">What to turn in:</h2>
<ul>
    <li>assignment6.py - Your code.</li>
    <li>assignment6.pdf - See above for the writeup, don't forget to include your images!</li>
</ul>
</div>
</div>