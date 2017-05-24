import sys
import os
import numpy as np
import cv2
from scipy.stats import norm
from scipy.signal import convolve2d
import math

import assignment6 as assignment6

def viz_gauss_pyramid(pyramid):
  """ This function creates a single image out of the given pyramid.
  """
  height = pyramid[0].shape[0]
  width = pyramid[0].shape[1]

  out = np.zeros((height*len(pyramid), width), dtype = float)

  for idx, layer in enumerate(pyramid):
    if layer.max() <= 1:
      layer = layer.copy() * 255

    out[(idx*height):((idx+1)*height),:] = cv2.resize(layer, (width, height), 
        interpolation = 3)

  return out.astype(np.uint8)

def viz_lapl_pyramid(pyramid):
  """ This function creates a single image out of the given pyramid.
  """
  height = pyramid[0].shape[0]
  width = pyramid[0].shape[1]

  out = np.zeros((height*len(pyramid), width), dtype = np.uint8)

  for idx, layer in enumerate(pyramid[:-1]):
     # We use 3 for interpolation which is cv2.INTER_AREA. Using a value is
     # safer for compatibility issues in different versions of OpenCV.
     patch = cv2.resize(layer, (width, height),
         interpolation = 3).astype(float)
     # scale patch to 0:256 range.
     patch = 128 + 127*patch/(np.abs(patch).max())

     out[(idx*height):((idx+1)*height),:] = patch.astype(np.uint8)

  #special case for the last layer, which is simply the remaining image.
  patch = cv2.resize(pyramid[-1], (width, height), 
      interpolation = 3)
  out[((len(pyramid)-1)*height):(len(pyramid)*height),:] = patch

  return out

def run_blend(black_image, white_image, mask):
  """ This function administrates the blending of the two images according to 
  mask.

  Assume all images are float dtype, and return a float dtype.
  """

  # Automatically figure out the size
  min_size = min(black_image.shape)
  depth = int(math.floor(math.log(min_size, 2))) - 4 # at least 16x16 at the highest level.

  gauss_pyr_mask = assignment6.gaussPyramid(mask, depth)
  gauss_pyr_black = assignment6.gaussPyramid(black_image, depth)
  gauss_pyr_white = assignment6.gaussPyramid(white_image, depth)


  lapl_pyr_black  = assignment6.laplPyramid(gauss_pyr_black)
  lapl_pyr_white = assignment6.laplPyramid(gauss_pyr_white)

  outpyr = assignment6.blend(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask)
  outimg = assignment6.collapse(outpyr)

  outimg[outimg < 0] = 0 # blending sometimes results in slightly out of bound numbers.
  outimg[outimg > 255] = 255
  outimg = outimg.astype(np.uint8)

  return lapl_pyr_black, lapl_pyr_white, gauss_pyr_black, gauss_pyr_white, \
      gauss_pyr_mask, outpyr, outimg

def test_reduce_expand():
  """This script will perform a unit test on the first two functions.
  """
  # Each subsequent layer is a reduction of the previous one
  reduce1 =[np.array([[   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.],
                      [   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.],
                      [   0.,    0.,  255.,  255.,  255.,  255.,    0.,    0.],
                      [   0.,    0.,  255.,  255.,  255.,  255.,    0.,    0.],
                      [   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.],
                      [   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.]]),
            np.array([[   0.64,    8.92,   12.11,    3.82],
                      [   8.29,  116.03,  157.46,   49.73],
                      [   3.82,   53.55,   72.67,   22.95]]),
            np.array([[ 12.21,  31.85],
                      [ 17.62,  45.97]]),
            np.array([[ 9.77]])] 

  reduce2 = [np.array([[ 255.,  255.,  255.,  255.,  255.,  255.,  255.],
                       [ 255.,  255.,  255.,  255.,  255.,  255.,  255.],
                       [ 255.,  255.,  125.,  125.,  125.,  255.,  255.],
                       [ 255.,  255.,  125.,  125.,  125.,  255.,  255.],
                       [   0.,    0.,    0.,    0.,    0.,    0.,    0.]]),
             np.array([[ 124.62,  173.95,  173.95,  124.62],
                       [ 165.35,  183.1 ,  183.1 ,  165.35],
                       [  51.6 ,   49.2 ,   49.2 ,   51.6 ]]),
             np.array([[  72.85,  104.71],
                       [  49.53,   68.66]]),
             np.array([[ 31.37]])] 

  if __name__ == "__main__":
    print 'Evaluating reduce.'
  for red_pyr in reduce1, reduce2:
    for imgin, true_out in zip(red_pyr[0:-1], red_pyr[1:]):
      if __name__ == "__main__":
        print "input:\n{}\n".format(imgin)

      usr_out = assignment6.reduce(imgin)

      if not type(usr_out) == type(true_out):
        if __name__ == "__main__":
          print "Error - reduce out has type {}. Expected type is {}.".format(
              type(usr_out), type(true_out))
        return False

      if not usr_out.shape == true_out.shape:
        if __name__ == "__main__":
          print "Error - reduce out has shape {}. Expected shape is {}.".format(
              usr_out.shape, true_out.shape)
        return False

      if not usr_out.dtype == true_out.dtype:
        if __name__ == "__main__":
          print "Error- reduce out has dtype {}. Expected dtype is {}.".format(
              usr_out.dtype, true_out.dtype)
        return False

      if not np.all(np.abs(usr_out - true_out) < 1):
        if __name__ == "__main__":
          print "Error- reduce out has value:\n{}\nExpected value:\n{}".format(
              usr_out, true_out)
        return False

  if __name__ == "__main__":
    print "reduce passed.\n"
    print "Evaluating expand."

  expandin = [np.array([[255]]),
              np.array([[125, 255],
                        [255,   0]]),
              np.array([[ 255.,    0.,  125.,  125.,  125.],
                        [ 255.,    0.,  125.,  125.,  125.],
                        [  50.,   50.,   50.,   50.,   50.]])] 

  expandout =[np.array([[ 163.2 ,  102.  ],
                        [ 102.  ,   63.75]]),
              np.array([[ 120.8 ,  164.75,  175.75,  102.  ],
                        [ 164.75,  158.75,  121.  ,   63.75],
                        [ 175.75,  121.  ,   42.05,   12.75],
                        [ 102.  ,   63.75,   12.75,    0.  ]]),
              np.array([[ 183.6, 114.75, 34.2, 56.25, 101.25, 112.5, 112.5,112.5, 101.25,  56.25],
                        [ 204. ,  127.5,  38.,  62.5,  112.5,  125.,  125., 125.,  112.5,  62.5 ],
                        [ 188.1, 119.75, 39.2, 61.25, 106.25, 117.5, 117.5,117.5, 105.75,  58.75],
                        [ 124.5,  88.75, 44. , 56.25,  81.25,  87.5,  87.5, 87.5,  78.75,  43.75],
                        [  56.4,  52.75, 43.8, 46.25,  51.25,  52.5,  52.5, 52.5,  47.25,  26.25],
                        [  22.5,    25.,  25.,   25.,    25.,   25.,   25.,  25.,   22.5,  12.5 ]])]

  for imgin, true_out in zip(expandin, expandout):
    if __name__ == "__main__":
      print "input:\n{}\n".format(imgin)

    usr_out = assignment6.expand(imgin)

    if not type(usr_out) == type(true_out):
      if __name__ == "__main__":
        print "Error - expand out has type {}. Expected type is {}.".format(
            type(usr_out), type(true_out))
      return False

    if not usr_out.shape == true_out.shape:
      if __name__ == "__main__":
        print "Error - expand out has shape {}. Expected shape is {}.".format(
            usr_out.shape, true_out.shape)
      return False

    if not usr_out.dtype == true_out.dtype:
      if __name__ == "__main__":
        print "Error - expand out has dtype {}. Expected dtype is {}.".format(
            usr_out.dtype, true_out.dtype)
      return False

    if not np.all(np.abs(usr_out - true_out) < 1):
      if __name__ == "__main__":
        print "Error - expand out has value:\n{}\nExpected value:\n{}".format(
            usr_out, true_out)
      return False

  if __name__ == "__main__":
    print "expand passed."

  if __name__ == "__main__":
    print "Tests for reduce and expand successful."
  return True

def test_gaussian_laplacian():
  """ This script will perform a unit test on your Gaussian and Laplacian
      pyramid functions.
  """
  gauss_pyr1 =[np.array([[   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.],
                         [   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.],
                         [   0.,    0.,  255.,  255.,  255.,  255.,    0.,    0.],
                         [   0.,    0.,  255.,  255.,  255.,  255.,    0.,    0.],
                         [   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.],
                         [   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.]]),
               np.array([[   0.64,    8.92,   12.11,    3.82],
                         [   8.29,  116.03,  157.46,   49.73],
                         [   3.82,   53.55,   72.67,   22.95]]),
               np.array([[ 12.21,  31.85],
                         [ 17.62,  45.97]]),
               np.array([[ 9.77]])] 

  gauss_pyr2 = [np.array([[ 255.,  255.,  255.,  255.,  255.,  255.,  255.],
                          [ 255.,  255.,  255.,  255.,  255.,  255.,  255.],
                          [ 255.,  255.,  125.,  125.,  125.,  255.,  255.],
                          [ 255.,  255.,  125.,  125.,  125.,  255.,  255.],
                          [   0.,    0.,    0.,    0.,    0.,    0.,    0.]]),
                np.array([[ 124.62,  173.95,  173.95,  124.62],
                          [ 165.35,  183.1 ,  183.1 ,  165.35],
                          [  51.6 ,   49.2 ,   49.2 ,   51.6 ]]),
                np.array([[  72.85,  104.71],
                          [  49.53,   68.66]]),
                np.array([[ 31.37]])] 

  if __name__ == "__main__":
    print 'Evaluating gaussPyramid.'

  for pyr in gauss_pyr1, gauss_pyr2:
    if __name__ == "__main__":
      print "input:\n{}\n".format(pyr[0])

    usr_out = assignment6.gaussPyramid(pyr[0], 3)

    if not type(usr_out) == type(pyr):
      if __name__ == "__main__":
        print "Error- gaussPyramid out has type {}. Expected type is {}.".format(
            type(usr_out), type(pyr))
      return False

    if not len(usr_out) == len(pyr):
      if __name__ == "__main__":
        print "Error- gaussPyramid out has len {}. Expected len is {}.".format(
            len(usr_out), len(pyr))
      return False

    for usr_layer, true_layer in zip(usr_out, pyr):
      if not type(usr_layer) == type(true_layer):
        if __name__ == "__main__":
          print "Error- output layer has type {}. Expected type is {}.".format(
              type(usr_layer), type(true_layer))
        return False

      if not usr_layer.shape == true_layer.shape:
        if __name__ == "__main__":
          print "Error- gaussPyramid layer has shape {}. Expected shape is {}.".format(
              usr_layer.shape, true_layer.shape)
        return False

      if not usr_layer.dtype == true_layer.dtype:
        if __name__ == "__main__":
          print "Error- gaussPyramid layer has dtype {}. Expected dtype is {}.".format(
              usr_layer.dtype, true_layer.dtype)
        return False

      if not np.all(np.abs(usr_layer - true_layer) < 1):
        if __name__ == "__main__":
          print "Error- gaussPyramid layer has value:\n{}\nExpected value:\n{}".format(
              usr_layer, true_layer)
        return False

  if __name__ == "__main__":
    print "gaussPyramid passed.\n"
    print "Evaluating laplPyramid."

  lapl_pyr1 =[np.array([[  -2.95,  -10.04,  -17.67,  -22.09,  -23.02,  -16.73,   -8.97,   -4.01],
                        [  -9.82,  -33.47,  -58.9 ,  -73.63,  -76.75,  -55.78,  -29.9 ,  -13.39],
                        [ -15.57,  -53.07,  161.59,  138.24,  133.29,  166.55,  -47.41,  -21.23],
                        [ -13.32,  -45.42,  175.06,  155.07,  150.83,  179.3 ,  -40.58,  -18.17],
                        [  -8.55,  -29.16,  -51.33,  -64.16,  -66.88,  -48.61,  -26.05,  -11.67],
                        [  -4.21,  -14.34,  -25.24,  -31.55,  -32.89,  -23.91,  -12.81,   -5.74]]),
              np.array([[ -11.59,  -11.88,  -13.1 ,  -11.22],
                        [  -7.53,   89.12,  124.84,   30.27],
                        [ -12.43,   25.91,   39.17,    2.97]]),
              np.array([[  5.96,  27.94],
                        [ 13.71,  43.53]]),
              np.array([[ 9.77]])] 

  lapl_pyr2 =[np.array([[ 146.27,  118.15,  101.65,   97.53,  101.65,  118.15,  146.27],
                        [ 121.16,   93.25,   79.83,   76.48,   79.83,   93.25,  121.16],
                        [ 118.2 ,   95.65,  -41.91,  -43.79,  -41.91,   95.65,  118.2 ],
                        [ 156.61,  142.69,    9.62,    8.85,    9.62,  142.69,  156.6 ],
                        [ -52.02,  -57.74,  -57.68,  -57.67,  -57.68,  -57.74,  -52.02]]),
              np.array([[  64.97,   97.02,   95.12,   79.3 ],
                        [ 107.73,  109.16,  107.63,  122.01],
                        [   7.53,   -6.95,   -7.81,   18.9 ]]),
              np.array([[ 52.77,  92.16],
                        [ 36.98,  60.82]]),
              np.array([[ 31.37]])] 

  for gauss_pyr, lapl_pyr in zip((gauss_pyr1, gauss_pyr2), (lapl_pyr1, lapl_pyr2)):
    if __name__ == "__main__":
      print "input:\n{}".format(gauss_pyr)

    usr_out = assignment6.laplPyramid(gauss_pyr)

    if not type(usr_out) == type(lapl_pyr):
      if __name__ == "__main__":
        print "Error- laplPyramid out has type {}. Expected type is {}.".format(
            type(usr_out), type(lapl_pyr))
      return False

    if not len(usr_out) == len(lapl_pyr):
      if __name__ == "__main__":
        print "Error- laplPyramid out has len {}. Expected len is {}.".format(
            len(usr_out), len(lapl_pyr))
      return False

    for usr_layer, true_layer in zip(usr_out, lapl_pyr):
      if not type(usr_layer) == type(true_layer):
        if __name__ == "__main__":
          print "Error- output layer has type {}. Expected type is {}.".format(
              type(usr_layer), type(true_layer))
        return False

      if not usr_layer.shape == true_layer.shape:
        if __name__ == "__main__":
          print "Error- laplPyramid layer has shape {}. Expected shape is {}.".format(
              usr_layer.shape, true_layer.shape)
        return False

      if not usr_layer.dtype == true_layer.dtype:
        if __name__ == "__main__":
          print "Error- laplPyramid layer has dtype {}. Expected dtype is {}.".format(
              usr_layer.dtype, true_layer.dtype)
        return False

      if not np.all(np.abs(usr_layer - true_layer) < 1):
        if __name__ == "__main__":
          print "Error- laplPyramid layer has value:\n{}\nExpected value:\n{}".format(
              usr_layer, true_layer)
        return False

  if __name__ == "__main__":
    print "laplPyramid passed."

  if __name__ == "__main__":
    print "Tests for Gaussian and Laplacian Pyramid successful."
  return True

def test_blend_collapse():
  """ This script will perform a unit test on your blend and collapse functions
  and output any errors for debugging.
  """
  lapl_pyr11 =[np.array([[ 0.,  0.,  0.,  0.],
                          [ 0.,  0.,  0.,  0.],
                          [ 0.,  0.,  0.,  0.]]),
                np.array([[ 0.,  0.],
                          [ 0.,  0.]])] 
  lapl_pyr12 =[np.array([[ 149.77,  122.46,  121.66,  178.69],
                          [ 138.08,  107.74,  106.84,  170.21],
                          [ 149.77,  122.46,  121.66,  178.69]]),
                np.array([[ 124.95,  169.58],
                          [ 124.95,  169.57]])] 
  lapl_pyr21 =[np.array([[ 149. ,  118.4,   99.2,   94.3,   99.2,  118.4,  149. ],
                          [ 137.2,  103.3,   81.9,   76.5,   81.9,  103.3,  137.2],
                          [ 148.1,  117.4,   97.9,   93.1,   97.9,  117.4,  148.1],
                          [ -63.1,  -81.3,  -92.8,  -95.6,  -92.8,  -81.3,  -63.1],
                          [ -18.5,  -23.8,  -27.2,  -28. ,  -27.2,  -23.8,  -18.5]]),
                np.array([[  70.4,  107.1,  104.5,   82.3],
                          [  76.7,  115.4,  113.1,   87.3],
                          [ -23.3,  -29.4,  -31. ,  -16.3]]),
                np.array([[  67.7,  100.3],
                          [  34. ,   50.4]])] 
  lapl_pyr22 =[np.array([[  -5. ,  -25.2,  -56.4,  149.8,  110.3,  116.2,  144.8],
                          [  -6.5,  -32.5,  -72.6,  119.5,   68.6,   76.2,  113. ],
                          [  -7.2,  -36. ,  -80.3,  105.2,   48.9,   57.2,   98. ],
                          [  -6.5,  -32.5,  -72.6,  119.5,   68.6,   76.2,  113. ],
                          [  -5. ,  -25.2,  -56.4,  149.8,  110.3,  116.2,  144.8]]),
                np.array([[ -20.9,    4.8,  102.6,   84.1],
                          [ -23.2,   22.3,  167.9,  133.1],
                          [ -20.9,    4.8,  102.6,   84.1]]),
                np.array([[ 17.6,  90.8],
                          [ 17.6,  90.8]])] 
  mask_pyr1 =[np.array([[ 0.,  0.,  1.,  1.],
                        [ 0.,  0.,  1.,  1.],
                        [ 0.,  0.,  1.,  1.]]),
              np.array([[ 0.03,  0.46],
                        [ 0.03,  0.46]])] 
  mask_pyr2 = [np.array([[ 0.,  0.,  0.,  0.,  1.,  1.,  1.],
                         [ 0.,  0.,  0.,  0.,  1.,  1.,  1.],
                         [ 0.,  0.,  0.,  0.,  1.,  1.,  1.],
                         [ 0.,  0.,  0.,  0.,  1.,  1.,  1.],
                         [ 0.,  0.,  0.,  0.,  1.,  1.,  1.]]),
               np.array([[ 0. ,  0. ,  0.5,  0.5],
                         [ 0. ,  0. ,  0.7,  0.7],
                         [ 0. ,  0. ,  0.5,  0.5]]),
               np.array([[ 0. ,  0.3],
                         [ 0. ,  0.3]])] 
  out_pyr1 =[np.array([[ 149.77,  122.46,    0.  ,    0.  ],
                       [ 138.08,  107.74,    0.  ,    0.  ],
                       [ 149.77,  122.46,    0.  ,    0.  ]]),
             np.array([[ 120.58,   92.42],
                       [ 120.58,   92.42]])] 
  out_pyr2 = [np.array([[  -5. ,  -25.2,  -56.4,  149.8,   99.2,  118.4,  149. ],
                        [  -6.5,  -32.5,  -72.6,  119.5,   81.9,  103.3,  137.2],
                        [  -7.2,  -36. ,  -80.3,  105.2,   97.9,  117.4,  148.1],
                        [  -6.5,  -32.5,  -72.6,  119.5,  -92.8,  -81.3,  -63.1],
                        [  -5. ,  -25.2,  -56.4,  149.8,  -27.2,  -23.8,  -18.5]]),
              np.array([[ -20.9,    4.8,  103.5,   83.2],
                        [ -23.2,   22.3,  129.5,  101. ],
                        [ -20.9,    4.8,   35.8,   33.9]]),
              np.array([[ 17.6,  93.6],
                        [ 17.6,  78.7]])] 
  outimg1 = np.array([[ 244.91,  218.31,   77.39,   41.59],
                      [ 243.79,  214.24,   85.99,   46.21],
                      [ 244.91,  218.31,   77.39,   41.59]]) 
  outimg2 = np.array([[   0.1,    0.1,   -0.1,  253.7,  241.3,  254. ,  256. ],
                      [  -0.3,   -0.5,   -2.7,  244.4,  250.3,  263.3,  263.2],
                      [  -0.6,   -1.4,   -6. ,  233.4,  267.8,  278.2,  274.6],
                      [  -0.9,   -2.1,   -8.7,  224.1,   42.2,   46.1,   37.3],
                      [  -1. ,   -2.4,   -9.6,  221.2,   61.5,   59.5,   47.5]])

  if __name__ == "__main__":
    print 'Evaluating blend.'

  for left_pyr, right_pyr, mask_pyr, out_pyr in ((lapl_pyr11, lapl_pyr12, mask_pyr1, out_pyr1), 
      (lapl_pyr21, lapl_pyr22, mask_pyr2, out_pyr2)):
    usr_out = assignment6.blend(left_pyr, right_pyr, mask_pyr)

    if not type(usr_out) == type(out_pyr):
      if __name__ == "__main__":
        print "Error- output layer has type {}. Expected type is {}.".format(
            type(usr_out), type(out_pyr))
      return False

    if not len(usr_out) == len(out_pyr):
      if __name__ == "__main__":
        print "Error- blend out has len {}. Expected len is {}.".format(
            len(usr_out), len(out_pyr))
      return False

    for usr_layer, true_layer, left_layer, right_layer, mask_layer in zip(usr_out, out_pyr, 
        left_pyr, right_pyr, mask_pyr):
      if not type(usr_layer) == type(true_layer):
        if __name__ == "__main__":
          print "Error- blend out has type {}. Expected type is {}.".format(
              type(usr_layer), type(true_layer))
        return False

      if not usr_layer.shape == true_layer.shape:
        if __name__ == "__main__":
          print "Error- blend output layer has shape {}. Expected shape is {}.".format(
              usr_layer.shape, true_layer.shape)
        return False

      if not usr_layer.dtype == true_layer.dtype:
        if __name__ == "__main__":
          print "Error- blend output layer has dtype {}. Expected dtype is {}.".format(
              usr_layer.dtype, true_layer.dtype)
        return False

      if not np.all(np.abs(usr_layer - true_layer) < 1):
        if __name__ == "__main__":
         print "Error- blend output layer has value:\n{}\nExpected value:\n{}\nInput left:\n{}\nInput right:\n{}\nInput mask:\n{}".format(
              usr_layer, true_layer, left_layer, right_layer, mask_layer)
        return False

  if __name__ == "__main__":
    print "blend passed.\n"
    print "Evaluating collapse."

  for pyr, img in ((out_pyr1, outimg1),(out_pyr2, outimg2)):
    if __name__ == "__main__":
      print "input:\n{}".format(pyr)

    usr_out = assignment6.collapse(pyr)

    if not type(usr_out) == type(img):
      if __name__ == "__main__":
        print "Error- collapse out has type {}. Expected type is {}.".format(
            type(usr_out), type(img))
      return False

    if not usr_out.shape == img.shape:
      if __name__ == "__main__":
        print "Error- collapse out has shape {}. Expected shape is {}.".format(
            usr_out.shape, img.shape)
      return False

    if not usr_out.dtype == img.dtype:
      if __name__ == "__main__":
        print "Error- collapse out has dtype {}. Expected dtype is {}.".format(
            usr_out.dtype, img.dtype)
      return False

    if not np.all(np.abs(usr_out - img) < 1):
      if __name__ == "__main__":
        print "Error- collapse out has value:\n{}\nExpected value:\n{}".format(
            usr_out, img)
      return False

  if __name__ == "__main__":
    print "collapse passed."

  if __name__ == "__main__":
    print "All unit tests successful."
  return True

if __name__ == "__main__":
  print 'Performing unit tests.'
  if not test_reduce_expand():
    print 'Reduce or Expand functions failed. Halting testing.'
    sys.exit()

  if not test_gaussian_laplacian():
    print 'Gaussian or Laplacian functions failed. Halting testing.'
    sys.exit()

  if not test_blend_collapse():
    print 'Blend or Collapse functions failed. Halting testing.'
    sys.exit()

  print 'Unit tests passed.'
  sourcefolder = os.path.abspath(os.path.join(os.curdir, 'images', 'source'))
  outfolder = os.path.abspath(os.path.join(os.curdir, 'images', 'output'))

  print 'Searching for images in {} folder'.format(sourcefolder)

  # Extensions recognized by opencv
  exts = ['.bmp', '.pbm', '.pgm', '.ppm', '.sr', '.ras', '.jpeg', '.jpg', 
    '.jpe', '.jp2', '.tiff', '.tif', '.png']

  # For every image in the source directory
  for dirname, dirnames, filenames in os.walk(sourcefolder):
    setname = os.path.split(dirname)[1]

    white_img = None
    black_img = None
    mask_img = None

    for filename in filenames:
      name, ext = os.path.splitext(filename)
      if ext in exts:
        if 'black' in name:
          print "Reading image {} from {}.".format(filename, dirname)
          black_img = cv2.imread(os.path.join(dirname, filename))

        if 'white' in name:
          print "Reading image {} from {}.".format(filename, dirname)
          white_img = cv2.imread(os.path.join(dirname, filename))

        if 'mask' in name:
          print "Reading image {} from {}.".format(filename, dirname)
          mask_img = cv2.imread(os.path.join(dirname, filename))

    if white_img == None or black_img == None or mask_img == None:
      print "Did not find black/white/mask images in folder: " + dirname
      continue

    assert black_img.shape == white_img.shape and black_img.shape == mask_img.shape, \
        "Error - the sizes of images and the mask are not equal"

    black_img = black_img.astype(float)
    white_img = white_img.astype(float)
    mask_img = mask_img.astype(float) / 255

    print "Applying blending."
    lapl_pyr_black_layers = []
    lapl_pyr_white_layers = []
    gauss_pyr_black_layers = []
    gauss_pyr_white_layers = []
    gauss_pyr_mask_layers = []
    out_pyr_layers = []
    out_layers = []

    for channel in range(3):
      lapl_pyr_black, lapl_pyr_white, gauss_pyr_black, gauss_pyr_white, gauss_pyr_mask,\
          outpyr, outimg = run_blend(black_img[:,:,channel], white_img[:,:,channel], \
                           mask_img[:,:,channel])
      
      lapl_pyr_black_layers.append(viz_lapl_pyramid(lapl_pyr_black))
      lapl_pyr_white_layers.append(viz_lapl_pyramid(lapl_pyr_white))
      gauss_pyr_black_layers.append(viz_gauss_pyramid(gauss_pyr_black))
      gauss_pyr_white_layers.append(viz_gauss_pyramid(gauss_pyr_white))
      gauss_pyr_mask_layers.append(viz_gauss_pyramid(gauss_pyr_mask))
      out_pyr_layers.append(viz_lapl_pyramid(outpyr))
      out_layers.append(outimg)
    
    lapl_pyr_black_img = cv2.merge(lapl_pyr_black_layers)
    lapl_pyr_white_img = cv2.merge(lapl_pyr_white_layers)
    gauss_pyr_black_img = cv2.merge(gauss_pyr_black_layers)
    gauss_pyr_white_img = cv2.merge(gauss_pyr_white_layers)
    gauss_pyr_mask_img = cv2.merge(gauss_pyr_mask_layers)
    outpyr = cv2.merge(out_pyr_layers)
    outimg = cv2.merge(out_layers)

    print "Writing images to folder {}".format(os.path.join(outfolder, setname))
    cv2.imwrite(os.path.join(outfolder, setname + '_lapl_pyr_black' + ext),
                lapl_pyr_black_img)
    cv2.imwrite(os.path.join(outfolder, setname + '_lapl_pyr_white' + ext),
                lapl_pyr_white_img)
    cv2.imwrite(os.path.join(outfolder, setname + '_gauss_pyr_black' + ext),
                gauss_pyr_black_img)
    cv2.imwrite(os.path.join(outfolder, setname + '_gauss_pyr_white' + ext),
                gauss_pyr_white_img)
    cv2.imwrite(os.path.join(outfolder, setname + '_gauss_pyr_mask' + ext),
                gauss_pyr_mask_img)
    cv2.imwrite(os.path.join(outfolder, setname + '_outpyr' + ext),
                outpyr)
    cv2.imwrite(os.path.join(outfolder, setname + '_outimg' + ext),
                outimg)
