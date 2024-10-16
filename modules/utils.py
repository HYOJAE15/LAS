import os

os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = pow(2,40).__str__()

import cv2

import numpy as np

from numba import njit

from PySide6.QtGui import QImage

import slidingwindow as sw 
import math 

import csv


def imread(path: str, checkImg: bool=True) -> np.array:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    if checkImg :
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    
    return img

def imwrite(path:str,
            img: np.array
            )-> None: 
    _, ext = os.path.splitext(path)

    if img.ndim == 3 : 
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 

    _, label_to_file = cv2.imencode(ext, img)
    label_to_file.tofile(path)

def imwrite_colormap(path, img): 
    _, ext = os.path.splitext(path)
    _, label_to_file = cv2.imencode(ext, img)
    label_to_file.tofile(path)


def createLayersFromLabel(label: np.array, 
                          num_class: int
                          ) -> list([np.array]):
    layers = []

    for idx in range(num_class):
        layers.append(label == idx)
        
    return layers


def cvtArrayToQImage(array: np.array) -> QImage:

    if len(array.shape) == 3 : 

        h, w, c = array.shape
        if c == 3:
            return QImage(array.data, w, h, 3 * w, QImage.Format_RGB888)
        elif c == 4: 
            return QImage(array.data, w, h, 4 * w, QImage.Format_RGBA8888)

    elif len(array.shape) == 2 :
        h, w = array.shape
        return QImage(array.data, w, h, QImage.Format_Mono)
    
def cvtPixmapToArray(pixmap):
    """Convert a QPixmap to a numpy array
    Args: 
        pixmap (QPixmap): The QPixmap to convert
    
    Returns:
        img (np.array): The converted QPixmap
    """
    
    ## Get the size of the current pixmap
    size = pixmap.size()
    h = size.width()
    w = size.height()

    ## Get the QImage Item and convert it to a byte string
    qimg = pixmap.toImage()
    byte_str = qimg.bits().tobytes()

    ## Using the np.frombuffer function to convert the byte string into an np array
    img = np.frombuffer(byte_str, dtype=np.uint8).reshape((w,h,4))

    return img

@njit
def mapLabelToColorMap(label: np.array, 
                       colormap: np.array, 
                       palette: list(list([int, int, int]))
                       )-> np.array:
    
    assert label.ndim == 2, "label must be 2D array"
    assert colormap.ndim == 3, "colormap must be 3D array"
    assert colormap.shape[2] == 4, "colormap must have 4 channels"

    for x in range(label.shape[0]):
        for y in range(label.shape[1]):
            colormap[x, y, :3] = palette[label[x, y]][:3]

    return colormap
   

def convertLabelToColorMap(
        label: np.array,
        palette: list(list([int, int, int])),
        alpha: int) -> np.array:
    assert label.ndim == 2, "label must be 2D array"
    assert alpha >= 0 and alpha <= 255, "alpha must be between 0 and 255"

    colormap = np.zeros((label.shape[0], label.shape[1], 4), dtype=np.uint8)
    colormap = mapLabelToColorMap(label, colormap, palette)
    colormap[:, :, 3] = alpha

    return colormap


def generateForNumberOfWindows(data, dimOrder, windowCount, overlapPercent, transforms=[]):
	"""
	Generates a set of sliding windows for the specified dataset, automatically determining the required window size in
	order to create the specified number of windows. `windowCount` must be a tuple specifying the desired number of windows
	along the Y and X axes, in the form (countY, countX).
	"""
	
	# Determine the dimensions of the input data
	width = data.shape[dimOrder.index('w')]
	height = data.shape[dimOrder.index('h')]
	
	# Determine the window size required to most closely match the desired window count along both axes
	countY, countX = windowCount
	windowSizeX = math.ceil(width / countX)
	windowSizeY = math.ceil(height / countY)
	
	# Generate the windows
	return sw.generateForSize(
		width,
		height,
		dimOrder,
		0,
		overlapPercent,
		transforms,
		overrideWidth = windowSizeX,
		overrideHeight = windowSizeY
	)

def blendImageWithColorMap(
        image, 
        label, 
        palette = np.array([[0, 0, 0],
                            [0, 0, 255],
                            [0, 255, 0],
                            [0, 255, 255],
                            [255, 0, 0]]), 
        alpha = 0.5
        ):
    """ blend image with color map 
    Args: 
        image (3d np.array): RGB image
        label (2d np.array): 1 channel gray-scale image
        pallete (2d np.array) 
        alpha (float)

    Returns: 
        color_map (3d np.array): RGB image
    """

    color_map = np.zeros_like(image)
        
    for idx, color in enumerate(palette) : 
        
        if idx == 0 :
            color_map[label == idx, :] = image[label == idx, :] * 1
        else :
            color_map[label == idx, :] = image[label == idx, :] * alpha + color * (1-alpha)

    return color_map
