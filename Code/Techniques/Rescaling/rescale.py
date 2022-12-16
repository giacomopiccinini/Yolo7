import numpy as np
import cv2


def letterbox(
    image,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):

    """Create a letterbox format of the original image. That is, the original image will first be reshaped so
    that its longer dimension matches the one specified in new_shape. For the smaller dimension, we do not rescale it completely,
    for this would mean deforming the image resulting in possibly incorrect detection. Hence, we rescale it and then add some padding
    (usually in grey)"""

    # Get current shape of image, neglect number of channels (not relevant for rescaling)
    shape = image.shape[:2]

    # Find minimum ratio between new and old dimensions (important if image is not square)
    # The minimum ratio will determine the scale-up necessary on the bigger dimension to match the desired
    minimum_ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Only scale down, do not scale up (for better test mAP)
    if not scaleup:
        minimum_ratio = min(minimum_ratio, 1.0)

    # Ratio tuple, i.e. along width and height
    ratio = minimum_ratio, minimum_ratio

    # Get the unpadded shape of the image, i.e. rescale both dimensions
    # with the same ratio and do nothing else
    unpadded_shape = int(round(shape[0] * minimum_ratio)), int(
        round(shape[1] * minimum_ratio)
    )

    # Compute the difference between the desired dimensions and the actual dimensions
    # Difference in height
    dh = new_shape[0] - unpadded_shape[0]

    # Difference in width
    dw = new_shape[1] - unpadded_shape[1]

    # Minimum rectangle (i.e. do not add unnecessary padding that will result in nothing, but make sure
    # that the new dimension can be evenly divided with the stride of our algorithm)
    if auto:
        dw = np.mod(dw, stride)
        dh = np.mod(dh, stride)
    # Stretch
    elif scaleFill:
        dw, dh = 0.0, 0.0
        unpadded_shape = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    # Divide padding into the 2 sides
    dw /= 2
    dh /= 2

    # If image is not already of the desired dimension,
    # first resize it
    if shape != unpadded_shape:
        image = cv2.resize(image, unpadded_shape[::-1], interpolation=cv2.INTER_LINEAR)

    # Compute padding dimension to be added to the image
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    # Add padding (i.e. border) of color
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    return image, ratio, (dw, dh)



def rescale_coordinates(small_shape, original_shape, coordinates, ratio_pad=None):
    
    """ Rescale box coordinates from the small standard shape of YOLO
    to the original """
    
    # Rescale coordinates
    if ratio_pad is None:  
        # Compute gain
        gain = min(original_shape[0] / small_shape[0], original_shape[1] / small_shape[1]) 
        
        # Compute pad
        pad = (original_shape[1] - small_shape[1] * gain) / 2, (original_shape[0] - small_shape[0] * gain) / 2  
        
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    # Remove padding
    coordinates[:, [0, 2]] -= pad[0] 
    coordinates[:, [1, 3]] -= pad[1] 
    
    # Rescale
    coordinates[:, :4] /= gain
    
    clip_coords(coordinates, small_shape)
    
    return coordinates


def clip_coords(boxes, image_shape):
    
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, image_shape[1])  # x1
    boxes[:, 1].clamp_(0, image_shape[0])  # y1
    boxes[:, 2].clamp_(0, image_shape[1])  # x2
    boxes[:, 3].clamp_(0, image_shape[0])  # y2