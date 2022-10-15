import torch
import torchvision
import numpy as np

def non_max_suppression(prediction, confidence_threshold=0.25, iou_threshold=0.45, classes=None, agnostic=False, multi_label=False):
    """
    Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    # Retrieve number of classes from prediction
    number_of_classes = prediction.shape[2] - 5 

    # Candidates that surpass a confidence threshold
    candidates = prediction[..., 4] > confidence_threshold

    # Settings
    max_box_width_height  = 4096 # Maximum box width and height in pixels

    max_detections = 300  # Maximum number of detections per image

    max_nms = 30000  # Maximum number of boxes into torchvision.ops.nms()

    # Ensure multi labelling is indeed possible
    multi_label &= number_of_classes > 1 

    # Initialise output (to be filled)
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]

    for index, image in enumerate(prediction): 

        image = image[candidates[index]]  # confidence

        # If none remain process next image
        if not image.shape[0]:
            continue

        # Compute confidence 
        if number_of_classes == 1:
            image[:, 5:] = image[:, 4:5] # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
                                 # so there is no need to multiplicate.
        else:
            image[:, 5:] *= image[:, 4:5]  # conf = obj_conf * cls_conf

        # Create box converting from (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(image[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > confidence_threshold).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check boxes number
        n_boxes = image.shape[0]  

        # If no boxes, continue
        if not n_boxes: 
            continue
        # Else, if maximum number of boxes is achieved 
        elif n_boxes > max_nms:  
            # Sort boxes by confidence, and take the first max_nms
            image = image[image[:, 4].argsort(descending=True)[:max_nms]]

        # Get classes
        c = x[:, 5:6] * (0 if agnostic else max_box_width_height)  

        # Get boxes, with offset by class
        boxes= x[:, :4] + c 

        # Get scores
        scores =  x[:, 4]

        # Perform non-maximum suppression (NMS) on the boxes according to their intersection-over-union (IoU)
        i = torchvision.ops.nms(boxes, scores, iou_threshold)  

        # If too many detections
        if i.shape[0] > max_detections:  
            # Restrict to allowed number
            i = i[:max_detections]

        # Write to output
        output[index] = image[i]

    return output




def xywh2xyxy(x):
    
    """ Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right """

    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)

    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y

    return y