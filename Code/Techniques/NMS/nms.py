import torch
import torchvision
import numpy as np


def non_max_suppression(
    prediction,
    confidence_threshold=0.25,
    iou_threshold=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
):
    """
    Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    # The result of prediction is:
    # [x, y, w, h] (0,1,2,3)
    # confidence in detection (4)
    # confidence in class (>= 5)

    # Retrieve number of classes from prediction (subtract box coordinates and confidence)
    number_of_classes = prediction.shape[2] - 5

    # Select candidates that surpass a confidence threshold
    candidates = prediction[..., 4] > confidence_threshold

    # Fix a few settings

    # Maximum box width and height in pixels
    max_box_width_height = 4096

    # Maximum number of detections per image
    max_detections = 300

    # Maximum number of boxes into torchvision.ops.nms()
    max_nms = 30000

    # Ensure multi labelling is indeed possible
    multi_label &= number_of_classes > 1

    for index, boxes_features in enumerate(prediction):

        # Retain valid candidates for bounding boxes
        boxes_features = boxes_features[candidates[index]]

        # If no boxes are detected, move on
        if not boxes_features.shape[0]:
            continue

        # Compute confidence
        if number_of_classes == 1:
            # For models with one class, class_loss is 0 and class_confidence is always 0.5, no need to multiplicate
            # Propagate detection confidence to all classes
            boxes_features[:, 5:] = boxes_features[:, 4:5]
        else:
            # Overall confidence, is confidence_in_object * confidence_in_class
            boxes_features[:, 5:] *= boxes_features[:, 4:5]

        # Extract boxes coordinates
        boxes_coordinates = boxes_features[:, :4]

        # Create boxes converting from (center x, center y, width, height) to (x1, y1, x2, y2)
        boxes = xywh2xyxy(boxes_coordinates)

        # Detections matrix n x 6 (xyxy, conf, cls)
        if multi_label:

            # Fetch indices and classes of valid detections
            detection_index, detected_class = (
                (boxes_features[:, 5:] > confidence_threshold).nonzero(as_tuple=False).T
            )

            # Fix detected class by adding the offset due to box + confidence
            detected_class += 5

            # Concatenate
            boxes_features = torch.cat(
                (
                    boxes[detection_index],
                    boxes_features[detection_index, detected_class, None],
                ),
                detected_class[:, None].float(),
                dim=1,
            )

        else:

            # Best class only
            confidences, detected_classes = boxes_features[:, 5:].max(1, keepdim=True)

            # Concatenate
            boxes_features = torch.cat(
                (boxes, confidences.float(), detected_classes), dim=1
            )[confidences.view(-1) > confidence_threshold]

        # Filter by class
        if classes is not None:
            boxes_features = boxes_features[
                (
                    boxes_features[:, 5:6]
                    == torch.tensor(classes, device=boxes_features.device)
                ).any(1)
            ]

        # Fetch number of detected boxes
        n_boxes = boxes_features.shape[0]

        # If no boxes, continue
        if not n_boxes:
            continue
        # Else, if maximum number of boxes is achieved
        elif n_boxes > max_nms:
            # Sort boxes by confidence, and take the first max_nms
            boxes_features = boxes_features[
                boxes_coordinates[:, 4].argsort(descending=True)[:max_nms]
            ]

        # Get classes
        c = boxes_features[:, 5:6] * (0 if agnostic else max_box_width_height)

        # Get boxes, with offset by class
        boxes = boxes_features[:, :4] + c

        # Get scores
        scores = boxes_features[:, 4]

        # Perform non-maximum suppression (NMS) on the boxes according to their intersection-over-union (IoU)
        i = torchvision.ops.nms(boxes, scores, iou_threshold)

        # If too many detections
        if i.shape[0] > max_detections:
            # Restrict to allowed number
            i = i[:max_detections]

        # Initialise output (to be filled)
        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]

        # Write to output
        output[index] = boxes_features[i]

    return output


def xywh2xyxy(x):

    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right"""

    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)

    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y

    return y
