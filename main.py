import cv2
import torch
import logging
import torch.backends.cudnn as cudnn

from copy import deepcopy

from Code.Parser.parser import parse
from Code.Utilities.torch import select_device
from Code.Classes.Ensemble import Ensemble
from Code.Techniques.Rescaling.rescale import rescale_coordinates
from Code.Loaders.LoadImages import LoadImages
from Code.Loaders.LoadStreams import LoadStreams
from Code.Techniques.NMS.nms import non_max_suppression

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

logging.getLogger("matplotlib").setLevel(logging.ERROR)


if __name__ == "__main__":

    # Set up logging
    logging.basicConfig(level=logging.NOTSET)
    log = logging.getLogger(__name__)

    # Parse arguments
    args = parse()

    # Set device
    device = select_device(args.device)

    # Load yolo model with given weights on the device of choice
    log.info(f"Loading model with weights from {args.weights}")
    model = Ensemble.load(args.weights, device)
    model.eval()

    # Enable half precision on GPUs
    use_half_precision = device.type != "cpu"
    if use_half_precision:
        model.half()

    # Load stride from model
    stride = int(model.stride.max())

    # Stream from camera
    if args.source == "webcam":
        log.info("Loading stream from webcam")
        dataset = LoadStreams()
    else:
        # Else load dataset
        log.info(f"Loading media from {args.source}")
        dataset = LoadImages(args.source, stride)

    # Loop over images (or frames) in dataset
    for path, image, image_BGR, capture in dataset:

        # Find image data-type (8-bit or 16-bit) and associate the max value to it
        if image.dtype == "uint8":
            maximum = 255.0
        elif image.dtype == "uint16":
            maximum = 65535.0
        else:
            raise TypeError(f"Image at {path} is neither 8-bit nor 16-bit")

        # Send image to device (e.g. GPU)
        image = torch.from_numpy(image).to(device)

        # Convert to float
        image = image.half() if use_half_precision else image.float()

        # Normalise image between 0 and 1
        image /= maximum

        # If image has three channels, insert a new dummy dimension at position 0 (should be batch)
        if image.ndimension() == 3:
            image = image.unsqueeze(0)

        # Run inference (calculating gradients would cause a GPU memory leak)
        with torch.no_grad():
            log.info(f"Running inference for {path}")
            # The output is a tuple, throw away second elemnt (out of 2)
            prediction = model(image)[0]

            # Prediction shape is (1 (=batch?), size_x*size_y*n_channels, 80(classes) + 5)

        # Apply NMS to prediction
        # The result are a list of boxes, with coordinates, confidencce and class
        # These boxes are 640 x 640, so need rescaling
        log.info("Applying non max suppression")
        small_boxes = non_max_suppression(
            prediction=prediction,
            confidence_threshold=args.conf_thres,
            iou_threshold=args.iou_thres,
            classes=args.classes,
            agnostic=args.agnostic_nms,
        )

        for i, small_box in enumerate(small_boxes):

            # Copy small box
            big_boxes = deepcopy(small_box)

            # Rescale coordinates and get the actual box
            big_boxes[:, :4] = rescale_coordinates(
                image_BGR.shape, image.shape[2:], small_box[:, :4]
            ).round()

            for big_box in big_boxes:

                # Extract corners used to draw the box
                top_left_corner = (int(big_box[0]), int(big_box[1]))
                bottom_right_corner = (int(big_box[2]), int(big_box[3]))

                # Draw the box
                cv2.rectangle(
                    image_BGR, top_left_corner, bottom_right_corner, (0, 255, 0), 2
                )
                
                line_thickness = 3
                font_thickness = max(line_thickness -1, 1)
                text_size = cv2.getTextSize("Person", 0, fontScale=line_thickness / 3, thickness=font_thickness)[0]
                c2 = top_left_corner[0] + text_size[0], top_left_corner[1] - text_size[1] - 3
                cv2.rectangle(image_BGR, top_left_corner, c2, (0, 255, 0), -1, cv2.LINE_AA)  # filled
                cv2.putText(image_BGR, "Person", (top_left_corner[0], top_left_corner[1] - 2), 0, line_thickness / 3, [225, 255, 255], thickness=font_thickness, lineType=cv2.LINE_AA)

            cv2.imwrite("prova.jpg", image_BGR)