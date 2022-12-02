import logging
import torch

from Code.Parser.parser import parse
from Code.Utilities.torch import select_device
from Code.Classes.Ensemble import Ensemble

from Code.Loaders.LoadImages import LoadImages
from Code.Loaders.LoadStreams import LoadStreams

from Code.Techniques.NMS.nms import non_max_suppression

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.NOTSET)

# Parse arguments
# args = parse()

# Set device
device = select_device("GPU")

# Load yolo model with given weights on the device of choice
model = Ensemble.load("yolov7.pt", device)
