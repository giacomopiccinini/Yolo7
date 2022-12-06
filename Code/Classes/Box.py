import numpy as np


class Box:

    """Class for bounding box coming from inference"""

    def __init__(self, centre_x, centre_y, width, height):

        """Initiate bounding box instance with coordinates of centre,
        and width, height"""

        # Store coordinates of centre and dimensions
        self.centre_x = centre_x
        self.centre_y = centre_y

        self.width = width
        self.height = height

    def xywh2xyxy(self):

        """Convert from coordinates of centre to coordinates of
        edges"""

        # x coordinate of left edge
        self.left = self.centre_x - self.width / 2

        # x coordinate of right edge
        self.right = self.centre_x + self.width / 2

        # y coordinate of top edge
        self.top = self.centre_y + self.height / 2

        # y coordinate of bottom edge
        self.bottom = self.centre_y - self.height / 2

    def area(self):

        """Compute area of bounding box"""

        self.area = self.width * self.height

    def get_centre(self):

        """Return coordinates of centre of bounding box"""

        return np.array([self.centre_x, self.centre_y])

    def get_xywh(self):

        """ " Return coordinates of bounding box in the (centre_x, centre_y, width, height)
        format"""

        return np.array([self.centre_x, self.centre_y, self.width, self.height])

    def get_xyxy(self):

        """ " Return coordinates of bounding box in the (left_edge, top_edge, right_edge, bottom_edge)
        format"""

        # Convert coordinates to edges
        self.xywh2xyxy()

        return np.array([self.left, self.top, self.right, self.bottom])
