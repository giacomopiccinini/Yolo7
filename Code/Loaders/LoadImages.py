import cv2
import numpy as np
from pathlib import Path
from yaml import safe_load
from pymediainfo import MediaInfo
from Code.Techniques.Rescaling.rescale import letterbox


class LoadImages:

    """Class for loading media (images and videos)"""

    # Constructor
    def __init__(self, path: str, stride=32) -> None:

        # Absolute path
        p = Path(path).absolute()

        if p.is_dir():
            files = p.rglob("*")  # If path is a directory, load all files recursively
        elif p.is_file():
            files = [p]  # If is a single file, load it alone
        else:
            raise Exception(f"ERROR: {p} does not exist")

        # Load admissible file formats
        with open("Settings/formats.yaml") as file:

            # Load yaml file
            d = safe_load(file)

            # Split formats
            image_formats = d["image_formats"]
            video_formats = d["video_formats"]

        # Divide between videos and images
        images = [path for path in files if path.suffix in image_formats]
        videos = [path for path in files if path.suffix in video_formats]

        # Unite images and videos
        files = images + videos

        # Retrieve shapes
        shapes = list(map(self.get_media_shape, files))

        # Store info
        self.files = files
        self.shapes = shapes
        self.n_files = len(files)

        # If no files are found, raise error
        assert self.n_files > 0, f"No images or videos found in {path}."

        # Store flag indicating if a file is a video
        self.video_flag = [False] * len(images) + [True] * len(videos)

        # Store stride and mode
        self.stride = stride
        self.mode = "image"

        # If any video is present
        if any(videos):
            # Initialise the first one, to be used when iterating
            self.new_video(videos[0])
        else:
            # If only imges are present, set to None
            self.capture = None

    def __len__(self):

        """Return length i.e. number of files"""

        return self.n_files

    def get_media_shape(self, media_path: Path) -> tuple:

        """Return (height, width) of media, either
        image or video"""

        # Parse the media
        media = MediaInfo.parse(media_path)

        # Only keep video or images
        track = [
            track for track in media.tracks if track.track_type in ["Video", "Image"]
        ][0]

        # Retrieve shape
        shape = (track.height, track.width)

        return shape

    def new_video(self, path) -> None:

        """Method to be called when a new video is processed"""

        # Initialise frame number
        self.frame = 0

        # Capture video using openCV
        self.capture = cv2.VideoCapture(path)

        # Get total number of frames for video at hand
        self.n_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self):

        """Set iteration number when looping"""
        self.count = 0

        return self

    def __next__(self):

        """Iterate through media"""

        # If we reach end of list, stop iteration
        if self.count == self.n_files:
            raise StopIteration

        # Select the path
        path = self.files[self.count]

        # If the media is a video
        if self.video_flag[self.count]:

            # Read framem and store if operation is successful
            successful, image_BGR = self.capture.read()

            # If not successful, then video has ended
            if not successful:

                # Increase counter of media files upon completion
                self.count += 1

                # Release capture as we have completed the video
                self.capture.release()

                # If all videos have been processed
                if self.count == self.n_files:
                    raise StopIteration

                # If it is not the last video, start reading the next one
                else:
                    # Retrieve new path after incrementing count
                    path = self.files[self.count]

                    # Initialise new video
                    self.new_video(path)

                    # Read framem and store if operation is successful
                    successful, image_BGR = self.capture.read()

            # Increase frame number by one after reading
            self.frame += 1

        # If we have an image
        else:
            # Read BGR image
            image_BGR = cv2.imread(path.__str__(), -1)

            # Increase counter
            self.count += 1

            # Check that the image exists
            assert image_BGR is not None, f"Can't read the image at {path}"

        # Resize to letterbox (i.e. with padding)
        image = letterbox(image_BGR, new_shape=(640, 640), stride=self.stride)[0]

        # Convert BGR to RGB
        image_RGB = image[:, :, ::-1]

        # Convert to channel first
        image_RGB = image_RGB.transpose(2, 0, 1)

        # Contiguous array
        image_RGB = np.ascontiguousarray(image_RGB)

        return path, image_RGB, image_BGR, self.capture
