import argparse


def parse():

    """Parser"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        const="GPU",
        default="GPU",
        nargs="?",
        type=str,
        choices=["GPU", "CPU"],
        help="Insert device type",
    )

    parser.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default="Models/yolov7.pt",
        help="model.pt path(s)",
    )
    parser.add_argument(
        "--source", type=str, default="Input", help="Source of images/videos"
    )

    parser.add_argument(
        "--img-size", type=int, default=640, help="Inference size (pixels)"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.25, help="Object confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45, help="IOU threshold for NMS"
    )
    parser.add_argument("--view-img", action="store_true", help="display results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-conf", action="store_true", help="save confidences in --save-txt labels"
    )
    parser.add_argument(
        "--nosave", action="store_true", help="do not save images/videos"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class: --class 0, or --class 0 2 3",
    )
    parser.add_argument(
        "--agnostic-nms", action="store_true", help="class-agnostic NMS"
    )
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument(
        "--project", default="runs/detect", help="save results to project/name"
    )
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument("--no-trace", action="store_true", help="don`t trace model")

    args = parser.parse_args()

    return args
