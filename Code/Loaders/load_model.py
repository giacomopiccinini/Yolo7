import torch
import torch.nn as nn

from Code.Classes.Ensemble import Ensemble
from Code.Utilities.download_model import download_model


def load_model(weights, map_location=None) -> nn:

    """Load an ensemble model with weights

    weights      = name of weights to be loaded
    map_location = GPU or CPU, where to store tensors

    """

    # Instantiate ensemble model
    model = Ensemble()

    # Load each weight
    for weight in weights if isinstance(weights, list) else [weights]:

        # Download pretrained model if not present
        download_model(weight)

        # Load checkpoint
        ckpt = torch.load(weight, map_location=map_location)

        # Append weight to model
        model.append(ckpt["ema" if ckpt.get("ema") else "model"].float().fuse().eval())

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print("Ensemble created with %s\n" % weights)
        for k in ["names", "stride"]:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble
