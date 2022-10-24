import torch
import torch.nn as nn
import requests
from pathlib import Path


class Ensemble(nn.ModuleList):

    """ Ensemble of models"""

    # Constructor
    def __init__(self):
        super(Ensemble, self).__init__()

    # Forward passing
    def forward(self, x, augment=False):

        """ Apply all models to input tensor x and
        concatenate the results. """

        # List of outputs to be filled
        y = []

        # For each model in the ensemble
        for module in self:

            # Append to y the result of the application of the model on x
            y.append(module(x, augment)[0])

        # Concatenate all outputs along axis 1 
        y = torch.cat(y, 1)  

        return y, None 

    @classmethod
    def load(cls, weights, map_location=None) -> nn:

        """ Load an ensemble model with weights 
        
        weights      = name of weights to be loaded
        map_location = GPU or CPU, where to store tensors. Shoud be a torch.device object

        """

        # Instantiate ensemble model
        model = Ensemble()

        # Load each weight
        for weight in weights if isinstance(weights, list) else [weights]:

            # Download pretrained model if not present
            download(weight)

            # Load checkpoint in the map location
            checkpoint = torch.load(weight, map_location=map_location)

            # Append weight to model
            model.append(checkpoint['ema' if checkpoint.get('ema') else 'model'].float().fuse().eval()) 
        
        # If a single model, return it (no ensemble)
        if len(model) == 1:
            return model[-1]  
        # Else, return the ensemble model
        else:
            for k in ['names', 'stride']:
                setattr(model, k, getattr(model[-1], k))
            return model

        
def download(model_path:str) -> None:

    """ Download pretrained weights from the Internet"""

    # Construct Path object
    file = Path(model_path)

    # If the model is not saved locally
    if not file.exists():

        # Interrogate GitHub API on releases
        api_response = requests.get("https://api.github.com/repos/WongKinYiu/yolov7/releases").json()[0]

        # Retrieve assets (i.e. models)
        assets = [asset['name'] for asset in api_response['assets']]

        # Retrieve version tag
        tag = api_response['tag_name']

        # Name of the file (with extension) i.e. removes path info
        name = file.name

        if name in assets:

            # Try downloading model from GitHub
            try:  

                # Construct download URL
                url = f'https://github.com/WongKinYiu/yolov7/releases/download/{tag}/{name}'

                print(f'Downloading {url} to {file}...')

                # Download
                torch.hub.download_url_to_file(url, file)

            except Exception as e:  
                
                # Print exception if something goes wrong
                print(f'Download error: {e}')
            