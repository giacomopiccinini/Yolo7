import torch.nn as nn

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