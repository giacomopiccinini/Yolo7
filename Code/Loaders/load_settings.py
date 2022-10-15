from yaml import safe_load
from glob import glob


def load_settings():

    """ Load all .yaml files """

    # Find all setting files
    files = glob("Settings/*.yaml")

    # Initialise settings dictionary
    settings = {}

    for file in files:

        # Read each file
        with open(file, "r") as stream:
            d = safe_load(stream)

        settings.update(d)

    return settings