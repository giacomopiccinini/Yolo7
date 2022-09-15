from pathlib import Path

def download_model(model_path):

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
          