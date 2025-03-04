import requests
from PIL import ImageFile, Image

def download_file(url: str, save_path: str) -> None:
    """
    Download file from url and save it to save_path.
    It is used for downloading dataset.
    """
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
    else:
        raise RuntimeError(f"Failed to download {url}")

def open_image(image_path: str) -> ImageFile.ImageFile:
    """
    Open image file from image_path.
    """
    return Image.open(image_path).convert("RGB")