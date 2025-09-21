
from io import BytesIO
from PIL import Image


def img_from_byte_array(arr):
    '''Decode and return an image from a numpy uint8 array containing the compressed image bytes.
    '''
    return Image.open(BytesIO(arr.tobytes()))