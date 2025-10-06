r"""Functions to download semantic correspondence datasets"""
import tarfile
import os

import requests

from . import pfpascal
from . import pfwillow
from . import caltech
from . import spair


def load_dataset(benchmark, datapath, thres, device, split='test', augmentation=False, feature_size=16):
    r"""Instantiates desired correspondence dataset"""
    correspondence_benchmark = {
        'pfpascal': pfpascal.PFPascalDataset,
        'pfwillow': pfwillow.PFWillowDataset,
        'caltech': caltech.CaltechDataset,
        'spair': spair.SPairDataset,
    }

    dataset = correspondence_benchmark.get(benchmark)
    if dataset is None:
        raise Exception('Invalid benchmark dataset %s.' % benchmark)

    return dataset(benchmark, datapath, thres, device, split, augmentation, feature_size)


def download_from_google(token_id, filename):
    r"""Downloads desired filename from Google drive"""
    print('Downloading %s ...' % os.path.basename(filename))

    destination = filename + '.tar.gz'
    
    # Try using gdown library first (more reliable for Google Drive)
    try:
        import gdown
        url = f'https://drive.google.com/uc?id={token_id}'
        gdown.download(url, destination, quiet=False)
        print(f"Successfully downloaded using gdown: {destination}")
    except ImportError:
        print("gdown not available, falling back to requests method...")
        # Fallback to the original requests method
        download_from_google_requests(token_id, destination)
    except Exception as e:
        print(f"gdown failed: {e}, falling back to requests method...")
        # Fallback to the original requests method
        download_from_google_requests(token_id, destination)
    
    # Verify the downloaded file is actually a gzip file
    try:
        file = tarfile.open(destination, 'r:gz')
    except tarfile.ReadError as e:
        # If it's not a gzip file, check what we actually downloaded
        with open(destination, 'rb') as f:
            first_bytes = f.read(100)
            if first_bytes.startswith(b'<!'):
                raise Exception(f"Downloaded HTML content instead of gzip file. This usually means the Google Drive link is invalid or requires authentication. First 100 bytes: {first_bytes}")
            else:
                raise Exception(f"Downloaded file is not a valid gzip file. Error: {e}")

    print("Extracting %s ..." % destination)
    file.extractall(filename)
    file.close()

    os.remove(destination)
    os.rename(filename, filename + '_tmp')
    os.rename(os.path.join(filename + '_tmp', os.path.basename(filename)), filename)
    os.rmdir(filename+'_tmp')


def download_from_google_requests(token_id, destination):
    r"""Fallback method using requests for Google Drive download"""
    url = 'https://docs.google.com/uc?export=download'
    session = requests.Session()

    response = session.get(url, params={'id': token_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': token_id, 'confirm': token}
        response = session.get(url, params=params, stream=True)
    
    # Check if we got redirected to a download page (common for large files)
    if 'text/html' in response.headers.get('content-type', ''):
        # Extract the actual download URL from the HTML response
        import re
        content = response.text
        
        # Try multiple patterns for different Google Drive HTML formats
        patterns = [
            r'href="(/uc\?export=download[^"]*)"',
            r'action="(/uc\?export=download[^"]*)"',
            r'window\.location\.href\s*=\s*["\']([^"\']*export=download[^"\']*)["\']',
            r'location\.href\s*=\s*["\']([^"\']*export=download[^"\']*)["\']'
        ]
        
        download_url = None
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                url_part = match.group(1)
                if url_part.startswith('/'):
                    download_url = 'https://docs.google.com' + url_part
                elif url_part.startswith('http'):
                    download_url = url_part
                else:
                    download_url = 'https://docs.google.com/' + url_part
                break
        
        if download_url:
            print(f"Found download URL: {download_url}")
            response = session.get(download_url, stream=True)
        else:
            # If we can't find the URL, try a different approach using the direct download URL
            print("Could not extract download URL from HTML, trying direct download...")
            direct_url = f'https://drive.google.com/uc?export=download&id={token_id}'
            response = session.get(direct_url, stream=True)
    
    save_response_content(response, destination)


def get_confirm_token(response):
    r"""Retrieves confirm token"""
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    r"""Saves the response to the destination"""
    chunk_size = 32768

    with open(destination, "wb") as file:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                file.write(chunk)


def download_dataset(datapath, benchmark):
    r"""Downloads semantic correspondence benchmark dataset from Google drive"""
    if not os.path.isdir(datapath):
        os.mkdir(datapath)

    file_data = {
        'pfwillow': ('1tDP0y8RO5s45L-vqnortRaieiWENQco_', 'PF-WILLOW'),
        'pfpascal': ('1OOwpGzJnTsFXYh-YffMQ9XKM_Kl_zdzg', 'PF-PASCAL'),
        'caltech': ('1IV0E5sJ6xSdDyIvVSTdZjPHELMwGzsMn', 'Caltech-101'),
        'spair': ('1s73NVEFPro260H1tXxCh1ain7oApR8of', 'SPair-71k')
    }

    file_id, filename = file_data[benchmark]
    abs_filepath = os.path.join(datapath, filename)

    if not os.path.isdir(abs_filepath):
        download_from_google(file_id, abs_filepath)
