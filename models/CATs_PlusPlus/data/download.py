r"""Functions to download semantic correspondence datasets"""
import tarfile
import os

import requests

try:
    from . import pfpascal
    from . import pfwillow
    from . import caltech
    from . import spair
except ImportError:
    # Fallback imports
    import models.CATs_PlusPlus.data.pfpascal as pfpascal
    import models.CATs_PlusPlus.data.pfwillow as pfwillow
    import models.CATs_PlusPlus.data.caltech as caltech
    import models.CATs_PlusPlus.data.spair as spair


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

    url = 'https://docs.google.com/uc?export=download'
    destination = filename + '.tar.gz'
    session = requests.Session()

    try:
        # First, get the direct download URL
        response = session.get(url, params={'id': token_id}, stream=True, allow_redirects=True)
        
        # Check if we got redirected to the actual download URL
        if response.url != url:
            print(f"Redirected to: {response.url}")
            # Check if the response is HTML (confirmation page) or actual file
            content_type = response.headers.get('content-type', '')
            if 'text/html' in content_type:
                # This is a confirmation page, we need to extract the download link
                print("Got confirmation page, extracting download link...")
                # Use the original method with confirmation token
                token = get_confirm_token(response)
                if token:
                    params = {'id': token_id, 'confirm': token}
                    response = session.get(url, params=params, stream=True)
                else:
                    # Try alternative method - look for download link in HTML
                    import re
                    html_content = response.text
                    # Look for download link pattern
                    download_match = re.search(r'href="(/uc\?export=download[^"]*)"', html_content)
                    if download_match:
                        download_url = 'https://docs.google.com' + download_match.group(1)
                        print(f"Found download link: {download_url}")
                        response = session.get(download_url, stream=True)
                    else:
                        raise Exception("Could not find download link in confirmation page")
            else:
                # This should be the actual file
                response = session.get(response.url, stream=True)
        else:
            # Original logic for handling confirmation token
            token = get_confirm_token(response)
            if token:
                params = {'id': token_id, 'confirm': token}
                response = session.get(url, params=params, stream=True)
        
        save_response_content(response, destination)
        
    except Exception as e:
        print(f"Original download method failed: {e}")
        print("Trying alternative method with gdown...")
        
        # Fallback to gdown
        try:
            import gdown
            file_url = f'https://drive.google.com/uc?id={token_id}'
            gdown.download(file_url, destination, quiet=False)
        except ImportError:
            raise Exception("gdown library not available. Please install it with: pip install gdown")
        except Exception as gdown_error:
            raise Exception(f"Both download methods failed. Original error: {e}, gdown error: {gdown_error}")
    file = tarfile.open(destination, 'r:gz')

    print("Extracting %s ..." % destination)
    file.extractall(filename)
    file.close()

    os.remove(destination)
    os.rename(filename, filename + '_tmp')
    os.rename(os.path.join(filename + '_tmp', os.path.basename(filename)), filename)
    os.rmdir(filename+'_tmp')


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
    else:
        print(f"Dataset {filename} already exists in {datapath}")
