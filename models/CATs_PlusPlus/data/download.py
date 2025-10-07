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
    import shutil  # Import shutil at the beginning of the function
    
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
    print("Extracting %s ..." % destination)
    # Extract to a temporary directory first
    temp_extract_dir = filename + '_extracting'
    
    # Create the extraction directory
    os.makedirs(temp_extract_dir, exist_ok=True)
    
    # Try using system tar command first (much faster than Python tarfile)
    import subprocess
    try:
        print("Using system tar for faster extraction...")
        result = subprocess.run(['tar', '-xzf', destination, '-C', temp_extract_dir], 
                              capture_output=True, text=True, check=True)
        print("Extraction complete using system tar")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("System tar not available, falling back to Python tarfile...")
        # Fallback to Python tarfile with optimized settings
        with tarfile.open(destination, 'r:gz', bufsize=1024*1024) as file:  # 1MB buffer
            members = file.getmembers()
            print(f"Extracting {len(members)} files...")
            
            # Extract files in batches for better performance
            batch_size = 100
            for i in range(0, len(members), batch_size):
                batch = members[i:i+batch_size]
                for member in batch:
                    file.extract(member, temp_extract_dir)
                
                if i % 1000 == 0:  # Progress every 1000 files
                    print(f"Extracted {min(i+batch_size, len(members))}/{len(members)} files...")
            
            print(f"Extraction complete: {len(members)} files extracted")

    # Remove the downloaded tar.gz file
    os.remove(destination)
    
    # Find the actual dataset directory inside the extracted content
    # Handle nested directory structures like SPair-71k/SPair-71k/
    def find_dataset_root(extract_dir, target_name):
        """Recursively find the dataset root directory"""
        items = os.listdir(extract_dir)
        print(f"Searching in {extract_dir} for {target_name}")
        print(f"Found items: {items}")
        
        # Look for a directory that matches our target name
        for item in items:
            item_path = os.path.join(extract_dir, item)
            if os.path.isdir(item_path):
                if item == target_name:
                    print(f"Found exact match: {item_path}")
                    return item_path
                # Check if there's a nested directory with the same name
                nested_path = os.path.join(item_path, target_name)
                if os.path.isdir(nested_path):
                    print(f"Found nested match: {nested_path}")
                    return nested_path
                # Recursively check subdirectories
                nested_result = find_dataset_root(item_path, target_name)
                if nested_result:
                    return nested_result
        print(f"No match found in {extract_dir}")
        return None
    
    # Find the actual dataset directory
    actual_dataset_dir = find_dataset_root(temp_extract_dir, os.path.basename(filename))
    
    print(f"Looking for dataset directory: {os.path.basename(filename)}")
    print(f"Found dataset directory: {actual_dataset_dir}")
    
    if actual_dataset_dir:
        # Move the found dataset directory to the final location
        if actual_dataset_dir != filename:
            if os.path.exists(filename):
                print(f"Removing existing directory: {filename}")
                shutil.rmtree(filename)
            print(f"Moving {actual_dataset_dir} to {filename}")
            shutil.move(actual_dataset_dir, filename)
        # Clean up the temporary extraction directory
        print(f"Cleaning up temporary directory: {temp_extract_dir}")
        shutil.rmtree(temp_extract_dir)
    else:
        # Fallback: if we can't find the expected structure, just rename the temp directory
        print(f"Dataset directory not found, using fallback method")
        print(f"Contents of temp directory: {os.listdir(temp_extract_dir)}")
        if os.path.exists(filename):
            print(f"Removing existing directory: {filename}")
            shutil.rmtree(filename)
        print(f"Renaming {temp_extract_dir} to {filename}")
        os.rename(temp_extract_dir, filename)
    
    print("Dataset downloaded and extracted successfully")


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
    import shutil
    
    if not os.path.isdir(datapath):
        os.mkdir(datapath)
    
    # Clean up any leftover "_extracting" directories from previous failed downloads
    for item in os.listdir(datapath):
        if item.endswith('_extracting'):
            extracting_path = os.path.join(datapath, item)
            if os.path.isdir(extracting_path):
                print(f"Cleaning up leftover extraction directory: {extracting_path}")
                shutil.rmtree(extracting_path)

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
