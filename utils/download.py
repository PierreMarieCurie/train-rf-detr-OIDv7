
import os
from urllib.parse import urlparse
import urllib.request
from tqdm import tqdm

def download_file(url: str, download_dir: str) -> str:
    """
    Download a file from the given URL into the specified directory.

    Args:
        url (str): Full URL of the file to download
        download_dir (str): Path where to save the file

    Returns:
        str: The local file path.
    """
    # Ensure the target directory exists
    os.makedirs(download_dir, exist_ok=True)

    # Extract the filename from the URL
    filename = os.path.basename(urlparse(url).path)
    if not filename:
        raise ValueError(f"Could not extract filename from URL: {url}")
    file_path = os.path.join(download_dir, filename)

    # Skip downloading if file already exists
    if os.path.exists(file_path):
        print(f"{file_path} already exists.")
        return file_path

    try:
        print(f"Download {file_path} from {url}...")

        # Open the URL for reading the file content
        with urllib.request.urlopen(url) as response:
            # Attempt to get the total size for progress tracking
            total_size = int(response.getheader('Content-Length', 0))
            chunk_size = 1024  # Download in 1KB chunks

            # Open the target file for writing and show a progress bar
            with open(file_path, 'wb') as out_file, tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=filename,
                leave=False
            ) as pbar:
                # Read and write the file chunk by chunk
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    pbar.update(len(chunk))
        print(f"Done.")

        return file_path

    except Exception as e:
        # Raise a clear error if download fails
        raise RuntimeError(f"Failed to download {filename} from {url}: {e}")

def download_from_manifest(manifest_path: str, download_dir=".") -> dict:
    """
    Download multiple files listed in a manifest file "manifest_path".
    Each line of the manifest should contain a name and a URL seperated by a comma.
    Empty lines or lines starting with '#' are ignored (treated as comments).
    
    Args:
        manifest_path (str): Full URL of the manifest file
        download_dir (str, optional): Path where to save the file. Defaults to current path.

    Returns:
        dict: Manifest file information as a dictionary.
    """
    # Check that the manifest file exists
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    # Read and clean all non-comment lines from the manifest
    with open(manifest_path, "r") as f:
        lines = []
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or ',' not in line:
                continue
            parts = line.split(",", 1)
            lines.append(parts)
    print(f"Found {len(lines)} files to download.")
    
    # Download each file listed in the manifest
    manifest_as_dict = {}
    for line in lines:
        file_path = download_file(line[1], download_dir)
        manifest_as_dict[line[0]] = file_path

    return manifest_as_dict