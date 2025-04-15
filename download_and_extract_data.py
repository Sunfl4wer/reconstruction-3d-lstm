import gdown
import tarfile
import os


def download_file(file_id, output_path):
    """
    Downloads a file from Google Drive using its file ID.

    :param file_id: str, The ID of the file to download.
    :param output_path: str, The path where the downloaded file will be saved.
    """
    # Construct the Google Drive URL
    url = f"https://drive.google.com/uc?id={file_id}"
    
    # Download the file
    gdown.download(url, output_path, quiet=False)
    
    print(f"File downloaded and saved to {output_path}")


def extract_tar_file(file_path, extract_to):
    """
    Extracts a .tar.gz file to the specified directory.

    :param file_path: str, The path to the .tar.gz file.
    :param extract_to: str, The directory where the files will be extracted.
    """
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
        print(f"Extracted {file_path} to {extract_to}")


# Download files
download_file("1rRiE7rkhb8Gi9-Lva-D4PnX1jzRTlGEf", "microCT-voxel-filled.tar.gz")
download_file("11fiX89VYipMGmDDvvjjxQ-zYXzT_HMIg", "microCT-multiview-100.tar.gz")

# Extract files
extract_tar_file("microCT-voxel-filled.tar.gz", "microCT-voxel-filled")
extract_tar_file("microCT-multiview-100.tar.gz", "microCT-multiview-100")