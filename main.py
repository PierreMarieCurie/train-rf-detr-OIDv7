from utils.download import download_file, download_from_manifest
from utils.dataset import create_coco_dataset
from utils.train import fine_tune_RFDETR
import os

CSV_FOLDER = 'csv_folder'
DOWNLOAD_SCRIPT_URL = "https://raw.githubusercontent.com/openimages/dataset/master/downloader.py"
DATASET_PATH = 'dataset' # A METTRE EN ARGUMENT !!!
TARGET_CLASSES = ["wheelchair", "bicycle"] # A METTRE EN ARGUMENT !!!
TARGET_CLASSES = ["snail"]

def main():
    
    print("------ Step 1: Download files ------")
    manifest_as_dict = download_from_manifest("csv_manifest.txt", download_dir=CSV_FOLDER) 
    download_file(DOWNLOAD_SCRIPT_URL, '.')

    print("------ Step 2: Dataset creation ------")
    create_coco_dataset(manifest_as_dict, TARGET_CLASSES, DATASET_PATH)

    print("------ Step 3: RF-DETR fine tuning ------")
    results_path = os.path.join("results", "_".join(TARGET_CLASSES))
    fine_tune_RFDETR(DATASET_PATH, results_path)

if __name__ == "__main__":
    main()
