import argparse
import os
from datetime import datetime
from .supervision import AVAILABLE_MODELS

DEFAULT_MANIFEST_PATH = "csv_manifest.txt"
DEFAULT_DATASET_FOLDER = "dataset"
DEFAULT_CSV_FOLDER = "OIDv7_csv"
DEFAULT_EPOCHS = 20
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_BATCH_SIZE = 8
DEFAULT_GRAD_ACCUM_STEPS = 1
DEFAULT_MODEL = "base"

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune RF-DETR using OIDv7 data")

    # Training args
    training_group = parser.add_argument_group("training arguments")

    training_group.add_argument(
        "--target-classes",
        nargs="+",
        required=True,
        help="List of target classes (e.g., tiger cheetah)"
    )
    
    training_group.add_argument(
        "--model",
        type=str,
        choices=["nano", "small", "medium", "base", "large"],
        default=DEFAULT_MODEL,
        help=f"Select model size. Default: {DEFAULT_MODEL}. Available: {AVAILABLE_MODELS}"
    )
    
    training_group.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})"
    )

    training_group.add_argument(
        "--learning-rate", "--lr",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})"
    )
    
    training_group.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})"
    )
    
    training_group.add_argument(
        "--grad-accum-steps",
        type=int,
        default=DEFAULT_GRAD_ACCUM_STEPS,
        help=f"Grad accumulation steps (default: {DEFAULT_GRAD_ACCUM_STEPS})"
    )
    
    training_group.add_argument(
        "--early-stopping",
        action="store_true",
        help=f"Early stopping (add --early-stopping to enable it)"
    )
    
    # Dataset args
    dataset_group = parser.add_argument_group("dataset arguments")
    dataset_group.add_argument(
        "--result-folder", 
        type=str, 
        default=None, 
        help="Path where the results will be saved. If not set, a timestamped folder with label names will be created."
    )
    dataset_group.add_argument(
        "--dataset-folder", 
        type=str, 
        default=DEFAULT_DATASET_FOLDER, 
        help=f"Folder where the dataset in COCO format will be saved (default: {DEFAULT_DATASET_FOLDER})"
    )
    dataset_group.add_argument(
        "--csv-folder", 
        type=str, 
        default=DEFAULT_CSV_FOLDER, 
        help=f"Folder where the CSV files will be saved (default: {DEFAULT_CSV_FOLDER})"
    )
    dataset_group.add_argument(
        "--manifest-path", 
        type=str, 
        default=DEFAULT_MANIFEST_PATH, 
        help=f"Path to the manifest file containing OIDv7 annotation links (default: {DEFAULT_MANIFEST_PATH})"
    )

    args = parser.parse_args()

    # Create default result folder if not provided
    if args.result_folder is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        name = "_".join(args.target_classes)
        args.result_folder = os.path.join("results", f"{name}_{timestamp}")

    return args