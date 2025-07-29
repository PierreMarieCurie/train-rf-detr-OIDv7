
from utils.download import download_file, download_from_manifest
from utils.dataset import create_coco_dataset
from utils.args import parse_args
from utils.supervision import get_model

DOWNLOAD_SCRIPT_URL = "https://raw.githubusercontent.com/openimages/dataset/master/downloader.py"

def main():
    args = parse_args()

    print("------ Step 1: Download files ------")
    manifest_as_dict = download_from_manifest(args.manifest_path, download_dir=args.csv_folder)
    download_file(DOWNLOAD_SCRIPT_URL, '.')

    print("------ Step 2: Dataset creation ------")
    create_coco_dataset(manifest_as_dict, args.target_classes, args.dataset_folder)

    print("------ Step 3: RF-DETR fine tuning ------")
    model = get_model(args.model)
    model.train(
        dataset_dir=args.dataset_folder,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.learning_rate,
        output_dir=args.result_folder,
        early_stopping=args.early_stopping,
    )

if __name__ == "__main__":
    main()