import csv
import os
from urllib.parse import urlparse
from tqdm import tqdm
from PIL import Image
import tempfile
import json
from downloader import download_all_images

DATA_TYPE = ['train', 'valid', 'test']

def get_label_names_from_display_names(csv_path, display_names):
    """
    Given a list of display names and a CSV file with 'LabelName' and 'DisplayName' columns,
    return a list of matching LabelNames. Returns None for names not found.

    Matching is case-insensitive and ignores spaces/underscores.
    """
    if len(display_names) != len(set(display_names)):
        raise ValueError("List display_names must be unique.")
    
    def normalize(name):
        return name.strip().lower().replace(" ", "").replace("_", "")

    # Build lookup map from normalized display names to original inputs
    target_names = {normalize(name): name for name in display_names}
    results = {name: None for name in display_names}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            norm_display = normalize(row["DisplayName"])
            if norm_display in target_names:
                original = target_names[norm_display]
                results[original] = row["LabelName"]

    # Check if all names exist
    for key, value in results.items():
        if value is None:
            raise ValueError(f"{key} is not a valid label in OIDv7")

    # Preserve input order in output
    return [results[name] for name in display_names]

def extract_OIDv7_data(csv_path, labels):
    """
    Extract information from Open Image v7 annotation file regarding targeted class names
    """

    annotations = []
    image_IDs = {} # use of a dict because 'in' is faster on a dict than a list
    xyxyn = []
    
    # Count lines for progress bar total
    with open(csv_path, 'r') as f:
        total_lines = sum(1 for _ in f) - 1  # minus 1 for the header

    with open(csv_path, newline='') as csvfile:
        
        reader = csv.DictReader(csvfile)
        filename = os.path.basename(urlparse(csv_path).path)
        
        for row in tqdm(reader, total=total_lines, desc=f"Processing {filename}"):
            
            if row['LabelName'] in labels:                    
                imageID = row['ImageID']
                
                if imageID not in image_IDs:
                    image_IDs[imageID] = len(image_IDs)
                                    
                annotations.append(
                    {
                        "image_id": image_IDs[imageID],
                        "category_id": labels.index(row['LabelName'])
                    }
                )
                
                xyxyn.append([float(row['XMin']), float(row['YMin']), float(row['XMax']), float(row['YMax'])])

    return list(image_IDs), annotations, xyxyn

def get_image_dimensions(folder_path):
    image_info = {}

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".jpg"):
            filepath = os.path.join(folder_path, filename)
            with Image.open(filepath) as img:
                width, height = img.size
                key = os.path.splitext(filename)[0]  # Remove .jpg
                image_info[key] = {
                    "width": width,
                    "height": height
                }

    return image_info

def xyxyn_to_xywh(xyxyn, image_width, image_height):
    x_min, y_min, x_max, y_max = xyxyn

    x = x_min * image_width
    y = y_min * image_height
    width = (x_max - x_min) * image_width
    height = (y_max - y_min) * image_height

    return [x, y, width, height]

def create_coco_dataset(manifest_as_dict, target_classes, path="./dataset"):
    """
    Create a directory with images and annotations in COCO format.
    """

    # Check if all keys exist in the dictionary
    for key in DATA_TYPE:
        if key not in manifest_as_dict:
            raise KeyError(f"Key '{key}' is missing in the dictionary.")
    
    labels = get_label_names_from_display_names(manifest_as_dict["class"], target_classes)
    
    # Create COCO categories
    categories = [
        {'id':i, "name": target_classes[i], "supercategory": "none"}
        for i in range(len(labels))
    ]

    for data_type in DATA_TYPE:
        
        # Extract relevant data from Open Images v7 dataset
        annotation_csv = manifest_as_dict[data_type]
        image_IDs, annotations, xyxyn = extract_OIDv7_data(annotation_csv, labels)
        
        # Download images
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt') as temp_file:
            
            for item in tqdm(image_IDs, desc="Writing the list of files to download"):
                file_path = f"{data_type}/{item}" + ".jpg"
                if not os.path.exists(os.path.join(path, file_path)):
                    line = f"{file_path}\n"
                    if data_type == "valid":
                        line = line.replace("valid", "validation")
                    temp_file.write(line)
            temp_file.flush()
            
            download_folder = os.path.join(path, f'{data_type}')

            args = {
            'download_folder': download_folder,
            'image_list': temp_file.name,
            'num_processes': 5
            }
            download_all_images(args)
        
        # Create COCO annotation 
        dimensions_dict = get_image_dimensions(download_folder)
        images = [{
            "id":i,
            "file_name":image+'.jpg',
            "height":dimensions_dict[image]["height"],
            "width":dimensions_dict[image]["width"],}
            for i, image in enumerate(image_IDs)
        ]
        annotations = [{**annotation,
            "bbox":xyxyn_to_xywh(xyxyn[i],
                                images[annotations[i]['image_id']]['width'],
                                images[annotations[i]['image_id']]['height']),
            "id":i,
            "area": images[annotations[i]['image_id']]['width']*images[annotations[i]['image_id']]['height'],
            "iscrowd": 0
            } for i, annotation in enumerate(annotations)
        ]
        
        # Create COCO json annotation file
        data = {
            "info": {},
            "licenses": {},
            "categories": categories,
            "images": images,
            "annotations": annotations 
        }
        with open(os.path.join(download_folder, "_annotations.coco.json"), "w") as f:
            json.dump(data, f, indent=4)