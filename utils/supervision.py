import os
from PIL import Image
import supervision as sv
from supervision.detection.core import Detections
from rfdetr.detr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRBase, RFDETRLarge

AVAILABLE_MODELS = ["nano", "small", "medium", "base", "large"]

def get_model(model: str, pretrain_weights: str | None = None):
    """
    Return an RF-DETR model instance based on the provided model name.

    Args:
        model (str): The name of the model to load. Must be one of AVAILABLE_MODELS.

    Returns:
        torch.nn.Module: The corresponding RF-DETR model instance.

    Raises:
        ValueError: If the model name is not recognized.
    """
    if model == "nano":
        return RFDETRNano(pretrain_weights=pretrain_weights)
    elif model == "small":
        return RFDETRSmall(pretrain_weights=pretrain_weights)
    elif model == "medium":
        return RFDETRMedium(pretrain_weights=pretrain_weights)
    elif model == "base":
        return RFDETRBase(pretrain_weights=pretrain_weights)
    elif model == "large":
        return RFDETRLarge(pretrain_weights=pretrain_weights)
    else:
        raise ValueError(f"Model type '{model}' not recognized. Available model types: {AVAILABLE_MODELS}")

def save_detections(image: Image.Image, detections: Detections, output_path: str = "detections.png") -> None:
    """
    Annotates an image with detection results and saves it to disk.

    Args:
        image (Image.Image): Input image to annotate.
        detections: Supervision detection object.
        output_path (str): Path to save the annotated image. Default is 'detections.png'.

    Returns:
        None
    """
    if len(detections.class_id) == 0:
        print("No detections to annotate.")
        return

    # Format labels with class_id and confidence
    labels = [
        f"id: {int(class_id)} (conf: {confidence:.2f})"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]

    # Annotate image
    annotated_image = sv.BoxAnnotator(thickness=1).annotate(image, detections)
    annotated_image = sv.LabelAnnotator(text_scale=0.3, text_padding=3).annotate(
        annotated_image, detections, labels 
    )

    # Ensure output directory exists
    dir_path = os.path.dirname(output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    # Save result
    annotated_image.save(output_path)
    print(f"Saved annotated image to {output_path}")