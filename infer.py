import argparse
from utils.supervision import get_model, save_detections, AVAILABLE_MODELS
from PIL import Image
from urllib.parse import urlparse
from urllib.request import urlopen
import io

def main(args):
    
    # Load image
    parsed = urlparse(args.image)
    if parsed.scheme in ("http", "https"):
        with urlopen(args.image) as response:
            image_data = response.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
    else:
        image = Image.open(args.image).convert("RGB")    
    
    # Load model
    model = get_model(args.model, pretrain_weights=args.checkpoint)
    
    # Run inference
    detections = model.predict(image)

    # Save results
    save_detections(image, detections, args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RF-DETR inference on a single image using a custom checkpoint.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, required=True, help=f"Name of the model. Available models: {AVAILABLE_MODELS}")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--output", type=str, default="detections.png", help="Path to save annotated output image")

    args = parser.parse_args()
    main(args)