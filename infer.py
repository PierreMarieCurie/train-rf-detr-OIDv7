import argparse
from PIL import Image, ImageDraw
from rfdetr import RFDETRBase

def load_and_resize_image(image_path, target_width=640):
    image = Image.open(image_path).convert("RGB")
    original_width, original_height = image.size
    scale = target_width / original_width
    new_height = int(original_height * scale)
    resized_image = image.resize((target_width, new_height))
    return resized_image

def draw_detections(image, detections):
    draw = ImageDraw.Draw(image)
    bboxes = detections.xyxy.tolist()
    for bbox, class_id in zip(bboxes, detections.class_id):
        draw.rectangle(bbox, outline="red", width=3)
        draw.text((bbox[0], bbox[1] - 10), str(class_id), fill="red")
    return image

def main(args):
    # Load model
    model = RFDETRBase(pretrain_weights=args.checkpoint, num_classes=1)

    # Load and resize image
    image = load_and_resize_image(args.image, target_width=args.width)

    # Run inference
    detections = model.predict(image)

    # Draw results
    annotated_image = draw_detections(image, detections)

    # Save output
    annotated_image.save(args.output)
    print(f"Image saved to: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RF-DETR inference on a single image using a custom checkpoint.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--output", type=str, default="output.jpg", help="Path to save annotated output image")
    parser.add_argument("--width", type=int, default=640, help="Target width to resize image (preserves aspect ratio)")

    args = parser.parse_args()
    main(args)

# uv run infer.py --image dataset/image2.png --checkpoint results/snail/checkpoint_best_ema.pth --output output.jpg