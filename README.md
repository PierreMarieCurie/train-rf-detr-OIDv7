# Fine-Tune RF-DETR on Custom Classes from Open Images V7 dataset

> **Want to use the latest [SOTA object detection models](https://github.com/roboflow/rf-detr) from Roboflow on your own custom classes?** First, check if your classes exist in the [Open Images dataset](https://storage.googleapis.com/openimages/web/index.html). If they do, good news! This repo shows you how to **fine-tune a RF-DETR model** with a **single command**.

It includes:

- 🔧 Train on any Open Images class(es) by name
- ⚡ No need to install Python dependencies manually — thanks to the Ultraviolet environment manager
- 🐳 Optional Docker support for full reproducibility and minimal local setup (TO DO)
- ⬇️ Downloads only relevant images and annotation files — not the full Open Images dataset
- 🗃️ Automatic conversion of Open Images to COCO format
- ⚙️ Configurable training parameters from CLI

## Requirements

- `uv` (UltraViolet) environment manager. If not, just run (macOS and Linux):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

- Python 3.8+. If not, just run:

```bash
uv python install 3.9.13
```

> Check [Astral documentation](https://docs.astral.sh/uv/getting-started/installation) if you need alternative installation methods.

## Usage

### Quickstart

**First, select your target class(es) from the 601 available in Open Images V7**. View the set of boxable classes as a **hierarchy** [here](https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy_visualizer/circle.html), or **explore the dataset visually** using the [Open Images Bounding Boxes Explorer](https://storage.googleapis.com/openimages/web/visualizer/index.html?type=detection).

For example, to fine-tune RF-DETR on **"Snail"** and **"Plate"** classes from Open Images V7:

```bash
uv run train.py \
  --target-classes snail plate \
  --epochs 10 \
  --batch-size 4
```

This will:

- Download relevant annotations and images
- Convert them to COCO format
- Fine-tune RF-DETR on the selected subset
- Save logs and checkpoints in a timestamped folder under `results/`

HERE PARLER DU SCRIPT DE VISUALISATION

### Advanced settings

You can customize training with the following arguments:

| Argument                   | Description                                                                                                        |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `--target-classes`         | **(Required)** One or more class names to fine-tune on (e.g., `snail, plate`)                                      |
| `--epochs`                 | Number of training epochs (default: `10`)                                                                          |
| `--learning-rate` / `--lr` | Learning rate (default: `1e-4`)                                                                                    |
| `--batch-size`             | Batch size (default: `8`)                                                                                          |
| `--grad-accum-steps`       | Gradient accumulation steps (default: `1`)                                                                         |
| `--result-folder`          | Custom path to save results. If not set, a timestamped folder with class names is auto-created in `results` folder |
| `--dataset-folder`         | Folder where the COCO-converted dataset will be stored (default: `dataset`)                                        |
| `--csv-folder`             | Folder where CSVs from Open Images will be saved (default: `OIDv7_csv`)                                            |
| `--manifest-path`          | Path to the Open Images V7 manifest JSON with links to the annotation files (default: `csv_manifest.txt`)          |

Alternatively, run:

```bash
uv run train.py -h
```

## 🐳 Docker (Coming Soon)

A Dockerfile is in progress to fully containerize training for reproducibility.

## 🧪 TODO

- [ ] Add example configs
- [ ] Add support for multi-class evaluation
- [ ] Upload pretrained weights
- [ ] Publish Docker image

## 📜 License

This repo is licensed under a [MIT License](LICENSE).

Please note that OIDv7 annotations are licensed by Google LLC under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license. The images are listed as having a [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) license.

Also, RF-DETR is licensed under a [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) license.

## 🙏 Acknowledgments

- This project builds on [RF-DETR](https://github.com/roboflow/rf-detr), a state-of-the-art, open-source object detection model developed and released by [Roboflow](https://roboflow.com/). Their work on detection models and dataset tools has been instrumental in making advanced vision systems more accessible.

- This project uses [uv](https://github.com/astral-sh/uv) by Astral — a blazing fast Python package manager and runtime that enables clean, reproducible environments without manual setup.
