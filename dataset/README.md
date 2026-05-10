# Dataset Setup

## Image Editing Dataset (PIE-Bench)

We use the [PIE-Bench](https://github.com/cure-lab/PnPInversion) editing benchmark for image editing evaluation.

### Download Instructions

1. Clone the PIE-Bench repository:
```bash
git clone https://github.com/cure-lab/PnPInversion.git
```

2. Download the annotation images and place them in `dataset/annotation_images/`:
```bash
# The images follow the structure:
# dataset/annotation_images/
#   ├── 0_random_140/
#   ├── 1_change_object_80/
#   ├── 2_add_object_80/
#   ├── ...
```

3. Copy the mapping file:
```bash
cp PnPInversion/data/mapping_file.json dataset/mapping_file.json
```

### Mapping File Format

The `mapping_file.json` contains entries like:
```json
{
    "0": {
        "image_path": "0_random_140/000000.jpg",
        "original_prompt": "[a cat] sitting on a bench",
        "editing_prompt": "[a dog] sitting on a bench",
        "editing_instruction": "change cat to dog",
        "blended_word": "cat dog",
        "mask": [...]
    }
}
```

## Text Editing Dataset

The text editing dataset (`text_editing_dataset.json`) is included in this repository.
It contains 765+ positive/negative sentiment sentence pairs generated for evaluating
discrete text inversion.

### Format
```json
{
    "0": {
        "positive_sentiment": ["Context sentence (positive)", "Content sentence (positive)"],
        "negative_sentiment": ["Context sentence (negative)", "Content sentence (negative)"]
    }
}
```

The task is to invert a negative sentiment sentence and edit it to have positive sentiment
while preserving the sentence structure.
