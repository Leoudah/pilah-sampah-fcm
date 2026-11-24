"""Predict one or more images with an exported waste classifier."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps


def prepare_image(original, config):
    image = ImageOps.exif_transpose(original).convert("RGB")
    shortest = min(image.size)
    size = tuple(config["resize"] * dim // shortest for dim in image.size)
    resized = image.resize(size, Image.Resampling.BILINEAR)
    crop = config["crop"]
    left, top = (round((dim - crop) / 2) for dim in size)
    pixels = np.asarray(resized.crop((left, top, left + crop, top + crop)), dtype=np.float32)
    mean = np.asarray(config["mean"], dtype=np.float32)
    std = np.asarray(config["std"], dtype=np.float32)
    return ((pixels / 255 - mean) / std).transpose(2, 0, 1)[None]


def image_tensor(path, config):
    with Image.open(path) as original:
        return prepare_image(original, config)


def load_model(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    model = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    metadata = model.get_modelmeta().custom_metadata_map
    return model, json.loads(metadata["preprocess"]), json.loads(metadata["labels"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("images", nargs="+", type=Path)
    args = parser.parse_args()

    model, config, labels = load_model(args.model)
    input_name = model.get_inputs()[0].name
    for path in args.images:
        scores = model.run(None, {input_name: image_tensor(path, config)})[0][0]
        best = np.argsort(scores)[::-1][:3]
        print(json.dumps({
            "image": str(path),
            "predictions": [{"label": labels[i], "score": float(scores[i])} for i in best],
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
