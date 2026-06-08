import json
import os
import os.path as osp
from typing import Any, Dict, Iterable, List, Optional


REQUIRED_INFER_FIELDS = {
    "idx",
    "question_type",
    "question_info",
    "images",
}

REQUIRED_QUESTION_INFO_FIELDS = {
    "question",
    "answer",
}

SUPPORTED_QUESTION_TYPES = {
    "open_ended",
    "closed_ended",
    "single_choice",
    "multi_choice",
}


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        results = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                results.append(json.loads(line))
        return results

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return [data]


def resolve_input_file(input_file: Optional[str]) -> str:
    if input_file:
        return input_file
    raise ValueError("--input-file must be provided.")


def normalize_image_paths(
        image_paths: Iterable[str],
        *,
        replace_src: Optional[str] = None,
        replace_dst: Optional[str] = None,
        max_images: Optional[int] = None,
) -> List[str]:
    results = []
    for image_path in image_paths:
        if replace_src is not None and replace_dst is not None:
            image_path = image_path.replace(replace_src, replace_dst)
        results.append(image_path)
    if max_images is not None:
        results = results[:max_images]
    return results


def validate_infer_samples(samples: List[Dict[str, Any]]) -> None:
    for idx, sample in enumerate(samples):
        missing = REQUIRED_INFER_FIELDS - set(sample)
        if missing:
            raise ValueError(
                f"Sample #{idx} is missing required fields: {sorted(missing)}"
            )

        question_type = sample["question_type"]
        if question_type not in SUPPORTED_QUESTION_TYPES:
            raise ValueError(
                f"Sample #{idx} has unsupported question_type '{question_type}'. "
                f"Supported values: {sorted(SUPPORTED_QUESTION_TYPES)}"
            )

        question_info = sample["question_info"]
        if not isinstance(question_info, dict):
            raise ValueError(f"Sample #{idx} field 'question_info' must be a dict.")

        missing_qi = REQUIRED_QUESTION_INFO_FIELDS - set(question_info)
        if missing_qi:
            raise ValueError(
                f"Sample #{idx} question_info is missing fields: {sorted(missing_qi)}"
            )

        images = sample["images"]
        if not isinstance(images, list) or not images:
            raise ValueError(
                f"Sample #{idx} field 'images' must be a non-empty list of image paths."
            )

        if question_type in {"single_choice", "multi_choice"} and "choices" not in question_info:
            raise ValueError(
                f"Sample #{idx} question_info must contain 'choices' for question_type '{question_type}'."
            )


def load_prediction_items(pred_dir: str) -> List[Dict[str, Any]]:
    files = sorted(
        osp.join(pred_dir, name)
        for name in os.listdir(pred_dir)
        if name.endswith(".json")
    )
    results: List[Dict[str, Any]] = []
    for file in files:
        data = load_json_or_jsonl(file)
        results.extend(data)
    return results


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def infer_output_dir_name(model_path: str) -> str:
    return osp.basename(osp.normpath(model_path))
