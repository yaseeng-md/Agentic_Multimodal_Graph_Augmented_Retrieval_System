from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from colpali_engine.models import ColPali, ColPaliProcessor
from transformers import BitsAndBytesConfig

ADAPTER_REPO_ID = "vidore/colpali-v1.3"
BASE_REPO_ID = "vidore/colpaligemma-3b-pt-448-base"

MODEL_ROOT = Path("models")
ADAPTER_DIR = MODEL_ROOT / "colpali-v1.3"
BASE_DIR = MODEL_ROOT / "colpaligemma-3b-pt-448-base"


def _has_any_model_weights(folder: Path) -> bool:
    return any(
        (folder / name).exists()
        for name in [
            "model.safetensors",
            "pytorch_model.bin",
            "model-00001-of-00002.safetensors",
            "model-00001-of-00003.safetensors",
        ]
    )


def _is_valid_adapter_snapshot(folder: Path) -> bool:
    required = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "special_tokens_map.json",
    ]
    return folder.exists() and all((folder / f).exists() for f in required)


def _is_valid_base_snapshot(folder: Path) -> bool:
    required = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
    ]
    return folder.exists() and all((folder / f).exists() for f in required) and _has_any_model_weights(folder)


def _patch_adapter_config(adapter_dir: Path, local_base_dir: Path) -> None:
    config_path = adapter_dir / "adapter_config.json"
    data = json.loads(config_path.read_text(encoding="utf-8"))

    # Point the adapter to the local base model directory.
    # Absolute path is safest on Windows.
    data["base_model_name_or_path"] = str(local_base_dir.resolve())

    config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _download_repo(repo_id: str, target_dir: Path) -> None:
    """
    Download a complete HF repository snapshot into target_dir.
    """
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = target_dir.with_name(target_dir.name + "_tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    shutil.rmtree(target_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading: {repo_id}")
    print(f"Target dir : {target_dir}")

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(tmp_dir),
    )

    metadata = {
        "repo_id": repo_id,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (tmp_dir / "snapshot_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    shutil.move(str(tmp_dir), str(target_dir))


def download_models_if_needed() -> None:
    """
    Download adapter and base model once, then patch the adapter so the whole
    stack can be loaded offline.
    """
    print(f"Adapter dir : {ADAPTER_DIR}")
    print(f"Base dir    : {BASE_DIR}")

    if not _is_valid_base_snapshot(BASE_DIR):
        _download_repo(BASE_REPO_ID, BASE_DIR)

    if not _is_valid_adapter_snapshot(ADAPTER_DIR):
        _download_repo(ADAPTER_REPO_ID, ADAPTER_DIR)

    _patch_adapter_config(ADAPTER_DIR, BASE_DIR)

    if not _is_valid_adapter_snapshot(ADAPTER_DIR):
        raise RuntimeError(f"Adapter snapshot incomplete: {ADAPTER_DIR}")

    if not _is_valid_base_snapshot(BASE_DIR):
        raise RuntimeError(f"Base snapshot incomplete: {BASE_DIR}")

    print("✓ Local offline model files are ready.")


def load_model_and_processor() -> tuple[ColPali, ColPaliProcessor]:
    """
    Download once if needed, then load strictly from local files.
    """
    download_models_if_needed()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = ColPali.from_pretrained(
        str(ADAPTER_DIR),
        quantization_config=bnb_config,
        device_map={"": 0},
        local_files_only=True,
    ).eval()

    processor = ColPaliProcessor.from_pretrained(
        str(ADAPTER_DIR),
        local_files_only=True,
    )

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory | Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

    return model, processor


def select_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda:0", torch.float16
    return "cpu", torch.float32



def load_visual_config() -> dict:
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        return config.get("VISUAL_RETRIVAL", {})
    except FileNotFoundError:
        return {}
    except Exception as exc:
        return {}


if __name__ == "__main__":
    model, proc = load_model_and_processor()