from dataclasses import dataclass
import torch
from PIL import Image
from pathlib import Path


@dataclass(slots=True)
class UserContext:
    tenant_id: str
    user_id: str
    roles: list[str]

@dataclass
class PageArtifacts:
    page_index: int
    page_number: int
    image: Image.Image
    original: torch.Tensor
    image_patch_embeddings: torch.Tensor
    x_patches: int
    y_patches: int
    payload: dict


@dataclass
class IngestJob:
    job_id: str
    pdf_paths: list[Path]
    tenant_id: str
    allowed_roles: list[str]
    allowed_user_ids: list[str]
    department: str
    doc_type: str
    sensitivity: str
