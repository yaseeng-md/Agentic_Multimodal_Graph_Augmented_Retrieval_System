"""Common data structures for evaluation datasets."""

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence
from PIL import Image


@dataclass
class EvalSample:
    """Standardized evaluation sample representation across all benchmarks."""

    id: str
    query: str
    ground_truth: Any  # string or list of acceptable ground truth strings
    contexts: list[str] = field(default_factory=list)
    context_images: list[Image.Image] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    hallucination_label: Optional[int] = None  # 1: hallucinated, 0: faithful, None: unlabeled

    @property
    def ground_truth_text(self) -> str:
        """Returns ground truth as a single normalized string."""
        if isinstance(self.ground_truth, (list, tuple)):
            return self.ground_truth[0] if self.ground_truth else ""
        return str(self.ground_truth or "")

    @property
    def is_multimodal(self) -> bool:
        """Returns True if the sample contains visual page context."""
        return len(self.context_images) > 0


@dataclass
class EvalDataset:
    """Standardized in-memory dataset container for evaluation runs."""

    name: str
    samples: list[EvalSample] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> EvalSample:
        return self.samples[idx]

    def __iter__(self):
        return iter(self.samples)

    def select(self, limit: int) -> "EvalDataset":
        """Returns a subset of the dataset up to limit samples."""
        return EvalDataset(name=self.name, samples=self.samples[:limit])
