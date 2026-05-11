"""
examples.imagenet.classnames
============================

Load the 1000 human-readable ImageNet class names in the canonical order used
by ``torchvision.datasets.ImageFolder``.

ImageFolder sorts class directories alphabetically by name (the WordNet synset
ids ``n01440764``, ``n01443537``, ...). The standard ImageNet/Kaggle
distribution ships a file ``LOC_synset_mapping.txt`` mapping each synset id to
a comma-separated label string, e.g.::

    n01440764 tench, Tinca tinca
    n01443537 goldfish, Carassius auratus

We sort by synset id and return the corresponding labels — this guarantees the
returned list aligns index-for-index with the class indices that
``ImageFolder(train).classes`` produces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List


def load_synset_to_label(synset_mapping_path: Path) -> Dict[str, str]:
    """
    Parse ``LOC_synset_mapping.txt`` into a ``{synset_id: label_text}`` dict.
    """
    mapping: Dict[str, str] = {}
    with open(synset_mapping_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            synset_id, label = line.split(" ", 1)
            mapping[synset_id] = label.strip()

    if len(mapping) != 1000:
        raise ValueError(
            f"Expected 1000 entries in {synset_mapping_path}, got {len(mapping)}."
        )
    return mapping


def load_imagenet_classnames(data_root: Path) -> List[str]:
    """
    Return class names in the order ``ImageFolder(data_root/'train').classes`` produces.

    Parameters
    ----------
    data_root:
        Directory containing ``LOC_synset_mapping.txt`` (typically the same
        directory that contains ``train/`` and ``val/``).

    Raises
    ------
    FileNotFoundError
        If the mapping file is missing.
    """
    mapping_path = Path(data_root) / "LOC_synset_mapping.txt"
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Cannot find {mapping_path}. It ships with the official ImageNet "
            "release and Kaggle's ImageNet-Object-Localization-Challenge dataset; "
            "place it next to the train/ and val/ directories."
        )
    synset_to_label = load_synset_to_label(mapping_path)
    sorted_synsets = sorted(synset_to_label.keys())
    return [synset_to_label[s] for s in sorted_synsets]
