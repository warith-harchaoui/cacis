"""
examples.imagenet.cost_matrix
=============================

Build a semantic cost matrix for ImageNet classes from FastText word
embeddings.

Construction
------------
For each class name (1000 entries, e.g., ``"tench, Tinca tinca"``) we:

1. Keep the **first comma-separated synonym** (``"tench"``).
2. Tokenize to lowercase ASCII words (``["tench"]`` here, but
   ``"great white shark"`` → ``["great", "white", "shark"]``).
3. Look up each token in FastText and **average** the resulting vectors.
   FastText ``.bin`` models can synthesize subword vectors for out-of-vocabulary
   tokens; ``.vec`` models skip OOV tokens.

Cost is cosine distance::

    C[i, j] = 1 - cos(emb_i, emb_j)   in [0, 2]
    C[i, i] = 0

We save ``{"C": Tensor(K, K), "class_names": List[str]}`` as a single
``.pt`` file, sized at roughly 4 MB for ImageNet-1k.

CLI
---
::

    python -m examples.imagenet.cost_matrix \\
        --data-root /data/imagenet \\
        --fasttext /models/cc.en.300.bin \\
        --out cost_matrix.pt

Either a ``.bin`` model (loaded via the ``fasttext`` package, supports subword
OOV) or a ``.vec`` / ``.vec.gz`` text file (loaded without external deps) is
accepted; pick by file extension.

Why this is small and local
---------------------------
The build runs once, embeds 1000 short strings, and outputs a tiny file. It
is intended to run on a laptop, not in the cloud — the resulting ``.pt`` is
then mounted into the training container as a model artifact.

Author
------
Warith Harchaoui <wharchaoui@nexton-group.com>
"""

from __future__ import annotations

import argparse
import gzip
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from examples.imagenet.classnames import load_imagenet_classnames

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z]+")


def _first_synonym(label: str) -> str:
    """Return the first comma-separated synonym, stripped of whitespace."""
    return label.split(",")[0].strip()


def _tokenize(label: str) -> List[str]:
    """Tokenize a label to lowercase ASCII word pieces."""
    return [m.group(0).lower() for m in _WORD_RE.finditer(label)]


def _load_vec_file(path: Path) -> Tuple[Dict[str, int], np.ndarray]:
    """
    Load a ``.vec`` / ``.vec.gz`` text file (word2vec / FastText text format).

    Format: header line ``"<n_words> <dim>"`` followed by one line per word
    of the form ``"word v1 v2 ... vd"``.
    """
    opener = gzip.open if path.name.endswith(".gz") else open
    logger.info("Loading FastText .vec from %s", path)

    vocab: Dict[str, int] = {}
    vectors: List[np.ndarray] = []
    with opener(path, "rt", encoding="utf-8") as f:
        header = f.readline().split()
        dim = int(header[1])
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) != dim + 1:
                continue
            vocab[parts[0]] = len(vectors)
            vectors.append(np.asarray(parts[1:], dtype=np.float32))
    return vocab, np.stack(vectors)


class FastTextEmbedder:
    """
    Adapter over FastText ``.bin`` or ``.vec`` files.

    ``.bin`` is preferred for ImageNet class names because compound labels like
    ``"airliner"``, ``"snowplow"``, or ``"hockey puck"`` benefit from FastText's
    subword model for any OOV token.

    Parameters
    ----------
    path:
        Path to ``cc.en.300.bin`` (binary, subword) or ``wiki-news-300d.vec``
        (text, in-vocab only).
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if self.path.suffix == ".bin":
            self._backend = "fasttext"
            try:
                import fasttext  # type: ignore[import-not-found]
            except ImportError as e:
                raise ImportError(
                    "Loading a .bin FastText model requires the 'fasttext' "
                    "package (try: pip install fasttext-wheel)."
                ) from e
            logger.info("Loading FastText .bin from %s", self.path)
            self._model = fasttext.load_model(str(self.path))
            self.dim: int = self._model.get_dimension()
            self._vocab: Optional[Dict[str, int]] = None
            self._vectors: Optional[np.ndarray] = None
        elif self.path.name.endswith((".vec", ".vec.gz")):
            self._backend = "vec"
            self._model = None
            self._vocab, self._vectors = _load_vec_file(self.path)
            self.dim = int(self._vectors.shape[1])
        else:
            raise ValueError(
                f"Unsupported FastText file extension: {self.path.name}. "
                "Expected .bin, .vec, or .vec.gz."
            )

    def word_vector(self, word: str) -> Optional[np.ndarray]:
        """Return the embedding for ``word``, or ``None`` if OOV (``.vec`` only)."""
        if self._backend == "fasttext":
            return np.asarray(self._model.get_word_vector(word), dtype=np.float32)
        assert self._vocab is not None and self._vectors is not None
        idx = self._vocab.get(word)
        return None if idx is None else self._vectors[idx]

    def label_vector(self, label: str) -> np.ndarray:
        """
        Embed a class label as the **mean** of its token vectors.

        Falls back to a zero vector (with a warning) if every token is OOV. The
        result is *not* normalized; callers should normalize before computing
        cosine similarity.
        """
        tokens = _tokenize(_first_synonym(label))
        vecs: List[np.ndarray] = []
        for tok in tokens:
            v = self.word_vector(tok)
            if v is not None:
                vecs.append(v)
        if not vecs:
            logger.warning("No FastText vector for label '%s' — using zeros.", label)
            return np.zeros(self.dim, dtype=np.float32)
        return np.mean(vecs, axis=0).astype(np.float32)


def build_cost_matrix(
    class_names: Sequence[str],
    embedder: FastTextEmbedder,
    *,
    metric: str = "cosine",
) -> np.ndarray:
    """
    Build a ``(K, K)`` semantic cost matrix.

    Parameters
    ----------
    class_names:
        K class labels in the order expected by the classifier (i.e., the order
        ``ImageFolder(train).classes`` returns).
    embedder:
        A configured :class:`FastTextEmbedder`.
    metric:
        Currently only ``"cosine"`` is supported. ``C[i, j] = 1 - cos_sim``.

    Returns
    -------
    np.ndarray
        ``(K, K)`` ``float32`` cost matrix with zeros on the diagonal and
        non-negative off-diagonal entries.
    """
    if metric != "cosine":
        raise ValueError(f"Unsupported metric: {metric}")

    K = len(class_names)
    embeds = np.stack([embedder.label_vector(n) for n in class_names]).astype(np.float32)

    norms = np.linalg.norm(embeds, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    embeds_n = embeds / norms

    sim = embeds_n @ embeds_n.T  # cosine similarity in [-1, 1]
    C = (1.0 - sim).astype(np.float32)
    np.fill_diagonal(C, 0.0)
    return np.clip(C, 0.0, None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute the ImageNet semantic cost matrix from FastText.",
    )
    parser.add_argument(
        "--data-root", type=Path, required=True,
        help="ImageNet root containing LOC_synset_mapping.txt",
    )
    parser.add_argument(
        "--fasttext", type=Path, required=True,
        help="Path to a FastText .bin model or .vec(.gz) file",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("cost_matrix.pt"),
        help="Output path for the cost matrix bundle",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    class_names = load_imagenet_classnames(args.data_root)
    logger.info("Loaded %d class names.", len(class_names))

    embedder = FastTextEmbedder(args.fasttext)
    C = build_cost_matrix(class_names, embedder)

    K = C.shape[0]
    od = C[~np.eye(K, dtype=bool)]
    logger.info(
        "Cost matrix: shape=%s | offdiag mean=%.4f median=%.4f max=%.4f",
        C.shape, od.mean(), np.median(od), od.max(),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"C": torch.from_numpy(C), "class_names": list(class_names)},
        args.out,
    )
    logger.info("Saved cost matrix to %s", args.out)


if __name__ == "__main__":
    main()
