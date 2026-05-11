"""
examples.imagenet
=================

End-to-end ImageNet training with cost-aware losses.

The cost matrix is derived from the **semantic** similarity of class names,
embedded with FastText. Confusions between semantically distant classes
(e.g., ``boat`` vs ``zebra``) incur higher cost than confusions between
semantically adjacent classes (e.g., ``tiger`` vs ``leopard``).

Pipeline
--------
1. ``cost_matrix.py``   — build ``C ∈ R^{1000×1000}`` from FastText embeddings.
2. ``data.py``          — standard ImageNet ``ImageFolder`` loaders with a
                          ``DistributedSampler`` for multi-GPU / multi-node.
3. ``model.py``         — torchvision ResNet (trained from scratch).
4. ``train.py``         — DDP + AMP training loop; works under ``torchrun``
                          on a single node or many.
5. ``Dockerfile``       — runtime image for AWS / GCP GPU instances.

See the module docstrings for usage; see the project ``README.md`` for the
end-to-end cloud workflow.
"""
