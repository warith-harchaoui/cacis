"""
examples.imagenet.cloud
=======================

Config-driven Vast.ai launcher for the ImageNet pipeline.

Layout
------
- :mod:`examples.imagenet.cloud.config`      — typed YAML loader (``VastConfig``).
- :mod:`examples.imagenet.cloud.launch_vast` — CLI: search for the cheapest
                                               matching offer, create an
                                               instance, set env vars, kick off
                                               the bootstrap.
- ``vast_bootstrap.sh``                      — bash script executed inside the
                                               container; downloads ImageNet
                                               from Kaggle, runs each loss in
                                               sequence, syncs outputs to
                                               Backblaze B2, self-destroys.

User-facing config template: ``config/cloud-vast.yaml.example``.
"""
