"""
examples.imagenet.cloud
=======================

Config-driven AWS launcher for the ImageNet training pipeline.

Layout
------
- :mod:`examples.imagenet.cloud.config`     — typed YAML loader.
- :mod:`examples.imagenet.cloud.launch_ec2` — provision an EC2 instance, run the
                                              training container, sync outputs
                                              back to S3, and self-terminate.

The user-facing config template lives at ``config/cloud.yaml.example``.
"""
