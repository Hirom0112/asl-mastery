"""From-scratch CV detectors per ADR 0011.

Four detectors share this subpackage: hand_detector (bbox),
hand_landmarks (21 keypoints), pose (8 upper-body keypoints), face
(bbox). Each ships its own architecture file, dataset class, and
training loop. The Modal entrypoints live in training/modal_app.py.

Every weight in this subpackage is trained by this project on
labeled data produced under /labeling. No pretrained backbones.
No torchvision.models imports. No load_state_dict reading foreign
weights. See docs/decisions/0012-strict-from-scratch-cv-constraint.md.
"""
