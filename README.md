# Analyzing the Inner Workings of Transformers in Compositional Generalization
This repository contains data and code for the paper "Analyzing the Inner Workings of Transformers in Compositional Generalization".

## Setup
We require Python 3.10 or later.
```
pip install -r requirements.txt
```
## Usage
### Train a base model
Train a base model for the pattern `[PATTERN_NAME]` and task `[TASK_NAME]`.
```
python scripts/train.sh [PATTERN_NAME] [TASK_NAME] [SEED]
```
### Subnetwork probing
Apply subnetwork probing for the pattern `[PATTERN_NAME]`, task `[TASK_NAME]`, and `[EPOCH_NUM]` epoch trained base model.
```
python scripts/subnetwork_probe.sh [PATTERN_NAME] [TASK_NAME] [EPOCH_NUM] [SEED]
```
### Concept scrubbing
Apply concept scrubbing for the pattern `[PATTERN_NAME]`, task `[TASK_NAME]`, and `[EPOCH_NUM]` epoch trained base model.
```
python scripts/scrub.sh [PATTERN_NAME] [TASK_NAME] [EPOCH_NUM] [SEED]
```
### Evaluate models
Evaluate models trained for the pattern `[PATTERN_NAME]`, task `[TASK_NAME]`, and `[EPOCH_NUM]` epochs.
```
python scripts/evaluate.sh [PATTERN_NAME] [TASK_NAME] [EPOCH_NUM] [SEED]
```
## Directory Structure
This repository follows the structure below:
```
CG_interp/
│── src/ # Contains the source code
│── data/ # Contains datasets
  |── base/ # For training
  |── scrub/ # For concept scrubbing
```

## Reference
The implementation of our code is based on the following projects.
- [LEACE](https://github.com/EleutherAI/concept-erasure)
- [Subnetwork Probing](https://github.com/stevenxcao/subnetwork-probing)
- [Tranformer (PyTorch)](https://github.com/pytorch/examples/tree/main/language_translation)

## Bibtex
```
@inproceedings{kumon2025compositional
    title={Analyzing the Inner Workings of Transformers in Compositional Generalization},
    author={Ryoma Kumon and Hitomi Yanaka},
    booktitle={Proceedings of the 2025 Annual Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
    year={2025},
}
```
