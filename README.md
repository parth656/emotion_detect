# Real-Time Emotion Detection

A computer-vision project for facial emotion recognition. The profile portfolio describes the project as real-time emotion detection from webcam video using Python, OpenCV, and deep learning.

## Features

The intended workflow is to capture frames, detect a face, preprocess the face image, run the trained emotion classifier, and display the predicted class. Confirm the exact supported emotion labels and model input shape from the implementation.

## Technology

- Python
- OpenCV
- Deep-learning model (check the source for the framework and architecture)
- Webcam/video input

## Setup

```bash
git clone https://github.com/parth656/emotion_detect.git
cd emotion_detect
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the dependencies specified by the repository's dependency file. The exact dependency filename and application entry point should be checked in the source tree before running the project.

## How to use

1. Install the required packages and model weights.
2. Allow webcam access if prompted.
3. Launch the configured application entry point.
4. Review predictions as estimates rather than definitive statements about a person's internal emotional state.

## Evaluation

Document the dataset, label mapping, class balance, train/validation/test split, and metrics. Include a confusion matrix and per-class precision/recall. Test lighting, occlusion, different faces, camera quality, and out-of-distribution inputs.

## Responsible use

Facial-expression classifiers can be inaccurate and biased across demographic groups, contexts, and cultures. An expression is not a reliable proof of a person's actual emotion. Do not use predictions for high-stakes decisions, surveillance, hiring, or discipline.

## Before release

- [ ] Add screenshots or a demo GIF.
- [ ] Include exact installation and run commands.
- [ ] Document model architecture and weights.
- [ ] Add tests for preprocessing and inference.
- [ ] State dataset licensing and known limitations.
