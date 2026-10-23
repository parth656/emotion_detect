# Emotion Detection Evaluation

Facial-expression classification estimates visual patterns; it does not reliably reveal a person's internal emotional state. Use this project only as a technical demonstration, not for high-stakes decisions or surveillance.

## Evaluation setup

Document the dataset source and license, label mapping, subject-level train/validation/test split, preprocessing, image size, and model checkpoint. Avoid splitting frames from the same person or recording across training and test sets.

## Metrics

Report:
- Accuracy and macro-F1
- Per-class precision and recall
- Confusion matrix
- Performance by lighting, occlusion, and camera conditions
- Known limitations and examples of failure

## Responsible use

Expressions vary by individual, culture, and context. Evaluate data quality and representation, avoid claims that a predicted class is a definitive emotion, and do not use outputs for hiring, discipline, or other high-impact decisions.
