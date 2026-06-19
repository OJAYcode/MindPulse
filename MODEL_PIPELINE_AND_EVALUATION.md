# MindPulse Model Pipeline and Evaluation Explanation

## 1. Project Aim

MindPulse is a multimodal stress-detection system. It uses two independent signals from a normal laptop:

- the webcam, for facial emotion recognition
- the microphone, for voice stress-tendency recognition

The project does not use cloud inference APIs. Both models are trained locally and saved as TensorFlow/Keras model artifacts. During inference, the face branch and voice branch make separate predictions first. Their outputs are then combined using late fusion to estimate a final stress level.

This design is important because stress is not always visible from one signal alone. A user may look calm but sound tense, or may sound neutral while their facial expression suggests pressure. Combining both signals makes the system more practical than depending on only face or only voice.

## 2. Dataset Preparation

### Face Dataset

The face branch uses a folder-per-emotion dataset format:

```text
data/processed/face_affectnet_clean/
  angry/
  happy/
  neutral/
  sad/
```

Each folder contains face images that belong to that emotion class. The preparation step standardizes the dataset so the training script can read one clean folder structure instead of many mixed source formats.

The latest face model was trained with four classes:

| Class | Meaning |
| --- | --- |
| angry | facial expression suggests anger or irritation |
| happy | facial expression suggests positive affect |
| neutral | facial expression is mostly relaxed or expressionless |
| sad | facial expression suggests sadness or low mood |

The data preparation step helped improve reliability by keeping the labels focused and removing unsupported classes from the final training run.

### Voice Dataset

The voice branch uses a folder-per-label audio format:

```text
data/processed/voice_iemocap_clean/
  happy/
  neutral/
  sad/
  stressed/
```

For the final model, the voice problem was simplified into a binary stress-tendency task:

| Original Labels | Final Training Label |
| --- | --- |
| happy, neutral, sad, calm | not_stressed |
| stressed, angry, fear, fearful | stressed |

This binary setup was chosen because it is more reliable for this project than trying to distinguish many detailed voice emotions. Voice emotion recognition is hard because different people speak with different accents, pitch ranges, microphone quality, and background noise. A focused binary task gives the model a clearer learning target.

## 3. Face Preprocessing Pipeline

The face model receives images as input. Before training, the pipeline applies several preprocessing steps:

| Step | Purpose |
| --- | --- |
| Load image files from class folders | Converts the folder structure into labeled training samples |
| Resize images | Makes every input the same size for MobileNetV2 |
| Normalize pixel values | Converts image values into a scale that the neural network can learn from more easily |
| Split into train, validation, and test sets | Keeps evaluation fair by testing on samples not used for training |
| Data augmentation | Helps the model handle small changes in lighting, position, and camera angle |
| Class weighting | Reduces the effect of class imbalance when one emotion has more samples than another |

The face training pipeline uses MobileNetV2 transfer learning. This means the model starts from a network that already understands general image features, then learns the specific emotion classes from the project dataset.

## 4. Face Model Architecture

The face branch uses MobileNetV2 as the backbone model.

MobileNetV2 is suitable for this project because:

- it is lightweight enough for laptop inference
- it is stronger than training a CNN from scratch on limited data
- it works well for image classification tasks
- it can be fine-tuned after the first training stage

The model is trained in two stages:

| Stage | What Happens | Why It Helps |
| --- | --- | --- |
| Transfer-learning stage | MobileNetV2 base is mostly frozen while the classifier head learns | Fast and stable first training stage |
| Fine-tuning stage | Some deeper MobileNetV2 layers are unfrozen | Allows the model to adapt better to facial emotion images |

The final face model outputs probabilities for:

```text
angry, happy, neutral, sad
```

For example, the model might output:

```json
{
  "angry": 0.10,
  "happy": 0.05,
  "neutral": 0.70,
  "sad": 0.15
}
```

The highest probability becomes the predicted face emotion.

## 5. Voice Preprocessing Pipeline

The voice model receives short audio clips as input. Raw audio cannot be used directly by the CNN, so it is converted into mel-spectrogram features.

| Step | Purpose |
| --- | --- |
| Load audio file | Reads the waveform from `.wav` or supported audio files |
| Resample audio | Keeps all clips at a consistent sample rate |
| Trim silence | Removes long quiet sections that do not help prediction |
| Normalize volume | Reduces differences caused by recording loudness |
| Convert to mel-spectrogram | Turns sound into a time-frequency representation |
| Convert amplitude to decibels | Makes the spectrogram more useful for learning |
| Pad or crop to fixed length | Ensures every voice sample has the same tensor shape |
| Cache processed tensors | Makes repeated training and evaluation faster |
| Apply light augmentation | Adds small variations so the model generalizes better |

The mel-spectrogram is useful because it represents how sound frequencies change over time. A CNN can then learn patterns such as intensity, pitch movement, vocal tension, and energy distribution.

## 6. Voice Model Architecture

The voice branch uses a small CNN trained on mel-spectrogram tensors.

The model is intentionally lightweight because the project must run on a normal laptop. It does not use a large transformer or cloud model. The CNN learns local patterns in the spectrogram in a similar way that an image CNN learns local patterns in an image.

The final voice model outputs probabilities for:

```text
not_stressed, stressed
```

For example:

```json
{
  "not_stressed": 0.32,
  "stressed": 0.68
}
```

The highest probability becomes the predicted voice stress tendency.

## 7. Late Fusion

MindPulse uses late fusion as the main presentation-ready fusion method.

Late fusion means:

1. The face model makes its own prediction.
2. The voice model makes its own prediction.
3. The fusion layer combines the two prediction outputs.
4. The final result is mapped to a stress level: `low`, `medium`, or `high`.

This is different from training one huge model that takes face and voice together. Late fusion is simpler, easier to explain, easier to debug, and better suited for a final-year project using separate datasets.

### Why Rule-Based Fusion Is Used

Rule-based fusion is the main method because the available face and voice datasets are separate. A learned fusion model needs paired samples where the same session has:

```text
face image + voice audio + real stress label
```

Since the current datasets are not naturally paired, rule-based fusion is more honest and more reliable for this stage of the project.

The optional learned fusion model remains available as an experimental add-on, but the rule-based method is the main project method.

## 8. Current Evaluation Results

The latest saved model artifacts were evaluated on held-out test data.

| Branch | Task | Labels | Accuracy | Precision | Recall | F1 Score |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Face model | Facial emotion recognition | angry, happy, neutral, sad | 81.7% | 82.1% | 81.7% | 81.6% |
| Voice model | Voice stress tendency recognition | not_stressed, stressed | 72.5% | 72.5% | 72.5% | 72.5% |

These results show that both branches crossed the target of 70% accuracy. The face branch is stronger because image transfer learning with MobileNetV2 is very effective. The voice branch also passed 70% after simplifying the task to binary stress tendency and using cleaner IEMOCAP-based data with light augmentation.

## 9. Evaluation Metrics Explained

### Accuracy

Accuracy measures the percentage of total predictions that were correct.

```text
accuracy = correct predictions / total predictions
```

If the model evaluates 1,000 samples and predicts 725 correctly, the accuracy is 72.5%.

Accuracy is easy to understand, but it does not tell the full story when classes are imbalanced. For example, if one class is much larger than another, a model can get good accuracy by mostly predicting the larger class. That is why precision, recall, and F1 score are also reported.

### Precision

Precision answers this question:

```text
When the model predicts a class, how often is it correct?
```

For the voice model, precision for `stressed` tells us how often the model is correct when it says a voice sample is stressed.

High precision means the model makes fewer false alarms.

Example:

```text
If the model predicts "stressed" 100 times and 73 are actually stressed, precision is 73%.
```

### Recall

Recall answers this question:

```text
Out of all real samples of a class, how many did the model find?
```

For the voice model, recall for `stressed` tells us how many truly stressed samples were detected.

High recall means the model misses fewer real cases.

Example:

```text
If there are 100 truly stressed samples and the model detects 72 of them, recall is 72%.
```

### F1 Score

F1 score combines precision and recall into one balanced metric.

```text
F1 = 2 * (precision * recall) / (precision + recall)
```

F1 is useful when we care about both false alarms and missed detections. In this project, F1 is important because stress detection should not simply guess the most common class. A good F1 score shows the model is balancing both sides of the classification problem.

### Confusion Matrix

A confusion matrix shows where the model is correct and where it gets confused.

For the voice model, a binary confusion matrix compares:

- actual `not_stressed` vs predicted `not_stressed`
- actual `not_stressed` vs predicted `stressed`
- actual `stressed` vs predicted `not_stressed`
- actual `stressed` vs predicted `stressed`

For the face model, the confusion matrix shows whether the model confuses emotions such as `sad` and `neutral`, or `angry` and `sad`.

This is useful because two models can have similar accuracy but very different error patterns.

## 10. Why the Combined System Is Stronger

The face model and voice model are not duplicates. They observe different types of human behavior:

| Signal | What It Captures | Common Weakness |
| --- | --- | --- |
| Face | expression, visual tension, visible emotional state | affected by lighting, camera angle, face position |
| Voice | tone, energy, speech pattern, vocal tension | affected by background noise, microphone quality, silence |

By combining both, MindPulse becomes more robust. If the webcam signal is weak, the voice branch can still contribute. If the microphone signal is noisy, the face branch can still contribute. This is the central advantage of multimodal AI.

The combination does not automatically mean the mathematical accuracy is higher unless a paired multimodal test set is available. However, it does improve practical reliability because the final decision is not dependent on only one sensor.

## 11. Why There Is No Single Overall Fusion Accuracy Yet

The face and voice models were evaluated separately because the datasets are separate. A true overall fusion accuracy requires paired validation samples like:

```text
data/raw/fusion_eval/sample_001/
  face.jpg
  voice.wav
  label.txt
```

The `label.txt` file must contain the real stress level for that exact face and voice pair:

```text
low
medium
high
```

Without this paired dataset, it would be misleading to report one final combined accuracy number. The correct academic statement is:

```text
The face branch achieved 81.7% accuracy and the voice branch achieved 72.5% accuracy on their respective held-out test sets. The late-fusion layer combines both branches for more robust stress-level prediction, but true combined accuracy requires paired multimodal validation data.
```

## 12. Final Interpretation

The current results are strong enough for a final-year prototype because both individual branches now exceed 70% accuracy, and the project includes the full end-to-end system:

- dataset preparation
- preprocessing
- local model training
- branch evaluation
- saved TensorFlow/Keras model artifacts
- late fusion
- live webcam and microphone inference
- backend API storage
- user dashboard

The project should be presented as a practical AI prototype, not as a clinical or medical diagnostic tool. Its value is that it demonstrates how multiple local signals can be combined into a usable stress-checking system with transparent training and evaluation.
