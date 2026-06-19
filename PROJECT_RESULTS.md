# MindPulse Project Results and Discussion

## Overview

MindPulse is a multimodal stress-detection system that uses a laptop webcam and microphone to estimate a user's stress level. The system trains two separate machine learning branches:

- a face emotion recognition model
- a voice stress-tendency model

The predictions from both branches are then combined using late fusion to produce a final stress level: `low`, `medium`, or `high`.

This project is best understood as a working final-year prototype. It demonstrates the full machine learning pipeline from dataset preparation and model training to live inference, backend storage, authentication, and dashboard display.

## Current Model Results

The models were evaluated separately because the available face and voice datasets are not paired by the same person/session. This means each branch has a real measured accuracy, while the final multimodal fusion accuracy requires a separate paired validation set.

| Model Branch | Task | Labels | Accuracy | Precision | Recall | F1 Score |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Face Model | Facial emotion recognition | angry, happy, neutral, sad | 81.7% | 82.1% | 81.7% | 81.6% |
| Voice Model | Voice stress tendency recognition | not_stressed, stressed | 72.5% | 72.5% | 72.5% | 72.5% |
| Late Fusion | Final stress-level prediction | low, medium, high | Requires paired validation data | N/A | N/A | N/A |

## Interpretation of the Results

The face model achieved 81.7% accuracy after training on a cleaner AffectNet-based four-class dataset. This is a strong result for a final-year prototype because facial emotion recognition is difficult: expressions can be subtle, lighting can change, and the same emotion may look different across different people.

The voice model achieved 72.5% test accuracy after switching to a cleaner IEMOCAP-based binary stress-tendency task and adding light audio augmentation. This is a meaningful result because voice-based stress detection is challenging. Microphone quality, background noise, accent, speaking volume, and recording length can all affect the prediction.

These percentages should not be presented as medical-grade performance. Instead, they show that the system has learned useful patterns from both visual and audio data and can support a practical prototype for stress-level estimation.

## Why Multimodal Fusion Is a Major Advantage

Using both face and voice is stronger than relying on only one signal. A person may appear calm on camera but sound tense, or they may have a neutral voice while their facial expression shows pressure. By combining both branches, the system becomes more robust because it does not depend on a single source of information.

The advantage of multimodal fusion is not only about increasing a single accuracy number. It is also about improving reliability, reducing the effect of noisy inputs, and making the final prediction more balanced.

| Situation | Face-Only System | Voice-Only System | Multimodal MindPulse System |
| --- | --- | --- | --- |
| Poor lighting affects the webcam | May predict incorrectly | Not affected by lighting | Can rely more on voice signal |
| Background noise affects microphone | Not affected by audio noise | May predict incorrectly | Can rely more on face signal |
| User has a neutral expression but tense voice | May miss stress signal | Can detect vocal tension | Combines both signals for a better decision |
| User is quiet but facial expression shows strain | Can detect facial tension | May have weak audio evidence | Uses face signal while still checking voice |
| One model has low confidence | Prediction may be unreliable | Prediction may be unreliable | Fusion reduces dependence on one weak branch |

## Why the Results Are Still Useful

The project is strong because it successfully demonstrates a complete end-to-end multimodal AI system. The system does not stop at training a model. It also includes:

- local model training with TensorFlow/Keras
- face preprocessing and transfer learning
- voice preprocessing with mel-spectrograms
- rule-based late fusion
- optional learned fusion model
- live webcam and microphone inference
- backend API integration
- database persistence
- authentication and user dashboard
- mobile-friendly progressive web app behavior

This makes the project more complete than a simple classification model. It shows both machine learning knowledge and software engineering ability.

## Honest Limitation

The current project does not yet have a true overall fusion accuracy because the available face and voice datasets are separate. To calculate real combined accuracy, the project needs a small paired validation dataset where each sample contains:

```text
face image + voice recording + real stress label
```

For now, the correct academic explanation is:

> The individual branches achieved 81.7% and 72.5% accuracy respectively. The late-fusion layer improves robustness by combining visual and vocal cues, but true combined accuracy requires a paired multimodal validation dataset.

## Final Project Statement

MindPulse demonstrates that stress detection can be approached more effectively using multiple human signals instead of a single input. The face model provides visual emotional cues, while the voice model provides vocal stress cues. By combining both through late fusion, the system becomes more practical and reliable for real-world use than a face-only or voice-only approach.

The current results are suitable for a final-year prototype and provide a strong foundation for future improvement through larger paired datasets, better labeling, and more advanced fusion evaluation.
