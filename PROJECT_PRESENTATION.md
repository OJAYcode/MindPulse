# Multimodal Stress Detection Project Notes

## Title

Local Multimodal Stress Detection Using Laptop Webcam and Microphone

## Problem Statement

The goal of this project is to estimate a user's stress level from two locally available signals:

- facial expression from a laptop webcam
- vocal emotion or stress tendency from a laptop microphone

The system is designed to run fully offline for model inference, avoid cloud AI APIs, and remain practical on a normal laptop.

## System Architecture

1. Face branch
- input: cropped face image
- detector: MediaPipe face detection
- model: MobileNetV2 transfer learning classifier
- output: facial emotion probabilities

2. Voice branch
- input: short microphone recording
- preprocessing: mel-spectrogram generation
- model: small CNN classifier
- output: voice emotion or stress-tendency probabilities

3. Fusion branch
- input: face probabilities + voice probabilities
- main method: rule-based late fusion
- optional method: logistic-regression fusion model
- output: final stress level `low`, `medium`, or `high`

4. Backend
- FastAPI REST API
- SQLite database
- stores timestamped inference results for dashboards and history views

## Why It Is Multimodal

This is still a multimodal system because:

- it uses two different input modalities, image and audio
- each modality is processed by a separate trained model
- the final stress prediction depends on combining both outputs

The system is not just a face model or just a voice model. The stress decision is produced after fusion.

## Dataset Strategy

The project supports separate datasets for face and voice training.

Face:
- trained on merged folder-per-emotion datasets
- active classes: `angry`, `happy`, `neutral`, `sad`

Voice:
- trained on merged folder-per-emotion datasets
- active classes: `calm`, `happy`, `neutral`, `sad`, `stressed`

Important note:
- because the main face and voice datasets are separate, the official final method is rule-based late fusion
- the learned logistic fusion model is kept only as an experimental add-on

## Training Summary

Face training:
- MobileNetV2 backbone
- pretrained ImageNet weights
- frozen training followed by fine-tuning
- balanced capped subset used for practical laptop training

Voice training:
- mel-spectrogram preprocessing
- small CNN classifier
- cached processed tensors for faster reruns

## Current Results

Face branch:
- accuracy: `0.4650`
- precision: `0.5575`
- recall: `0.4650`
- F1: `0.4301`

Voice branch:
- accuracy: `0.3385`
- precision: `0.2093`
- recall: `0.3385`
- F1: `0.2548`

These are practical baseline results and are reported honestly without inflating performance.

## Final Presentation Configuration

Use this as the official story in the report and demo:

- face model: trained local MobileNetV2 branch
- voice model: trained local CNN branch
- fusion: rule-based late fusion
- backend: FastAPI + SQLite
- live input: laptop webcam + laptop microphone
- optional logistic fusion: experimental only

## Demo Flow

1. Start the API
- `python scripts/run_api.py`

2. Run live inference
- `python scripts/run_live_inference.py`

3. Safe validation path
- `python scripts/run_live_inference.py --demo --max-cycles 2`

4. Inspect API outputs
- `GET /health`
- `GET /latest-result`
- `GET /history`

## Example Output JSON

```json
{
  "timestamp": "2026-04-06T03:15:45.432447Z",
  "face_emotion": "sad",
  "face_confidence": 0.6,
  "voice_emotion": "stressed",
  "voice_confidence": 0.7,
  "stress_level": "high",
  "source": "laptop_webcam_mic"
}
```

## Limitations

- face and voice datasets are not fully geographically representative
- demographic composition is not explicitly labeled in the available dataset files
- separate branch datasets make true paired multimodal learning difficult
- model performance is moderate and should be presented as a baseline, not a clinical-grade system

## Future Work

- collect paired multimodal stress data from the same participants
- tune fusion with a paired calibration manifest
- improve face generalization with better data balance and augmentation
- improve voice robustness across microphones, accents, and background noise
- add a lightweight dashboard frontend for real-time monitoring
