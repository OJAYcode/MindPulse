from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

FACE_LABEL_MAP = {
    "anger": "angry",
    "angry": "angry",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
}

IEMOCAP_TO_PROJECT_LABEL = {
    "ang": "stressed",
    "dis": "stressed",
    "fea": "stressed",
    "fru": "stressed",
    "hap": "happy",
    "exc": "happy",
    "neu": "neutral",
    "sad": "sad",
}

ANNOTATION_PATTERN = re.compile(r"^\[[^\]]+\]\s+(?P<utterance>\S+)\s+(?P<label>[a-zA-Z]{3})\s+\[")


def _safe_copy_or_link(source: Path, destination: Path, dry_run: bool = False) -> bool:
    if destination.exists():
        return False
    if dry_run:
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return True


def _unique_destination(target_dir: Path, prefix: str, source: Path) -> Path:
    clean_stem = re.sub(r"[^a-zA-Z0-9_.-]+", "_", source.stem)
    return target_dir / f"{prefix}_{clean_stem}{source.suffix.lower()}"


def prepare_affectnet_faces(source_root: Path, target_root: Path, dry_run: bool = False) -> dict:
    counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()

    for split_dir in ("Train", "Test"):
        split_path = source_root / split_dir
        if not split_path.exists():
            continue
        for source_label_dir in sorted(item for item in split_path.iterdir() if item.is_dir()):
            target_label = FACE_LABEL_MAP.get(source_label_dir.name.lower())
            if target_label is None:
                skipped[source_label_dir.name] += len(
                    [file_path for file_path in source_label_dir.rglob("*") if file_path.suffix.lower() in IMAGE_EXTENSIONS]
                )
                continue
            for image_path in source_label_dir.rglob("*"):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                destination = _unique_destination(
                    target_root / target_label,
                    f"affectnet_{split_dir.lower()}",
                    image_path,
                )
                if _safe_copy_or_link(image_path, destination, dry_run=dry_run):
                    counts[target_label] += 1

    return {
        "source": str(source_root),
        "target": str(target_root),
        "added_or_existing_ready": dict(counts),
        "skipped_labels": dict(skipped),
    }


def _load_iemocap_annotations(source_root: Path) -> dict[str, str]:
    annotations: dict[str, str] = {}
    for annotation_path in source_root.rglob("dialog/EmoEvaluation/*.txt"):
        for line in annotation_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            match = ANNOTATION_PATTERN.match(line.strip())
            if not match:
                continue
            utterance_id = match.group("utterance")
            raw_label = match.group("label").lower()
            target_label = IEMOCAP_TO_PROJECT_LABEL.get(raw_label)
            if target_label is not None:
                annotations[utterance_id] = target_label
    return annotations


def prepare_iemocap_voice(source_root: Path, target_root: Path, dry_run: bool = False) -> dict:
    annotations = _load_iemocap_annotations(source_root)
    counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()

    for audio_path in source_root.rglob("*"):
        if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        target_label = annotations.get(audio_path.stem)
        if target_label is None:
            skipped["missing_or_unsupported_label"] += 1
            continue
        destination = _unique_destination(target_root / target_label, "iemocap", audio_path)
        if _safe_copy_or_link(audio_path, destination, dry_run=dry_run):
            counts[target_label] += 1

    return {
        "source": str(source_root),
        "target": str(target_root),
        "annotations_found": len(annotations),
        "added_or_existing_ready": dict(counts),
        "skipped": dict(skipped),
        "label_mapping": IEMOCAP_TO_PROJECT_LABEL,
    }


def _count_target_files(root: Path, extensions: set[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not root.exists():
        return counts
    for label_dir in sorted(item for item in root.iterdir() if item.is_dir()):
        counts[label_dir.name] = sum(1 for file_path in label_dir.rglob("*") if file_path.suffix.lower() in extensions)
    return counts


def prepare_new_datasets(
    face_source: Path,
    voice_source: Path,
    face_target: Path,
    voice_target: Path,
    manifest_path: Path,
    dry_run: bool = False,
) -> dict:
    report = {
        "dry_run": dry_run,
        "face_before": _count_target_files(face_target, IMAGE_EXTENSIONS),
        "voice_before": _count_target_files(voice_target, AUDIO_EXTENSIONS),
        "face_prepare": prepare_affectnet_faces(face_source, face_target, dry_run=dry_run),
        "voice_prepare": prepare_iemocap_voice(voice_source, voice_target, dry_run=dry_run),
        "face_after": _count_target_files(face_target, IMAGE_EXTENSIONS) if not dry_run else {},
        "voice_after": _count_target_files(voice_target, AUDIO_EXTENSIONS) if not dry_run else {},
    }
    if not dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare newly added AffectNet and IEMOCAP datasets for training.")
    parser.add_argument("--face-source", type=Path, default=Path("AffectNet new Facedataset"))
    parser.add_argument("--voice-source", type=Path, default=Path("IEMOCAP New Voicedatasets"))
    parser.add_argument("--face-target", type=Path, default=Path("data/raw/face"))
    parser.add_argument("--voice-target", type=Path, default=Path("data/raw/voice"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/dataset_prepare_report.json"))
    parser.add_argument("--dry-run", action="store_true", help="Report what would be prepared without copying/linking files.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = prepare_new_datasets(
        face_source=args.face_source,
        voice_source=args.voice_source,
        face_target=args.face_target,
        voice_target=args.voice_target,
        manifest_path=args.manifest,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))
