import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

from training.face.evaluate import evaluate_face_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the face emotion model.")
    parser.add_argument("--demo", action="store_true", help="Use the synthetic demo dataset.")
    parser.add_argument(
        "--max-files-per-class",
        type=int,
        default=9000,
        help="Optional cap per class for evaluation. Use 0 or omit for all files.",
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Optional face dataset root to evaluate from.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    max_files = None if not args.max_files_per_class or args.max_files_per_class <= 0 else args.max_files_per_class
    evaluate_face_model(demo_mode=args.demo, max_files_per_class=max_files, data_dir=args.data_dir)
