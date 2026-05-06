import _bootstrap  # noqa: F401

import argparse

from training.voice.evaluate import evaluate_voice_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the voice emotion or stress tendency model.")
    parser.add_argument("--demo", action="store_true", help="Use the synthetic demo dataset.")
    parser.add_argument(
        "--max-files-per-class",
        type=int,
        default=1000,
        help="Optional cap per class for evaluation. Use 0 or omit for all files.",
    )
    parser.add_argument(
        "--label-mode",
        choices=["emotion", "stress"],
        default="stress",
        help="Use original emotion folders or collapse them into calm/neutral/stressed labels.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    max_files = None if not args.max_files_per_class or args.max_files_per_class <= 0 else args.max_files_per_class
    evaluate_voice_model(demo_mode=args.demo, max_files_per_class=max_files, label_mode=args.label_mode)
