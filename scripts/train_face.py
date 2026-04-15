import _bootstrap  # noqa: F401

import argparse

from training.face.train import train_face_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the face emotion model.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of frozen-backbone training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--demo", action="store_true", help="Use the synthetic demo dataset.")
    parser.add_argument(
        "--max-files-per-class",
        type=int,
        default=750,
        help="Optional cap per class for faster balanced training. Use 0 or omit for all files.",
    )
    parser.add_argument("--fine-tune-epochs", type=int, default=4, help="Extra epochs after unfreezing top layers.")
    parser.add_argument("--fine-tune-layers", type=int, default=60, help="How many backbone layers to unfreeze.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    max_files = None if not args.max_files_per_class or args.max_files_per_class <= 0 else args.max_files_per_class
    train_face_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        demo_mode=args.demo,
        max_files_per_class=max_files,
        fine_tune_epochs=args.fine_tune_epochs,
        fine_tune_layers=args.fine_tune_layers,
    )
