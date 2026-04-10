"""
Batch CSR watermarking and extraction for text images using
"Test 4: document (with diagonal compensation)" workflow.
"""

import argparse
import csv
import os
from typing import List, Tuple

import cv2
import numpy as np

from csr_watermark import CSRWatermark, bits_to_message, compute_ber, message_to_bits


def collect_images(input_dir: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = []
    for name in os.listdir(input_dir):
        if name.lower().endswith(exts):
            files.append(os.path.join(input_dir, name))
    files.sort()
    return files


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_payload_bits(message: str, capacity: int) -> np.ndarray:
    bits = message_to_bits(message)
    if len(bits) < capacity:
        bits = np.concatenate([bits, np.zeros(capacity - len(bits), dtype=np.int8)])
    else:
        bits = bits[:capacity]
    return bits


def run_batch(
    input_dir: str,
    output_dir: str,
    message: str,
    embedding_strength: float,
    save_extracted_preview: bool,
) -> Tuple[int, float, float]:
    csrw = CSRWatermark(embedding_strength=embedding_strength)
    original_bits = build_payload_bits(message, csrw.capacity)

    watermarked_dir = os.path.join(output_dir, "watermarked")
    preview_dir = os.path.join(output_dir, "extracted_preview")
    ensure_dir(output_dir)
    ensure_dir(watermarked_dir)
    if save_extracted_preview:
        ensure_dir(preview_dir)

    image_paths = collect_images(input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {input_dir}")

    csv_path = os.path.join(output_dir, "extract_report.csv")
    total_errors = 0
    total_bits = 0
    accuracy_list = []

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image",
            "bit_errors",
            "bit_length",
            "ber",
            "accuracy_percent",
            "decoded_message",
        ])

        for idx, image_path in enumerate(image_paths, start=1):
            image_name = os.path.basename(image_path)
            image = cv2.imread(image_path)
            if image is None:
                print(f"[{idx}/{len(image_paths)}] Skip unreadable image: {image_name}")
                continue

            # Test 4 workflow: apply watermark to document and extract with diagonal compensation.
            watermarked_doc, _, _ = csrw.apply_to_document(image, original_bits)

            watermarked_path = os.path.join(watermarked_dir, image_name)
            cv2.imwrite(watermarked_path, watermarked_doc)

            loaded_doc = cv2.imread(watermarked_path)
            extracted_bits, _ = csrw.extract_from_document(loaded_doc, avoid_text=True)

            ber, errors = compute_ber(original_bits, extracted_bits)
            decoded = bits_to_message(extracted_bits)
            valid_len = min(len(original_bits), len(extracted_bits))
            accuracy = (1.0 - ber) * 100.0 if valid_len > 0 else 0.0

            total_errors += errors
            total_bits += valid_len
            accuracy_list.append(accuracy)

            if save_extracted_preview:
                # Save the resized extraction area for quick inspection.
                gray = cv2.cvtColor(loaded_doc, cv2.COLOR_BGR2GRAY)
                fh = csrw.unit_height * 2
                fw = csrw.unit_width * 2
                region = gray[:fh, :fw]
                preview = region[: fh // 2, : fw // 2]
                cv2.imwrite(os.path.join(preview_dir, image_name), preview)

            writer.writerow([
                image_name,
                errors,
                valid_len,
                f"{ber:.6f}",
                f"{accuracy:.2f}",
                decoded,
            ])

            print(
                f"[{idx}/{len(image_paths)}] {image_name}: errors={errors}/{valid_len}, "
                f"BER={ber:.6f}, acc={accuracy:.2f}%"
            )

    avg_ber = (total_errors / total_bits) if total_bits > 0 else 1.0
    avg_acc = (sum(accuracy_list) / len(accuracy_list)) if accuracy_list else 0.0

    print("\nBatch finished")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Processed images: {len(accuracy_list)} / {len(image_paths)}")
    print(f"Total errors: {total_errors} / {total_bits}")
    print(f"Average BER: {avg_ber:.6f}")
    print(f"Average Accuracy: {avg_acc:.2f}%")
    print(f"CSV report: {csv_path}")

    return len(accuracy_list), avg_ber, avg_acc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch CSR watermarking/extraction for ti_a×b/ti using Test 4 method."
    )
    parser.add_argument(
        "--input-dir",
        default="results/eval_watermark/ti_a×b/ti",
        help="Input folder containing text images.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/eval_watermark/ti_a×b/ti_csr",
        help="Output folder for watermarked images and extraction report.",
    )
    parser.add_argument(
        "--message",
        default="3354421163338888",
        help="Watermark message string.",
    )
    parser.add_argument(
        "--embedding-strength",
        type=float,
        default=30.0,
        help="DCT embedding strength r.",
    )
    parser.add_argument(
        "--save-extracted-preview",
        action="store_true",
        help="Whether to save extracted watermark region previews.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        message=args.message,
        embedding_strength=args.embedding_strength,
        save_extracted_preview=args.save_extracted_preview,
    )


if __name__ == "__main__":
    main()
