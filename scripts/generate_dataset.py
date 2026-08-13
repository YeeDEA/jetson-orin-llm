"""CLI wrapper for the synthetic robot-command dataset generator.

Same flow as notebooks/01-dataset-synthesis.ipynb.

Usage:
    python scripts/generate_dataset.py --count 50000 --output robot_dataset.json
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jetson_llm.data import generate_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic NL-command -> robot-code dataset")
    parser.add_argument("--count", type=int, default=50000, help="number of samples (notebook default: 50000)")
    parser.add_argument("--output", default="robot_dataset.json", help="output JSON path")
    args = parser.parse_args()
    generate_dataset(target_count=args.count, output_file=args.output)


if __name__ == "__main__":
    main()
