#!/usr/bin/env python3
"""Batch convert documents to specified format."""

import argparse
from pathlib import Path
from typing import List

from vertector_data_ingestion import (
    UniversalConverter,
    ConverterConfig,
    LocalMpsConfig,
    CloudGpuConfig,
    CloudCpuConfig,
    ExportFormat,
    setup_logging,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch convert documents using Vertector"
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory containing documents"
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for converted files"
    )

    parser.add_argument(
        "--format",
        choices=["markdown", "json", "doctags"],
        default="markdown",
        help="Output format (default: markdown)"
    )

    parser.add_argument(
        "--pattern",
        default="*.*",
        help="File pattern to match (default: *.*)"
    )

    parser.add_argument(
        "--preset",
        choices=["local-mps", "cloud-gpu", "cloud-cpu", "default"],
        default="default",
        help="Configuration preset (default: default)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        help="Number of parallel workers (overrides preset)"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process directories recursively"
    )

    return parser.parse_args()


def get_config(preset: str, workers: int = None) -> ConverterConfig:
    """Get configuration based on preset."""
    if preset == "local-mps":
        config = LocalMpsConfig()
    elif preset == "cloud-gpu":
        config = CloudGpuConfig()
    elif preset == "cloud-cpu":
        config = CloudCpuConfig()
    else:
        config = ConverterConfig()

    if workers:
        config.batch_processing_workers = workers

    return config


def find_documents(
    input_dir: Path,
    pattern: str,
    recursive: bool
) -> List[Path]:
    """Find documents matching pattern."""
    if recursive:
        return list(input_dir.rglob(pattern))
    else:
        return list(input_dir.glob(pattern))


def main():
    """Main conversion function."""
    args = parse_args()

    # Setup
    setup_logging(log_level=args.log_level)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Get configuration
    config = get_config(args.preset, args.workers)

    # Find documents
    documents = find_documents(args.input_dir, args.pattern, args.recursive)

    if not documents:
        print(f"No documents found matching '{args.pattern}' in {args.input_dir}")
        return

    print(f"Found {len(documents)} documents to convert")
    print(f"Using preset: {args.preset}")
    print(f"Workers: {config.batch_processing_workers}")
    print(f"Output format: {args.format}")

    # Convert format string to enum
    format_map = {
        "markdown": ExportFormat.MARKDOWN,
        "json": ExportFormat.JSON,
        "doctags": ExportFormat.DOCTAGS,
    }
    export_format = format_map[args.format]

    # Initialize converter
    converter = UniversalConverter(config)

    # Process documents
    successful = 0
    failed = 0

    for doc_path in documents:
        try:
            print(f"\nProcessing: {doc_path.name}")

            # Determine output path
            if args.recursive:
                # Preserve directory structure
                rel_path = doc_path.relative_to(args.input_dir)
                output_subdir = args.output_dir / rel_path.parent
                output_subdir.mkdir(parents=True, exist_ok=True)
                output_path = output_subdir / f"{doc_path.stem}.{args.format}"
            else:
                output_path = args.output_dir / f"{doc_path.stem}.{args.format}"

            # Convert and export
            converter.convert_and_export(
                source=doc_path,
                output_path=output_path,
                format=export_format
            )

            print(f"  ✓ Saved to: {output_path}")
            successful += 1

        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed += 1

    # Summary
    print("\n" + "=" * 60)
    print("Conversion Complete")
    print("=" * 60)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(documents)}")


if __name__ == "__main__":
    main()
