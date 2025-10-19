"""
Clean all generated files and folders.
This script removes all outputs, models, and interim data while preserving source code.

Usage:
    python clean_generated.py --all              # Clean everything
    python clean_generated.py --interim          # Clean only interim data
    python clean_generated.py --results          # Clean only results
    python clean_generated.py --models           # Clean only models
    python clean_generated.py --dry-run          # Show what would be deleted
"""

import os
import shutil
from pathlib import Path
import argparse


class CleanupManager:
    def __init__(self, base_dir='.', dry_run=False):
        self.base_dir = Path(base_dir)
        self.dry_run = dry_run
        self.total_size = 0
        self.file_count = 0
        self.dir_count = 0

    def get_size(self, path):
        """Get size of file or directory in bytes."""
        if path.is_file():
            return path.stat().st_size
        elif path.is_dir():
            total = 0
            try:
                for item in path.rglob('*'):
                    if item.is_file():
                        total += item.stat().st_size
            except:
                pass
            return total
        return 0

    def format_size(self, size_bytes):
        """Convert bytes to human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"

    def remove_path(self, path, description):
        """Remove a file or directory."""
        if not path.exists():
            return

        size = self.get_size(path)
        self.total_size += size

        if path.is_file():
            self.file_count += 1
            if self.dry_run:
                print(f"[DRY-RUN] Would delete file: {path} ({self.format_size(size)})")
            else:
                try:
                    path.unlink()
                    print(f"[DELETED] {description}: {path} ({self.format_size(size)})")
                except Exception as e:
                    print(f"[ERROR] Failed to delete {path}: {e}")

        elif path.is_dir():
            file_count = len([f for f in path.rglob('*') if f.is_file()])
            self.dir_count += 1
            self.file_count += file_count

            if self.dry_run:
                print(f"[DRY-RUN] Would delete folder: {path} ({file_count} files, {self.format_size(size)})")
            else:
                try:
                    shutil.rmtree(path)
                    print(f"[DELETED] {description}: {path} ({file_count} files, {self.format_size(size)})")
                except Exception as e:
                    print(f"[ERROR] Failed to delete {path}: {e}")

    def clean_interim_data(self):
        """Clean interim processed data under Dhan-Shomadhan/."""
        print("\n" + "="*80)
        print("CLEANING INTERIM DATA")
        print("="*80 + "\n")

        interim_paths = [
            (self.base_dir / 'Dhan-Shomadhan' / 'processed_data', 'Interim processed data'),
            (self.base_dir / 'dataset', 'Old dataset folder (if exists)'),
            (self.base_dir / 'dataset_test', 'Test dataset folder'),
        ]

        for path, desc in interim_paths:
            self.remove_path(path, desc)

        # Clean any dataset_run_* folders
        for path in self.base_dir.glob('dataset_run_*'):
            self.remove_path(path, f'Experiment dataset folder')

    def clean_results(self):
        """Clean all result folders and files."""
        print("\n" + "="*80)
        print("CLEANING RESULTS")
        print("="*80 + "\n")

        results_paths = [
            (self.base_dir / 'results', 'Results folder'),
            (self.base_dir / 'runs', 'YOLO runs folder'),
            (self.base_dir / 'outputs', 'Outputs folder'),
        ]

        for path, desc in results_paths:
            self.remove_path(path, desc)

    def clean_models(self):
        """Clean all model files."""
        print("\n" + "="*80)
        print("CLEANING MODELS")
        print("="*80 + "\n")

        model_paths = [
            (self.base_dir / 'models', 'Models folder'),
        ]

        for path, desc in model_paths:
            self.remove_path(path, desc)

        # Clean individual .pt files in root
        for pt_file in self.base_dir.glob('*.pt'):
            self.remove_path(pt_file, 'Model weight file')

    def clean_cache(self):
        """Clean Python cache files."""
        print("\n" + "="*80)
        print("CLEANING CACHE")
        print("="*80 + "\n")

        # __pycache__ folders
        for pycache in self.base_dir.rglob('__pycache__'):
            self.remove_path(pycache, 'Python cache folder')

        # .pyc files
        for pyc_file in self.base_dir.rglob('*.pyc'):
            self.remove_path(pyc_file, 'Python compiled file')

        # .pyo files
        for pyo_file in self.base_dir.rglob('*.pyo'):
            self.remove_path(pyo_file, 'Python optimized file')

    def clean_logs(self):
        """Clean log files."""
        print("\n" + "="*80)
        print("CLEANING LOGS")
        print("="*80 + "\n")

        for log_file in self.base_dir.glob('*.log'):
            self.remove_path(log_file, 'Log file')

    def clean_all(self):
        """Clean everything."""
        print("\n" + "="*80)
        print("CLEANING ALL GENERATED FILES")
        print("="*80)

        if self.dry_run:
            print("\n[DRY-RUN MODE] No files will actually be deleted\n")

        self.clean_interim_data()
        self.clean_results()
        self.clean_models()
        self.clean_cache()
        self.clean_logs()

        self.print_summary()

    def print_summary(self):
        """Print summary of cleanup."""
        print("\n" + "="*80)
        print("CLEANUP SUMMARY")
        print("="*80 + "\n")

        if self.dry_run:
            print("[DRY-RUN] No files were actually deleted")
        else:
            print("[COMPLETED] Cleanup finished")

        print(f"\nDirectories removed: {self.dir_count}")
        print(f"Files removed: {self.file_count}")
        print(f"Space freed: {self.format_size(self.total_size)}")

        if self.dry_run:
            print("\nRun without --dry-run to actually delete these files")

    def list_generated_files(self):
        """List all generated files without deleting."""
        print("\n" + "="*80)
        print("GENERATED FILES AND FOLDERS")
        print("="*80 + "\n")

        categories = {
            'Interim Data': [
                self.base_dir / 'Dhan-Shomadhan' / 'processed_data',
                self.base_dir / 'dataset',
                self.base_dir / 'dataset_test',
            ],
            'Results': [
                self.base_dir / 'results',
                self.base_dir / 'runs',
                self.base_dir / 'outputs',
            ],
            'Models': [
                self.base_dir / 'models',
            ],
            'Cache': [
                self.base_dir / '__pycache__',
            ]
        }

        total_size = 0

        for category, paths in categories.items():
            print(f"\n{category}:")
            print("-" * 60)

            cat_size = 0
            cat_found = False

            for path in paths:
                if path.exists():
                    cat_found = True
                    size = self.get_size(path)
                    cat_size += size

                    if path.is_dir():
                        file_count = len([f for f in path.rglob('*') if f.is_file()])
                        print(f"  {path} ({file_count} files, {self.format_size(size)})")
                    else:
                        print(f"  {path} ({self.format_size(size)})")

            # Check for pattern-based paths
            if category == 'Interim Data':
                for run_folder in self.base_dir.glob('dataset_run_*'):
                    cat_found = True
                    size = self.get_size(run_folder)
                    cat_size += size
                    file_count = len([f for f in run_folder.rglob('*') if f.is_file()])
                    print(f"  {run_folder} ({file_count} files, {self.format_size(size)})")

            if category == 'Cache':
                for pycache in self.base_dir.rglob('__pycache__'):
                    cat_found = True
                    size = self.get_size(pycache)
                    cat_size += size
                    print(f"  {pycache} ({self.format_size(size)})")

            if category == 'Models':
                for pt_file in self.base_dir.glob('*.pt'):
                    cat_found = True
                    size = self.get_size(pt_file)
                    cat_size += size
                    print(f"  {pt_file} ({self.format_size(size)})")

            if not cat_found:
                print("  (none found)")
            else:
                total_size += cat_size
                print(f"  Subtotal: {self.format_size(cat_size)}")

        print("\n" + "="*80)
        print(f"Total size: {self.format_size(total_size)}")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Clean all generated files and folders',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  List all generated files:
    python clean_generated.py --list

  Dry-run (see what would be deleted):
    python clean_generated.py --all --dry-run

  Clean everything:
    python clean_generated.py --all

  Clean only interim data:
    python clean_generated.py --interim

  Clean only results:
    python clean_generated.py --results

  Clean only models:
    python clean_generated.py --models

  Clean cache files:
    python clean_generated.py --cache

  Clean specific items:
    python clean_generated.py --interim --cache --dry-run
        """
    )

    parser.add_argument('--all', action='store_true',
                       help='Clean all generated files')
    parser.add_argument('--interim', action='store_true',
                       help='Clean interim data (processed_data folder)')
    parser.add_argument('--results', action='store_true',
                       help='Clean results folder')
    parser.add_argument('--models', action='store_true',
                       help='Clean models folder')
    parser.add_argument('--cache', action='store_true',
                       help='Clean Python cache files')
    parser.add_argument('--logs', action='store_true',
                       help='Clean log files')
    parser.add_argument('--list', action='store_true',
                       help='List all generated files without deleting')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be deleted without actually deleting')
    parser.add_argument('--base-dir', type=str, default='.',
                       help='Base directory (default: current directory)')

    args = parser.parse_args()

    cleaner = CleanupManager(base_dir=args.base_dir, dry_run=args.dry_run)

    # If --list is specified, just list and exit
    if args.list:
        cleaner.list_generated_files()
        return

    # If no specific option, show help
    if not any([args.all, args.interim, args.results, args.models, args.cache, args.logs]):
        parser.print_help()
        print("\n" + "="*80)
        print("DEFAULT: Listing all generated files")
        print("Use --all to clean everything, or specific options to clean selected items")
        print("="*80)
        cleaner.list_generated_files()
        return

    # Confirm if not dry-run
    if not args.dry_run and not args.list:
        print("\n" + "="*80)
        print("WARNING: This will permanently delete files!")
        print("="*80 + "\n")

        print("Items to be cleaned:")
        if args.all:
            print("  - All interim data")
            print("  - All results")
            print("  - All models")
            print("  - All cache files")
            print("  - All log files")
        else:
            if args.interim:
                print("  - Interim data")
            if args.results:
                print("  - Results")
            if args.models:
                print("  - Models")
            if args.cache:
                print("  - Cache files")
            if args.logs:
                print("  - Log files")

        response = input("\nContinue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("\nCancelled.")
            return

    # Perform cleanup
    if args.all:
        cleaner.clean_all()
    else:
        if args.interim:
            cleaner.clean_interim_data()
        if args.results:
            cleaner.clean_results()
        if args.models:
            cleaner.clean_models()
        if args.cache:
            cleaner.clean_cache()
        if args.logs:
            cleaner.clean_logs()

        if any([args.interim, args.results, args.models, args.cache, args.logs]):
            cleaner.print_summary()


if __name__ == '__main__':
    main()
