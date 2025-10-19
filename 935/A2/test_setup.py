import sys
from pathlib import Path


def test_imports():
    print("Testing package imports...")
    print("-" * 60)

    packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('ultralytics', 'Ultralytics YOLO'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('sklearn', 'Scikit-learn'),
        ('PIL', 'Pillow'),
        ('yaml', 'PyYAML')
    ]

    failed = []

    for module_name, display_name in packages:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"[OK] {display_name:20s} - version {version}")
        except ImportError as e:
            print(f"[FAIL] {display_name:20s} - FAILED")
            failed.append((display_name, str(e)))

    print("-" * 60)

    if failed:
        print("\nFailed imports:")
        for name, error in failed:
            print(f"  {name}: {error}")
        return False

    return True


def test_cuda():
    print("\nTesting CUDA availability...")
    print("-" * 60)

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")

        if cuda_available:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("No CUDA GPU detected. Training will use CPU (slower).")

        print("-" * 60)
        return True

    except Exception as e:
        print(f"Error checking CUDA: {e}")
        print("-" * 60)
        return False


def test_dataset():
    print("\nTesting dataset structure...")
    print("-" * 60)

    dataset_dir = Path('./Dhan-Shomadhan')

    if not dataset_dir.exists():
        print(f"[FAIL] Dataset directory not found: {dataset_dir}")
        print("  Please ensure the dataset is in the correct location.")
        print("-" * 60)
        return False

    print(f"[OK] Dataset directory found: {dataset_dir}")

    expected_dirs = [
        'Field Background',
        'White Background'
    ]

    for dir_name in expected_dirs:
        dir_path = dataset_dir / dir_name
        if dir_path.exists():
            n_subdirs = len(list(dir_path.iterdir()))
            print(f"[OK] {dir_name:20s} - {n_subdirs} disease classes")
        else:
            print(f"[FAIL] {dir_name:20s} - NOT FOUND")

    print("-" * 60)
    return True


def test_modules():
    print("\nTesting project modules...")
    print("-" * 60)

    modules = [
        'config',
        'data_preprocessor',
        'trainer',
        'evaluator',
        'inference',
        'feature_visualizer',
        'main'
    ]

    failed = []

    for module_name in modules:
        try:
            __import__(module_name)
            print(f"[OK] {module_name}.py")
        except Exception as e:
            print(f"[FAIL] {module_name}.py - FAILED: {e}")
            failed.append((module_name, str(e)))

    print("-" * 60)

    if failed:
        print("\nFailed module imports:")
        for name, error in failed:
            print(f"  {name}: {error}")
        return False

    return True


def test_config():
    print("\nTesting configuration...")
    print("-" * 60)

    try:
        from config import Config

        print(f"[OK] Base directory: {Config.BASE_DIR}")
        print(f"[OK] Dataset directory: {Config.DATASET_DIR}")
        print(f"[OK] Device: {Config.DEVICE}")
        print(f"[OK] Class names: {', '.join(Config.CLASS_NAMES)}")

        Config.ensure_directories()
        print(f"[OK] Directories created")

        print("-" * 60)
        return True

    except Exception as e:
        print(f"[FAIL] Configuration test failed: {e}")
        print("-" * 60)
        return False


def main():
    print("\n" + "=" * 60)
    print("Rice Disease Classification System - Setup Test")
    print("=" * 60 + "\n")

    results = []

    results.append(("Package imports", test_imports()))
    results.append(("CUDA support", test_cuda()))
    results.append(("Dataset structure", test_dataset()))
    results.append(("Project modules", test_modules()))
    results.append(("Configuration", test_config()))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "[PASSED]" if passed else "[FAILED]"
        print(f"{test_name:20s} - {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n[OK] All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. Prepare dataset:  python main.py prepare")
        print("  2. Train model:      python main.py train --device cuda")
        print("  3. Evaluate:         python main.py evaluate --model ./models/best_model.pt --scenarios")
        return 0
    else:
        print("\n[FAIL] Some tests failed. Please fix the issues above.")
        print("\nCommon solutions:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Ensure dataset is in ./Dhan-Shomadhan/ directory")
        print("  - Check Python version (requires 3.12)")
        return 1


if __name__ == '__main__':
    sys.exit(main())
