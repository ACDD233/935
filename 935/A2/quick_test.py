"""
Quick test script to verify the system works end-to-end with a small dataset subset.
This is for testing only - not for production use.
"""

import os
import sys
from pathlib import Path
import shutil


def quick_test():
    print("\n" + "="*80)
    print("QUICK SYSTEM TEST")
    print("="*80)
    print("\nThis will run a quick test with reduced epochs to verify everything works.")
    print("NOT for final results - just for testing!\n")

    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Test cancelled.")
        return

    from config import Config

    print("\n[1/5] Testing imports...")
    try:
        from data_preprocessor import prepare_dataset_for_yolo
        from trainer import train_model
        from evaluator import ModelEvaluator
        from inference import predict
        from feature_visualizer import visualize_features
        print("✓ All imports successful")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return

    print("\n[2/5] Preparing small dataset...")
    try:
        if not Config.DATASET_DIR.exists():
            print(f"✗ Dataset not found at {Config.DATASET_DIR}")
            print("  Please ensure Dhan-Shomadhan folder exists in current directory")
            return

        splits, stats = prepare_dataset_for_yolo(
            source_dir=str(Config.DATASET_DIR),
            output_dir='./dataset_test',
            seed=42
        )
        print(f"✓ Dataset prepared: {stats['total']} images")
    except Exception as e:
        print(f"✗ Dataset preparation failed: {e}")
        return

    print("\n[3/5] Training model (10 epochs - test only)...")
    try:
        original_processed_dir = Config.PROCESSED_DIR
        Config.PROCESSED_DIR = Path('./dataset_test')

        trainer, results = train_model(
            epochs=10,
            batch_size=8,
            device='cuda' if 'cuda' in Config.DEVICE else 'cpu'
        )

        best_model = trainer.get_best_model_path()
        print(f"✓ Training completed: {best_model}")

    except Exception as e:
        print(f"✗ Training failed: {e}")
        Config.PROCESSED_DIR = original_processed_dir
        return

    print("\n[4/5] Testing inference...")
    try:
        test_dir = Config.PROCESSED_DIR / 'test'
        test_image = None

        for class_dir in test_dir.iterdir():
            if class_dir.is_dir():
                for img in class_dir.glob('*.jpg'):
                    test_image = img
                    break
                if test_image:
                    break

        if test_image:
            pred = predict(
                model_path=str(best_model),
                source=str(test_image),
                device='cpu',
                save_results=False
            )
            print(f"✓ Inference successful: predicted {pred['class_name']} with {pred['confidence']:.3f} confidence")
        else:
            print("⚠ No test image found for inference test")

    except Exception as e:
        print(f"✗ Inference failed: {e}")

    print("\n[5/5] Testing evaluation...")
    try:
        evaluator = ModelEvaluator(str(best_model), device='cpu')
        metrics = evaluator.evaluate_on_split('test', save_plots=False)
        print(f"✓ Evaluation successful: accuracy = {metrics.get('accuracy', 0):.3f}")

    except Exception as e:
        print(f"✗ Evaluation failed: {e}")

    Config.PROCESSED_DIR = original_processed_dir

    print("\n" + "="*80)
    print("QUICK TEST COMPLETED")
    print("="*80)
    print("\nNote: This was a quick test with only 10 epochs.")
    print("For actual results, use:")
    print("  python main.py train --epochs 100 --device cuda")
    print("\nClean up test files:")
    print("  rm -rf ./dataset_test")

    cleanup = input("\nDelete test files now? (y/n): ")
    if cleanup.lower() == 'y':
        try:
            shutil.rmtree('./dataset_test')
            print("✓ Test files deleted")
        except Exception as e:
            print(f"⚠ Could not delete test files: {e}")


if __name__ == '__main__':
    try:
        quick_test()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
