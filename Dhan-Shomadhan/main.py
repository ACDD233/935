"""
Main controller for dataset splitting, training, testing, and visualization
"""
import os
import sys
from config import parse_args
from split_dataset import split_dataset
from train import train_model
from test import test_model
from visualize import visualize_gradcam
from logger import create_experiment_logger


def main():
    """Main function"""
    config = parse_args()

    logger = create_experiment_logger(config)
    logger.log_experiment_start()

    split_info = None
    train_results = None
    test_results = None

    try:
        if config.mode == 'split' or config.mode == 'all':
            print("\n[1/4] Dataset splitting")
            split_info = split_dataset(config)

        if config.mode == 'train' or config.mode == 'all':
            print("\n[2/4] Model training")

            split_info_path = os.path.join(config.dataset_dir, 'split_info.json')
            if not os.path.exists(split_info_path):
                print(f"Error: Dataset not split. Run: python main.py --mode split --random_seed {config.random_seed}")
                sys.exit(1)

            train_results = train_model(config)

        if config.mode == 'test' or config.mode == 'all':
            print("\n[3/4] Model testing")

            model_path = os.path.join(config.models_dir, 'fold_1_best.pt')
            if not os.path.exists(model_path):
                print(f"Error: No trained model found. Run: python main.py --mode train --random_seed {config.random_seed}")
                sys.exit(1)

            test_results = test_model(config)

        if config.mode == 'visualize' or config.mode == 'all':
            print("\n[4/4] True YOLO Attention Visualization")

            model_path = os.path.join(config.models_dir, 'fold_1_best.pt')
            if not os.path.exists(model_path):
                if config.mode == 'all':
                    print("Skipping visualization (no trained model)")
                else:
                    print("Error: No trained model found")
                    sys.exit(1)
            else:
                print("Using true YOLO attention mechanism (Grad-CAM)")
                visualize_gradcam(config)

        logger.log_experiment_results(split_info, train_results, test_results)

        print("\n" + "=" * 60)
        print("All tasks completed")
        print(f"Results saved to: {config.output_base_dir}")
        print("=" * 60)

        logger.print_summary_table()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
