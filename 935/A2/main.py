import argparse
import sys
from pathlib import Path

from config import Config
from data_preprocessor import prepare_dataset_for_yolo
from trainer import train_model
from evaluator import evaluate_model, ModelEvaluator
from inference import predict
from feature_visualizer import visualize_features
from cross_validator import run_cross_validation


def setup_argparse():
    parser = argparse.ArgumentParser(
        description='Rice Disease Classification System using YOLOv8s-cls',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Prepare dataset:
    python main.py prepare --source ./Dhan-Shomadhan --output ./dataset

  Train model:
    python main.py train --epochs 100 --batch-size 16 --device cuda

  Evaluate model:
    python main.py evaluate --model ./models/best_model.pt --split test
    python main.py evaluate --model ./models/best_model.pt --scenarios

  Inference:
    python main.py infer --model ./models/best_model.pt --source ./test_image.jpg --device cpu
    python main.py infer --model ./models/best_model.pt --source ./test_folder/ --device cuda

  Visualize features:
    python main.py visualize --model ./models/best_model.pt --image ./test_image.jpg

  Cross-validation:
    python main.py cross-validate --n-folds 5 --epochs 100 --device cuda
        """
    )

    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')

    prepare_parser = subparsers.add_parser('prepare', help='Prepare and split dataset')
    prepare_parser.add_argument('--source', type=str, default='./Dhan-Shomadhan',
                               help='Source dataset directory')
    prepare_parser.add_argument('--output', type=str, default=None,
                               help='Output directory (default: <source>/processed_data)')
    prepare_parser.add_argument('--seed', type=int, default=42,
                               help='Random seed for data splitting')

    train_parser = subparsers.add_parser('train', help='Train classification model')
    train_parser.add_argument('--epochs', type=int, default=100,
                             help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=16,
                             help='Batch size for training')
    train_parser.add_argument('--imgsz', type=int, default=224,
                             help='Input image size')
    train_parser.add_argument('--device', type=str, default=None,
                             help='Device to use (cuda/cpu, default: auto-detect)')
    train_parser.add_argument('--resume', action='store_true',
                             help='Resume training from last checkpoint')

    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--model', type=str, required=True,
                            help='Path to trained model (.pt file)')
    eval_parser.add_argument('--split', type=str, default='test',
                            choices=['train', 'val', 'test'],
                            help='Dataset split to evaluate on')
    eval_parser.add_argument('--scenarios', action='store_true',
                            help='Evaluate on different scenarios (white/field/mixed backgrounds)')
    eval_parser.add_argument('--device', type=str, default=None,
                            help='Device to use (cuda/cpu)')

    infer_parser = subparsers.add_parser('infer', help='Run inference on images')
    infer_parser.add_argument('--model', type=str, required=True,
                             help='Path to trained model (.pt file)')
    infer_parser.add_argument('--source', type=str, required=True,
                             help='Path to image file or directory')
    infer_parser.add_argument('--device', type=str, default='cpu',
                             help='Device to use (cuda/cpu)')
    infer_parser.add_argument('--no-save', action='store_true',
                             help='Do not save prediction results')

    viz_parser = subparsers.add_parser('visualize', help='Visualize learned features')
    viz_parser.add_argument('--model', type=str, required=True,
                           help='Path to trained model (.pt file)')
    viz_parser.add_argument('--image', type=str,
                           help='Single image for feature visualization')
    viz_parser.add_argument('--white-bg', type=str,
                           help='White background image for comparison')
    viz_parser.add_argument('--field-bg', type=str,
                           help='Field background image for comparison')
    viz_parser.add_argument('--device', type=str, default='cuda',
                           help='Device to use (cuda/cpu)')

    cv_parser = subparsers.add_parser('cross-validate', help='Run K-fold cross-validation')
    cv_parser.add_argument('--n-folds', type=int, default=5,
                          help='Number of folds (default: 5)')
    cv_parser.add_argument('--epochs', type=int, default=100,
                          help='Number of training epochs per fold')
    cv_parser.add_argument('--batch-size', type=int, default=16,
                          help='Batch size for training')
    cv_parser.add_argument('--device', type=str, default=None,
                          help='Device to use (cuda/cpu)')

    return parser


def run_prepare(args):
    print("\n" + "="*80)
    print("DATASET PREPARATION")
    print("="*80)

    output_dir = args.output if args.output else str(Path(args.source) / 'processed_data')

    splits, stats = prepare_dataset_for_yolo(
        source_dir=args.source,
        output_dir=output_dir,
        seed=args.seed
    )

    print("\n✓ Dataset preparation completed successfully")
    print(f"  Processed dataset saved to: {output_dir}")
    print(f"  Location: Under main dataset folder as required")
    print("\nNext step: Train the model using:")
    print("  python main.py train --epochs 100 --batch-size 16 --device cuda")


def run_train(args):
    print("\n" + "="*80)
    print("MODEL TRAINING")
    print("="*80)

    if not Config.PROCESSED_DIR.exists():
        print(f"\n✗ Error: Processed dataset not found at {Config.PROCESSED_DIR}")
        print("Please run dataset preparation first:")
        print("  python main.py prepare")
        sys.exit(1)

    Config.ensure_directories()

    trainer, results = train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        resume=args.resume
    )

    print("\n✓ Training completed successfully")
    print(f"  Best model saved to: {Config.MODELS_DIR / 'best_model.pt'}")
    print("\nNext steps:")
    print("  1. Evaluate model:")
    print("     python main.py evaluate --model ./models/best_model.pt --scenarios")
    print("  2. Run inference:")
    print("     python main.py infer --model ./models/best_model.pt --source <image_path>")


def run_evaluate(args):
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)

    if not Path(args.model).exists():
        print(f"\n✗ Error: Model not found at {args.model}")
        sys.exit(1)

    if not Config.PROCESSED_DIR.exists():
        print(f"\n✗ Error: Processed dataset not found at {Config.PROCESSED_DIR}")
        print("Please run dataset preparation first:")
        print("  python main.py prepare")
        sys.exit(1)

    if args.device:
        Config.DEVICE = args.device

    evaluator = ModelEvaluator(args.model, device=Config.DEVICE)

    if args.scenarios:
        results = evaluator.evaluate_scenarios()
        print("\n✓ Scenario evaluation completed successfully")
    else:
        results = evaluator.evaluate_on_split(args.split)
        print(f"\n✓ Evaluation on {args.split} split completed successfully")

    print("\nResults saved to:", Config.RESULTS_DIR)


def run_inference(args):
    print("\n" + "="*80)
    print("INFERENCE")
    print("="*80)

    if not Path(args.model).exists():
        print(f"\n✗ Error: Model not found at {args.model}")
        sys.exit(1)

    if not Path(args.source).exists():
        print(f"\n✗ Error: Source not found at {args.source}")
        sys.exit(1)

    save_results = not args.no_save

    predictions = predict(
        model_path=args.model,
        source=args.source,
        device=args.device,
        save_results=save_results
    )

    print("\n✓ Inference completed successfully")
    if save_results:
        print(f"  Results saved to: {Config.RESULTS_DIR / 'predictions'}")


def run_visualize(args):
    print("\n" + "="*80)
    print("FEATURE VISUALIZATION")
    print("="*80)

    if not Path(args.model).exists():
        print(f"\n✗ Error: Model not found at {args.model}")
        sys.exit(1)

    if args.image and not Path(args.image).exists():
        print(f"\n✗ Error: Image not found at {args.image}")
        sys.exit(1)

    if args.white_bg and not Path(args.white_bg).exists():
        print(f"\n✗ Error: White background image not found at {args.white_bg}")
        sys.exit(1)

    if args.field_bg and not Path(args.field_bg).exists():
        print(f"\n✗ Error: Field background image not found at {args.field_bg}")
        sys.exit(1)

    visualize_features(
        model_path=args.model,
        image_path=args.image,
        white_bg_image=args.white_bg,
        field_bg_image=args.field_bg,
        device=args.device
    )

    print("\n✓ Feature visualization completed successfully")
    print(f"  Results saved to: {Config.RESULTS_DIR / 'feature_visualization'}")


def run_cross_validate(args):
    print("\n" + "="*80)
    print(f"{args.n_folds}-FOLD CROSS-VALIDATION")
    print("="*80)

    if not Config.DATASET_DIR.exists():
        print(f"\n✗ Error: Dataset not found at {Config.DATASET_DIR}")
        print("Please ensure the dataset is in ./Dhan-Shomadhan/ directory")
        sys.exit(1)

    Config.ensure_directories()

    fold_results, statistics = run_cross_validation(
        n_folds=args.n_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device
    )

    print("\n✓ Cross-validation completed successfully")
    print(f"\nFinal Results (Mean ± Std):")
    print(f"  Test set:         {statistics['test_set']['mean']:.4f} ± {statistics['test_set']['std']:.4f}")
    print(f"  White background: {statistics['white_background']['mean']:.4f} ± {statistics['white_background']['std']:.4f}")
    print(f"  Field background: {statistics['field_background']['mean']:.4f} ± {statistics['field_background']['std']:.4f}")
    print(f"  Mixed:            {statistics['mixed']['mean']:.4f} ± {statistics['mixed']['std']:.4f}")


def main():
    parser = setup_argparse()
    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        sys.exit(0)

    try:
        if args.mode == 'prepare':
            run_prepare(args)
        elif args.mode == 'train':
            run_train(args)
        elif args.mode == 'evaluate':
            run_evaluate(args)
        elif args.mode == 'infer':
            run_inference(args)
        elif args.mode == 'visualize':
            run_visualize(args)
        elif args.mode == 'cross-validate':
            run_cross_validate(args)
        else:
            parser.print_help()

    except KeyboardInterrupt:
        print("\n\nOperation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
