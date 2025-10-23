"""
Testing module for model evaluation
"""
import os
import json
import pandas as pd
import numpy as np
from ultralytics import YOLO
from datetime import datetime


class Tester:
    """YOLO model tester"""

    def __init__(self, config):
        self.config = config
        self.dataset_dir = config.dataset_dir
        self.models_dir = config.models_dir
        self.results_dir = config.results_dir
        self.n_splits = config.n_splits

        self.test_results_dir = os.path.join(self.results_dir, 'test_results')
        os.makedirs(self.test_results_dir, exist_ok=True)
        self.test_results = []

    def _filter_test_data(self, val_data_dir, fold_num):
        """根据背景参数过滤测试数据"""
        import shutil
        
        # 创建过滤后的数据目录
        filtered_dir = os.path.join(self.test_results_dir, f'filtered_data_fold_{fold_num}_{self.config.background}')
        
        # 如果背景类型是 mixed，直接使用原始数据
        if self.config.background == 'mixed':
            return val_data_dir
        
        # 复制原始数据结构
        if os.path.exists(filtered_dir):
            shutil.rmtree(filtered_dir)
        
        # 获取所有类别
        class_names = [d for d in os.listdir(val_data_dir) if os.path.isdir(os.path.join(val_data_dir, d))]
        
        for class_name in class_names:
            class_dir = os.path.join(val_data_dir, class_name)
            filtered_class_dir = os.path.join(filtered_dir, class_name)
            os.makedirs(filtered_class_dir, exist_ok=True)
            
            # 获取所有图像文件
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            filtered_files = []
            for img_file in image_files:
                if self.config.background == 'white':
                    # 只保留包含 'wb' 的文件
                    if 'wb' in img_file.lower():
                        filtered_files.append(img_file)
                elif self.config.background == 'field':
                    # 只保留不包含 'wb' 的文件
                    if 'wb' not in img_file.lower():
                        filtered_files.append(img_file)
            
            # 复制过滤后的文件
            for img_file in filtered_files:
                src_path = os.path.join(class_dir, img_file)
                dst_path = os.path.join(filtered_class_dir, img_file)
                shutil.copy2(src_path, dst_path)
            
            print(f"  {class_name}: {len(filtered_files)}/{len(image_files)} images")
        
        # 检查是否有有效数据
        total_images = sum(len([f for f in os.listdir(os.path.join(filtered_dir, class_name)) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                          for class_name in class_names)
        
        if total_images == 0:
            return None
        
        print(f"  Total filtered images: {total_images}")
        return filtered_dir

    def test_single_fold(self, fold_num):
        """Test a single fold"""
        print(f"\nTesting Fold {fold_num}/{self.n_splits}")

        model_path = os.path.join(self.models_dir, f'fold_{fold_num}_best.pt')
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            return None

        val_data_dir = os.path.join(self.dataset_dir, f'fold_{fold_num}', 'val')
        if not os.path.exists(val_data_dir):
            print(f"Val data not found: {val_data_dir}")
            return None

        # 根据背景参数过滤测试数据
        filtered_data_dir = self._filter_test_data(val_data_dir, fold_num)
        if not filtered_data_dir:
            print(f"No valid test data found for background type: {self.config.background}")
            return None

        model = YOLO(model_path)

        try:
            metrics = model.val(
                data=filtered_data_dir,
                split='val',
                batch=self.config.batch_size,
                device=self.config.device,
                workers=self.config.workers,
                plots=True,
                save_json=True,
                project=self.test_results_dir,
                name=f'fold_{fold_num}_{self.config.background}',
                exist_ok=True
            )

            fold_result = {
                'fold': fold_num,
                'background': self.config.background,
                'model_path': model_path,
                'test_data_dir': filtered_data_dir,
                'top1_accuracy': float(metrics.top1) if hasattr(metrics, 'top1') else None,
                'top5_accuracy': float(metrics.top5) if hasattr(metrics, 'top5') else None,
            }

            self.test_results.append(fold_result)

            if fold_result['top1_accuracy']:
                print(f"Fold {fold_num} ({self.config.background}) - Top1: {fold_result['top1_accuracy']:.4f}")

            return fold_result

        except Exception as e:
            print(f"Error in Fold {fold_num}: {e}")
            return None

    def test_all_folds(self):
        """Test all folds"""
        print(f"\nStarting {self.n_splits}-Fold testing with background: {self.config.background}")

        for fold_num in range(1, self.n_splits + 1):
            result = self.test_single_fold(fold_num)
            if result is None:
                print(f"Skipping Fold {fold_num}")

        self.save_results()
        return self.test_results

    def save_results(self):
        """Save test results"""
        if not self.test_results:
            return

        top1_accuracies = [r['top1_accuracy'] for r in self.test_results if r['top1_accuracy'] is not None]

        summary = {
            'random_seed': self.config.random_seed,
            'n_splits': self.n_splits,
            'background': self.config.background,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'test_results': self.test_results,
        }

        if top1_accuracies:
            summary['top1_mean'] = float(np.mean(top1_accuracies))
            summary['top1_std'] = float(np.std(top1_accuracies))
            summary['top1_min'] = float(np.min(top1_accuracies))
            summary['top1_max'] = float(np.max(top1_accuracies))

        results_json_path = os.path.join(self.test_results_dir, f'test_summary_{self.config.background}.json')
        with open(results_json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)

        df_results = pd.DataFrame(self.test_results)
        results_csv_path = os.path.join(self.test_results_dir, f'test_results_{self.config.background}.csv')
        df_results.to_csv(results_csv_path, index=False, encoding='utf-8-sig')

        if top1_accuracies:
            print(f"\nTesting complete ({self.config.background}) - Mean Top1: {summary['top1_mean']:.4f} ± {summary['top1_std']:.4f}\n")


def test_model(config, fold_num=None):
    """Main testing function"""
    tester = Tester(config)

    if fold_num is not None:
        result = tester.test_single_fold(fold_num)
        tester.test_results = [result] if result else []
        tester.save_results()
        return result
    else:
        return tester.test_all_folds()


if __name__ == '__main__':
    from config import parse_args
    config = parse_args()
    test_model(config)
