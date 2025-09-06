import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import json
import time
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from dataset import LRRC15Dataset
from models import LRRC15GradePredictor
from train_rtx3090 import train_model, mil_collate_fn
from utils import visualize_results, save_predictions

def overnight_experiments():
    """Comprehensive overnight experiment suite for RTX3090"""
    
    # Create results directory
    results_dir = Path("overnight_results")
    results_dir.mkdir(exist_ok=True)
    
    # Device configuration
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available! Make sure RTX3090 drivers are installed.")
    
    device = 'cuda:0'
    torch.cuda.set_device(0)
    print(f"🌙 Starting Overnight RTX3090 Experiments")
    print(f"🚀 GPU: {torch.cuda.get_device_properties(0).name}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Results will be saved to: {results_dir.absolute()}")
    
    # GPU-safe worker count to prevent display timeout
    num_workers = 2  # Reduced from 6 to prevent system overload
    print(f"🔧 Using {num_workers} workers for data loading (GPU-safe)")
    
    # Base transforms
    transform_train = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val_test = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Experiment configurations
    experiments = [
        # Experiment 1: 2-class with different grading methods - SAFE PARAMETERS
        {
            'name': '2class_median_split',
            'num_grades': 2,
            'grading_method': 'two_tier',
            'use_attention': True,
            'learnable_thresholds': True,
            'batch_size': 2,  # Reduced from 4 to prevent display timeout
            'max_patches': 20,  # Reduced from 30
            'num_epochs': 100
        },
        {
            'name': '2class_quantile_split', 
            'num_grades': 2,
            'grading_method': 'quantile',
            'use_attention': True,
            'learnable_thresholds': True,
            'batch_size': 2,  # Reduced from 4
            'max_patches': 20,  # Reduced from 30
            'num_epochs': 100
        },
        # Experiment 2: 3-class quantile - SAFE PARAMETERS
        {
            'name': '3class_quantile',
            'num_grades': 3,
            'grading_method': 'quantile',
            'use_attention': True,
            'learnable_thresholds': True,
            'batch_size': 2,  # Reduced from 4
            'max_patches': 18,  # Reduced from 30
            'num_epochs': 120
        },
        # Experiment 3: 4-class quantile - SAFE PARAMETERS
        {
            'name': '4class_quantile',
            'num_grades': 4,
            'grading_method': 'quantile',
            'use_attention': True,
            'learnable_thresholds': True,
            'batch_size': 1,  # Reduced from 3
            'max_patches': 15,  # Reduced from 25
            'num_epochs': 120
        },
        # Experiment 4: Ablation - No attention - SAFE PARAMETERS
        {
            'name': '2class_no_attention',
            'num_grades': 2,
            'grading_method': 'two_tier',
            'use_attention': False,
            'learnable_thresholds': True,
            'batch_size': 3,  # Reduced from 6
            'max_patches': 25,  # Reduced from 35
            'num_epochs': 100
        },
        # Experiment 5: Ablation - No learnable thresholds - SAFE PARAMETERS
        {
            'name': '2class_no_thresholds',
            'num_grades': 2,
            'grading_method': 'two_tier',
            'use_attention': True,
            'learnable_thresholds': False,
            'batch_size': 2,  # Reduced from 4
            'max_patches': 20,  # Reduced from 30
            'num_epochs': 100
        },
        # Experiment 6: High resolution patches - SAFE PARAMETERS
        {
            'name': '2class_high_res',
            'num_grades': 2,
            'grading_method': 'two_tier', 
            'use_attention': True,
            'learnable_thresholds': True,
            'batch_size': 1,  # Reduced from 2 for high-res
            'max_patches': 12,  # Reduced from 20
            'num_epochs': 80,
            'image_size': 768  # Higher resolution
        },
        # Experiment 7: Maximum patches - SAFE PARAMETERS
        {
            'name': '2class_max_patches',
            'num_grades': 2,
            'grading_method': 'two_tier',
            'use_attention': True,
            'learnable_thresholds': True,
            'batch_size': 1,  # Reduced from 2
            'max_patches': 30,  # Reduced from 50
            'num_epochs': 100
        }
    ]
    
    all_results = []
    total_experiments = len(experiments)
    
    for exp_idx, exp_config in enumerate(experiments):
        print(f"\n{'='*80}")
        print(f"🧪 EXPERIMENT {exp_idx+1}/{total_experiments}: {exp_config['name']}")
        print(f"{'='*80}")
        
        exp_start_time = time.time()
        
        try:
            # Custom transforms for high-res experiment
            if 'image_size' in exp_config:
                img_size = exp_config['image_size']
                transform_train_exp = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.3),
                    transforms.RandomRotation(20),
                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                transform_val_test_exp = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                transform_train_exp = transform_train
                transform_val_test_exp = transform_val_test
            
            # Try both image folders
            datasets = []
            for folder_path in ['D:/ECP_LRRC15_MILpart/LRRC15_TIL',
                               'D:/ECP_LRRC15_MILpart/Guro_HE_Patch']:
                try:
                    dataset = LRRC15Dataset(
                        excel_path='D:/YOLO_ESP_LRRC15/AsanGuro_merged_ECP_083025.xlsx',
                        image_base_dir=folder_path,
                        transform=transform_train_exp,
                        max_patches_per_patient=exp_config['max_patches']
                    )
                    # Override grading method
                    dataset.df = dataset.create_grade_labels(
                        dataset.df, 
                        method=exp_config['grading_method'], 
                        n_grades=exp_config['num_grades']
                    )
                    if len(dataset) > 0:
                        datasets.append(dataset)
                        print(f"✅ Loaded {len(dataset)} patients from {folder_path}")
                except Exception as e:
                    print(f"⚠️ Could not load from {folder_path}: {e}")
            
            if not datasets:
                print(f"❌ No datasets loaded for {exp_config['name']}")
                continue
            
            # Use largest dataset
            full_dataset = max(datasets, key=len)
            print(f"📊 Using dataset with {len(full_dataset)} patients")
            print(f"🏷️ Grade distribution: {np.bincount(full_dataset.df['grade'].values)}")
            
            # 5-fold Cross-validation
            print(f"🔄 Running 5-fold Cross-Validation")
            
            # Hold out 20% for test
            train_val_indices, test_indices = train_test_split(
                range(len(full_dataset)), 
                test_size=0.2, 
                random_state=42,
                stratify=[full_dataset.df.iloc[full_dataset.valid_indices[i]]['grade'] 
                          for i in range(len(full_dataset))]
            )
            
            # CV on remaining 80%
            cv_results = []
            labels_for_cv = [full_dataset.df.iloc[full_dataset.valid_indices[i]]['grade'] 
                            for i in train_val_indices]
            
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_indices, labels_for_cv)):
                print(f"\n🔄 Fold {fold+1}/5")
                
                # Create fold datasets
                fold_train_indices = [train_val_indices[i] for i in train_idx]
                fold_val_indices = [train_val_indices[i] for i in val_idx]
                
                fold_train_dataset = torch.utils.data.Subset(full_dataset, fold_train_indices)
                fold_val_dataset = torch.utils.data.Subset(full_dataset, fold_val_indices)
                fold_val_dataset.dataset.transform = transform_val_test_exp
                
                print(f"  Train: {len(fold_train_dataset)}, Val: {len(fold_val_dataset)}")
                
                # Create dataloaders
                fold_train_loader = DataLoader(
                    fold_train_dataset, 
                    batch_size=exp_config['batch_size'], 
                    shuffle=True, 
                    num_workers=num_workers,
                    collate_fn=mil_collate_fn,
                    pin_memory=True,
                    persistent_workers=True if num_workers > 0 else False
                )
                fold_val_loader = DataLoader(
                    fold_val_dataset, 
                    batch_size=exp_config['batch_size'], 
                    shuffle=False, 
                    num_workers=num_workers,
                    collate_fn=mil_collate_fn,
                    pin_memory=True,
                    persistent_workers=True if num_workers > 0 else False
                )
                
                # Create model
                fold_model = LRRC15GradePredictor(
                    num_grades=exp_config['num_grades'],
                    use_attention=exp_config['use_attention'],
                    learnable_thresholds=exp_config['learnable_thresholds']
                )
                
                # Train fold
                trained_fold_model, fold_history = train_model(
                    fold_model, 
                    fold_train_loader, 
                    fold_val_loader, 
                    num_epochs=exp_config['num_epochs'], 
                    device=device,
                    mixed_precision=True,
                    fold_num=f"{exp_config['name']}_fold{fold+1}"
                )
                
                # Load best model
                checkpoint_path = f'best_mil_model_rtx3090_{exp_config["name"]}_fold{fold+1}.pth'
                checkpoint = torch.load(checkpoint_path)
                best_val_acc = checkpoint['best_val_acc']
                
                cv_results.append({
                    'fold': fold+1,
                    'val_accuracy': best_val_acc,
                    'model_path': checkpoint_path
                })
                
                print(f"  ✅ Fold {fold+1} Val Acc: {best_val_acc:.2f}%")
                
                # GPU memory cleanup - CRITICAL for preventing display timeout
                del fold_train_loader, fold_val_loader, trained_fold_model
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure GPU operations complete
                time.sleep(0.5)  # Brief pause to prevent GPU overload
            
            # CV Results
            cv_accs = [r['val_accuracy'] for r in cv_results]
            cv_mean = np.mean(cv_accs)
            cv_std = np.std(cv_accs)
            
            print(f"\n📊 CV Results for {exp_config['name']}:")
            print(f"🎯 Mean Accuracy: {cv_mean:.2f} ± {cv_std:.2f}%")
            
            # Final test evaluation
            print(f"\n🏆 Final Test Evaluation...")
            
            # Train final model on all training data
            all_train_dataset = torch.utils.data.Subset(full_dataset, train_val_indices)
            test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
            test_dataset.dataset.transform = transform_val_test_exp
            
            all_train_loader = DataLoader(
                all_train_dataset, 
                batch_size=exp_config['batch_size'], 
                shuffle=True, 
                num_workers=num_workers,
                collate_fn=mil_collate_fn,
                pin_memory=True,
                persistent_workers=True if num_workers > 0 else False
            )
            test_loader = DataLoader(
                test_dataset, 
                batch_size=exp_config['batch_size'], 
                shuffle=False, 
                num_workers=num_workers,
                collate_fn=mil_collate_fn,
                pin_memory=True,
                persistent_workers=True if num_workers > 0 else False
            )
            
            # Final model
            final_model = LRRC15GradePredictor(
                num_grades=exp_config['num_grades'],
                use_attention=exp_config['use_attention'],
                learnable_thresholds=exp_config['learnable_thresholds']
            )
            
            # Train final model
            final_trained_model, _ = train_model(
                final_model, 
                all_train_loader, 
                test_loader,
                num_epochs=exp_config['num_epochs'], 
                device=device,
                mixed_precision=True,
                fold_num=f"{exp_config['name']}_final"
            )
            
            # Final test
            final_checkpoint_path = f'best_mil_model_rtx3090_{exp_config["name"]}_final.pth'
            final_checkpoint = torch.load(final_checkpoint_path)
            final_trained_model.load_state_dict(final_checkpoint['model_state_dict'])
            
            final_trained_model.eval()
            test_metrics = {'correct': 0, 'total': 0}
            
            with torch.no_grad():
                for batch in test_loader:
                    bags = batch['bags'].to(device)
                    bag_sizes = batch['bag_sizes']
                    targets = {'grade': batch['grades'].to(device)}
                    
                    outputs = final_trained_model(bags, bag_sizes)
                    _, predicted = outputs['grade_logits'].max(1)
                    
                    test_metrics['total'] += targets['grade'].size(0)
                    test_metrics['correct'] += predicted.eq(targets['grade']).sum().item()
            
            test_acc = 100. * test_metrics['correct'] / test_metrics['total']
            
            # Save experiment results
            exp_time = time.time() - exp_start_time
            result = {
                'experiment': exp_config['name'],
                'config': exp_config,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'cv_folds': cv_accs,
                'test_accuracy': test_acc,
                'test_size': len(test_dataset),
                'dataset_size': len(full_dataset),
                'time_hours': exp_time / 3600,
                'timestamp': datetime.now().isoformat()
            }
            
            all_results.append(result)
            
            print(f"\n🎯 {exp_config['name']} RESULTS:")
            print(f"CV: {cv_mean:.2f} ± {cv_std:.2f}%")
            print(f"Test: {test_acc:.2f}%")
            print(f"Time: {exp_time/3600:.2f} hours")
            
            # Save intermediate results
            result_file = results_dir / f'overnight_results_{exp_config["name"]}.json'
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Final cleanup - prevent GPU memory buildup
            del final_trained_model, all_train_loader, test_loader
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"🧹 GPU memory cleaned: {torch.cuda.memory_allocated(0)/1024**3:.1f}GB used")
            
        except Exception as e:
            print(f"❌ EXPERIMENT {exp_config['name']} FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"🌅 OVERNIGHT EXPERIMENTS COMPLETED!")
    print(f"{'='*80}")
    
    # Save all results
    summary = {
        'total_experiments': len(all_results),
        'started_at': datetime.now().isoformat(),
        'total_time_hours': sum(r.get('time_hours', 0) for r in all_results),
        'system_info': {
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
            'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 'N/A',
            'cpu_count': os.cpu_count(),
            'workers_used': num_workers
        },
        'results': all_results
    }
    
    summary_file = results_dir / 'overnight_experiments_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📄 Summary saved to: {summary_file.absolute()}")
    
    # Print summary table
    print(f"📊 RESULTS SUMMARY:")
    print(f"{'Experiment':<25} {'CV Mean±Std':<15} {'Test Acc':<10} {'Time(h)':<8}")
    print(f"{'-'*65}")
    
    for result in all_results:
        name = result['experiment'][:24]
        cv_result = f"{result['cv_mean']:.1f}±{result['cv_std']:.1f}%"
        test_result = f"{result['test_accuracy']:.1f}%"
        time_result = f"{result['time_hours']:.1f}h"
        print(f"{name:<25} {cv_result:<15} {test_result:<10} {time_result:<8}")
    
    print(f"\n🏆 BEST EXPERIMENTS:")
    if all_results:
        best_cv = max(all_results, key=lambda x: x['cv_mean'])
        best_test = max(all_results, key=lambda x: x['test_accuracy'])
        
        print(f"Best CV: {best_cv['experiment']} ({best_cv['cv_mean']:.1f}±{best_cv['cv_std']:.1f}%)")
        print(f"Best Test: {best_test['experiment']} ({best_test['test_accuracy']:.1f}%)")
    
    total_time = sum(r['time_hours'] for r in all_results)
    print(f"\n⏱️ Total Runtime: {total_time:.1f} hours")
    
    return all_results

if __name__ == "__main__":
    results = overnight_experiments()
