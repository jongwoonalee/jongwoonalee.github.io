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
from utils import visualize_results, save_predictions, visualize_experiment_results, estimate_experiment_times

def send_notification(message):
    """Send notification when experiment completes"""
    try:
        # macOS notification
        os.system(f'osascript -e \'display notification "{message}" with title "Deep Learning Experiment"\'')
        print(f"🔔 NOTIFICATION: {message}")
    except:
        print(f"📢 ALERT: {message}")

def apple_mps_experiments():
    """Apple MPS optimized experiments - No OOM, frequent checkpoints"""
    
    # Create results directory
    results_dir = Path("apple_mps_results")
    results_dir.mkdir(exist_ok=True)
    
    # Device configuration for Apple Silicon
    if torch.backends.mps.is_available():
        device = 'mps'
        print("🍎 Using Apple MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = 'cuda:0'
        print("🔥 Using CUDA")
    else:
        device = 'cpu'
        print("💻 Using CPU")
    
    print(f"🚀 Starting Apple MPS Experiments")
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Results will be saved to: {results_dir.absolute()}")
    
    # MPS-optimized worker count
    num_workers = 0 if device == 'mps' else min(2, os.cpu_count() or 2)
    print(f"🔧 Using {num_workers} workers for MPS compatibility")
    
    # Conservative transforms for MPS
    transform_train = transforms.Compose([
        transforms.Resize((384, 384)),  # Smaller than 512 for MPS memory
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),  # Less rotation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val_test = transforms.Compose([
        transforms.Resize((384, 384)),  # Consistent smaller size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # MPS-OPTIMIZED EXPERIMENTS - Skip completed 2class_median_split
    experiments = [
        # Experiment 2: 2-class quantile split - MPS OPTIMIZED
        {
            'name': '2class_quantile_split_mps',
            'num_grades': 2,
            'grading_method': 'quantile',
            'use_attention': True,
            'learnable_thresholds': True,
            'batch_size': 1,  # Very conservative for MPS
            'max_patches': 12,  # Much smaller for memory
            'num_epochs': 60,   # Fewer epochs, more frequent saves
            'checkpoint_every': 5  # Save every 5 epochs
        },
        # Experiment 3: 3-class quantile - MPS OPTIMIZED  
        {
            'name': '3class_quantile_mps',
            'num_grades': 3,
            'grading_method': 'quantile',
            'use_attention': True,
            'learnable_thresholds': True,
            'batch_size': 1,  # Single batch for safety
            'max_patches': 10,  # Small patches for 3-class
            'num_epochs': 80,
            'checkpoint_every': 5
        },
        # Experiment 4: 4-class quantile - MPS OPTIMIZED
        {
            'name': '4class_quantile_mps',
            'num_grades': 4,
            'grading_method': 'quantile',
            'use_attention': True,
            'learnable_thresholds': True,
            'batch_size': 1,  # Single batch
            'max_patches': 8,   # Very small for 4-class complexity
            'num_epochs': 100,
            'checkpoint_every': 5
        },
        # Experiment 5: No attention ablation - MPS OPTIMIZED
        {
            'name': '2class_no_attention_mps',
            'num_grades': 2,
            'grading_method': 'two_tier',
            'use_attention': False,  # No attention = less memory
            'learnable_thresholds': True,
            'batch_size': 2,  # Can use slightly more without attention
            'max_patches': 15,  # More patches since no attention
            'num_epochs': 60,
            'checkpoint_every': 5
        },
        # Experiment 6: No thresholds ablation - MPS OPTIMIZED
        {
            'name': '2class_no_thresholds_mps',
            'num_grades': 2,
            'grading_method': 'two_tier',
            'use_attention': True,
            'learnable_thresholds': False,  # Simpler model
            'batch_size': 1,
            'max_patches': 12,
            'num_epochs': 60,
            'checkpoint_every': 5
        },
        # Experiment 7: High resolution - MPS OPTIMIZED
        {
            'name': '2class_high_res_mps',
            'num_grades': 2,
            'grading_method': 'two_tier',
            'use_attention': True,
            'learnable_thresholds': True,
            'batch_size': 1,  # Must be 1 for high-res
            'max_patches': 6,   # Very few patches for 512x512
            'num_epochs': 50,   # Shorter due to complexity
            'image_size': 512,  # Smaller than 768 for MPS
            'checkpoint_every': 5
        },
        # Experiment 8: More patches - MPS OPTIMIZED
        {
            'name': '2class_more_patches_mps',
            'num_grades': 2,
            'grading_method': 'two_tier',
            'use_attention': True,
            'learnable_thresholds': True,
            'batch_size': 1,
            'max_patches': 20,  # More patches but smaller batches
            'num_epochs': 70,
            'checkpoint_every': 5
        }
    ]
    
    all_results = []
    total_experiments = len(experiments)
    
    print(f"\n🧪 Will run {total_experiments} MPS-optimized experiments")
    print("💾 Frequent checkpoints every 5 epochs to prevent data loss")
    print("🔔 Notifications when each experiment completes")
    
    # Show timing estimates
    print("\n" + "="*60)
    estimate_experiment_times()
    print("="*60)
    
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
                    transforms.RandomRotation(10),  # Less aggressive for high-res
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
            
            # Load dataset
            datasets = []
            for folder_path in ['/Volumes/One Touch/ECP_LRRC15_MILpart/LRRC15_TIL',
                               '/Volumes/One Touch/ECP_LRRC15_MILpart/Guro_HE_Patch']:
                try:
                    dataset = LRRC15Dataset(
                        excel_path='/Volumes/One Touch/YOLO_ESP_LRRC15/AsanGuro_merged_ECP_083025.xlsx',
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
            
            # Combine all datasets instead of using only the largest
            if len(datasets) == 1:
                full_dataset = datasets[0]
            else:
                # Combine datasets by merging their DataFrames - only valid patients
                import pandas as pd
                combined_dfs = []
                combined_valid_indices = []
                current_offset = 0
                
                for ds in datasets:
                    # Only include rows for valid patients
                    valid_df = ds.df.iloc[ds.valid_indices].copy()
                    combined_dfs.append(valid_df)
                    # Update valid indices with offset
                    new_indices = [i + current_offset for i in range(len(valid_df))]
                    combined_valid_indices.extend(new_indices)
                    current_offset += len(valid_df)
                
                combined_df = pd.concat(combined_dfs, ignore_index=True)
                print(f"✅ Combined {len(datasets)} datasets")
                
                # Create a new dataset with combined data
                full_dataset = datasets[0]  # Use first dataset as template
                full_dataset.df = combined_df
                full_dataset.valid_indices = combined_valid_indices
                
            print(f"📊 Using combined dataset with {len(full_dataset)} patients")
            print(f"🏷️ Grade distribution: {np.bincount(full_dataset.df['grade'].values)}")
            
            # Train-Val-Test split: 60%-20%-20% for conference paper
            print(f"📊 Using Train-Val-Test split: 60%-20%-20%")
            print(f"🔄 3-fold CV on validation set for model selection")
            
            # First split: 80% train+val, 20% test (held-out)
            train_val_indices, test_indices = train_test_split(
                range(len(full_dataset)), 
                test_size=0.2, 
                random_state=42,
                stratify=[full_dataset.df.iloc[full_dataset.valid_indices[i]]['grade'] 
                          for i in range(len(full_dataset))]
            )
            
            # Second split: 60% train, 20% val from remaining 80%
            train_indices, val_indices = train_test_split(
                train_val_indices,
                test_size=0.25,  # 0.25 * 0.8 = 0.2 (20% of total)
                random_state=42,
                stratify=[full_dataset.df.iloc[full_dataset.valid_indices[i]]['grade'] 
                          for i in train_val_indices]
            )
            
            print(f"📊 Data splits - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
            
            # 3-fold CV only on validation set for model selection
            cv_results = []
            labels_for_val = [full_dataset.df.iloc[full_dataset.valid_indices[i]]['grade'] 
                             for i in val_indices]
            
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            
            # Train model on training set with 3-fold CV on validation set for hyperparameter tuning
            best_val_acc = 0
            best_model_state = None
            
            for fold, (val_fold1_idx, val_fold2_idx) in enumerate(skf.split(val_indices, labels_for_val)):
                fold_start_time = time.time()
                print(f"\n🔄 Validation Fold {fold+1}/3 - {exp_config['name']}")
                print(f"{'='*50}")
                
                # Use training set + 2/3 of validation set for training
                # Use 1/3 of validation set for validation in this fold
                fold_train_indices = train_indices + [val_indices[i] for i in val_fold1_idx]
                fold_val_indices = [val_indices[i] for i in val_fold2_idx]
                
                # Create separate dataset instances to avoid transform conflicts
                import copy
                train_dataset_copy = copy.deepcopy(full_dataset)
                val_dataset_copy = copy.deepcopy(full_dataset)
                
                # Set transforms on separate dataset instances
                train_dataset_copy.transform = transform_train_exp
                val_dataset_copy.transform = transform_val_test_exp
                
                fold_train_dataset = torch.utils.data.Subset(train_dataset_copy, fold_train_indices)
                fold_val_dataset = torch.utils.data.Subset(val_dataset_copy, fold_val_indices)
                
                print(f"  Train: {len(fold_train_dataset)}, Val: {len(fold_val_dataset)}")
                
                # Create dataloaders with MPS-safe settings
                fold_train_loader = DataLoader(
                    fold_train_dataset, 
                    batch_size=exp_config['batch_size'], 
                    shuffle=True, 
                    num_workers=num_workers,
                    collate_fn=mil_collate_fn,
                    pin_memory=False,  # Disable pin_memory for MPS
                    persistent_workers=False
                )
                fold_val_loader = DataLoader(
                    fold_val_dataset, 
                    batch_size=exp_config['batch_size'], 
                    shuffle=False, 
                    num_workers=num_workers,
                    collate_fn=mil_collate_fn,
                    pin_memory=False,
                    persistent_workers=False
                )
                
                # Create model
                fold_model = LRRC15GradePredictor(
                    num_grades=exp_config['num_grades'],
                    use_attention=exp_config['use_attention'],
                    learnable_thresholds=exp_config['learnable_thresholds']
                )
                
                # Train fold with frequent checkpointing
                trained_fold_model, fold_history = train_model_mps_safe(
                    fold_model, 
                    fold_train_loader, 
                    fold_val_loader, 
                    num_epochs=exp_config['num_epochs'], 
                    device=device,
                    mixed_precision=False,  # Disable for MPS compatibility
                    fold_num=f"{exp_config['name']}_fold{fold+1}",
                    checkpoint_every=exp_config.get('checkpoint_every', 5)
                )
                
                # Load best model
                checkpoint_path = f'best_mil_model_mps_{exp_config["name"]}_fold{fold+1}.pth'
                checkpoint = torch.load(checkpoint_path, map_location=device)
                best_val_acc = checkpoint['best_val_acc']
                
                fold_time = time.time() - fold_start_time
                cv_results.append({
                    'fold': fold+1,
                    'val_accuracy': best_val_acc,
                    'model_path': checkpoint_path,
                    'fold_time_minutes': fold_time / 60
                })
                
                print(f"✅ Fold {fold+1}/3 COMPLETE: {best_val_acc:.2f}% accuracy in {fold_time/60:.1f} minutes")
                print(f"{'='*50}")
                
                # MPS memory cleanup
                del fold_train_loader, fold_val_loader, trained_fold_model
                if device == 'mps':
                    torch.mps.empty_cache()
                elif device == 'cuda':
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
                time.sleep(1)  # Brief pause for MPS
            
            # CV Results for model selection
            cv_accs = [r['val_accuracy'] for r in cv_results]
            cv_mean = np.mean(cv_accs)
            cv_std = np.std(cv_accs)
            
            print(f"\n📊 CV Results for {exp_config['name']} (Model Selection):")
            print(f"🎯 Mean Validation Accuracy: {cv_mean:.2f} ± {cv_std:.2f}%")
            print(f"📈 Individual folds: {[f'{acc:.1f}%' for acc in cv_accs]}")
            avg_fold_time = np.mean([r['fold_time_minutes'] for r in cv_results])
            print(f"⏱️ Average fold time: {avg_fold_time:.1f} minutes")
            
            # Select best model based on CV results
            best_fold_idx = np.argmax(cv_accs)
            best_model_path = cv_results[best_fold_idx]['model_path']
            print(f"🏆 Best model from fold {best_fold_idx+1}: {best_model_path}")
            
            # FINAL TEST SET EVALUATION (Conference Paper Standard)
            print(f"\n🧪 FINAL TEST SET EVALUATION")
            print(f"{'='*50}")
            
            # Create test dataset
            test_dataset_copy = copy.deepcopy(full_dataset)
            test_dataset_copy.transform = transform_val_test_exp
            test_dataset = torch.utils.data.Subset(test_dataset_copy, test_indices)
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=mil_collate_fn
            )
            
            # Load best model and evaluate on test set
            test_model = LRRC15GradePredictor(
                num_grades=exp_config['num_grades'],
                use_attention=exp_config.get('use_attention', False),
                learnable_thresholds=exp_config.get('learnable_thresholds', False)
            ).to(device)
            
            checkpoint = torch.load(best_model_path, map_location=device)
            test_model.load_state_dict(checkpoint['model_state_dict'])
            test_model.eval()
            
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for batch in test_loader:
                    inputs = batch['bag'].to(device)
                    targets = batch['grade'].to(device)
                    
                    outputs = test_model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    
                    test_total += targets.size(0)
                    test_correct += (predicted == targets).sum().item()
            
            test_accuracy = 100 * test_correct / test_total
            print(f"🎯 FINAL TEST ACCURACY: {test_accuracy:.2f}%")
            print(f"📊 Test samples: {test_total}")
            print(f"{'='*50}")
            
            # Clean up test model
            del test_model, test_loader
            if device == 'mps':
                torch.mps.empty_cache()
            elif device == 'cuda':
                torch.cuda.empty_cache()
            
            # Save experiment results with test accuracy
            exp_time = time.time() - exp_start_time
            result = {
                'experiment': exp_config['name'],
                'config': exp_config,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'cv_folds': cv_accs,
                'test_accuracy': test_accuracy,  # Final test set performance
                'best_fold': best_fold_idx + 1,
                'data_splits': {
                    'train': len(train_indices),
                    'val': len(val_indices), 
                    'test': len(test_indices)
                },
                'device': device,
                'time_hours': exp_time / 3600,
                'timestamp': datetime.now().isoformat()
            }
            
            all_results.append(result)
            
            print(f"\n🎯 {exp_config['name']} RESULTS:")
            print(f"📊 CV (Model Selection): {cv_mean:.2f} ± {cv_std:.2f}%")
            print(f"🧪 Final Test Accuracy: {test_accuracy:.2f}%")
            print(f"⏱️ Time: {exp_time/3600:.2f} hours")
            
            # Save intermediate results
            result_file = results_dir / f'mps_results_{exp_config["name"]}.json'
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Send completion notification
            send_notification(f"Experiment {exp_config['name']} completed! Test Accuracy: {test_accuracy:.1f}%")
            
            # Final cleanup
            if device == 'mps':
                torch.mps.empty_cache()
            elif device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"❌ EXPERIMENT {exp_config['name']} FAILED: {e}")
            send_notification(f"Experiment {exp_config['name']} FAILED: {str(e)[:50]}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"🍎 APPLE MPS EXPERIMENTS COMPLETED!")
    print(f"{'='*80}")
    
    # Save all results
    summary = {
        'total_experiments': len(all_results),
        'completed_at': datetime.now().isoformat(),
        'total_time_hours': sum(r.get('time_hours', 0) for r in all_results),
        'device': device,
        'results': all_results
    }
    
    summary_file = results_dir / 'apple_mps_experiments_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📄 Summary saved to: {summary_file.absolute()}")
    
    # Print summary table with test accuracies
    print(f"📊 RESULTS SUMMARY (Conference Paper Format):")
    print(f"{'Experiment':<30} {'CV(Val)':<12} {'Test Acc':<10} {'Time(h)':<8}")
    print(f"{'-'*65}")
    
    for result in all_results:
        name = result['experiment'][:29]
        cv_result = f"{result['cv_mean']:.1f}±{result['cv_std']:.1f}%"
        test_result = f"{result['test_accuracy']:.1f}%"
        time_result = f"{result['time_hours']:.1f}h"
        print(f"{name:<30} {cv_result:<12} {test_result:<10} {time_result:<8}")
    
    print(f"\n🏆 BEST EXPERIMENTS:")
    if all_results:
        best_test = max(all_results, key=lambda x: x['test_accuracy'])
        best_cv = max(all_results, key=lambda x: x['cv_mean'])
        print(f"Best Test: {best_test['experiment']} ({best_test['test_accuracy']:.1f}%)")
        print(f"Best CV: {best_cv['experiment']} ({best_cv['cv_mean']:.1f}±{best_cv['cv_std']:.1f}%)")
    
    total_time = sum(r['time_hours'] for r in all_results)
    print(f"\n⏱️ Total Runtime: {total_time:.1f} hours")
    
    # Final notification
    best_result = max(all_results, key=lambda x: x['test_accuracy']) if all_results else None
    send_notification(f"ALL MPS EXPERIMENTS COMPLETE! Best Test: {best_result['test_accuracy']:.1f}%" if best_result else "All experiments finished")
    
    # Generate comprehensive visualizations
    print(f"\n🎨 Generating comprehensive result visualizations...")
    try:
        visualize_experiment_results(results_dir)
        print(f"✅ All visualizations complete!")
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")
    
    return all_results

def train_model_mps_safe(model, train_loader, val_loader, num_epochs=100, device='mps', 
                        mixed_precision=False, fold_num=None, checkpoint_every=5):
    """MPS-safe training with frequent checkpoints"""
    
    from train_rtx3090 import CombinedMILLoss
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
    
    model = model.to(device)
    
    # MPS-optimized components
    criterion = CombinedMILLoss(alpha=0.7, beta=0.2, gamma=0.1)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)  # Conservative LR
    
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    plateau_scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    best_val_acc = 0
    best_val_loss = float('inf')
    history = {'train': [], 'val': []}
    patience_counter = 0
    max_patience = 25  # More patience for smaller batches
    
    print(f"🍎 MPS Training Started: {num_epochs} epochs")
    print(f"💾 Checkpoints every {checkpoint_every} epochs")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_metrics = {'loss': 0, 'correct': 0, 'total': 0}
        
        for batch_idx, batch in enumerate(train_loader):
            bags = batch['bags'].to(device, non_blocking=False)  # No non_blocking for MPS
            bag_sizes = batch['bag_sizes']
            
            targets = {
                'grade': batch['grades'].to(device, non_blocking=False),
                'h_score_norm': batch['h_scores'].to(device, non_blocking=False) / 5.0
            }
            
            optimizer.zero_grad()
            
            # Forward pass (no mixed precision for MPS)
            outputs = model(bags, bag_sizes)
            loss, loss_components = criterion(outputs, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Metrics
            train_metrics['loss'] += loss.item()
            _, predicted = outputs['grade_logits'].max(1)
            train_metrics['total'] += targets['grade'].size(0)
            train_metrics['correct'] += predicted.eq(targets['grade']).sum().item()
            
            # Memory cleanup for MPS
            if batch_idx % 5 == 0:  # Every 5 batches
                if device == 'mps':
                    torch.mps.empty_cache()
                elif device == 'cuda':
                    torch.cuda.empty_cache()
        
        # Validation phase
        model.eval()
        val_metrics = {'loss': 0, 'correct': 0, 'total': 0}
        
        with torch.no_grad():
            for batch in val_loader:
                bags = batch['bags'].to(device, non_blocking=False)
                bag_sizes = batch['bag_sizes']
                
                targets = {
                    'grade': batch['grades'].to(device, non_blocking=False),
                    'h_score_norm': batch['h_scores'].to(device, non_blocking=False) / 5.0
                }
                
                outputs = model(bags, bag_sizes)
                loss, _ = criterion(outputs, targets)
                
                val_metrics['loss'] += loss.item()
                _, predicted = outputs['grade_logits'].max(1)
                val_metrics['total'] += targets['grade'].size(0)
                val_metrics['correct'] += predicted.eq(targets['grade']).sum().item()
        
        # Calculate metrics
        train_loss = train_metrics['loss'] / len(train_loader)
        val_loss = val_metrics['loss'] / len(val_loader)
        train_acc = 100. * train_metrics['correct'] / train_metrics['total']
        val_acc = 100. * val_metrics['correct'] / val_metrics['total']
        
        scheduler.step()
        plateau_scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        elapsed_time = time.time() - start_time
        
        # Enhanced real-time progress display
        progress_bar = '█' * int((epoch + 1) / num_epochs * 20) + '░' * (20 - int((epoch + 1) / num_epochs * 20))
        eta = (elapsed_time / (epoch + 1)) * (num_epochs - epoch - 1)
        lr_display = f"{current_lr:.2e}" if current_lr < 0.001 else f"{current_lr:.4f}"
        
        print(f'[{progress_bar}] Epoch {epoch+1:3d}/{num_epochs} | '
              f'Train: {train_acc:5.2f}% | Val: {val_acc:5.2f}% | '
              f'Loss: {val_loss:.4f} | LR: {lr_display} | '
              f'Time: {epoch_time:4.1f}s | ETA: {eta/60:4.1f}min | '
              f'Best: {best_val_acc:5.2f}%')
        
        # Model checkpointing
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'best_val_loss': best_val_loss,
                'config': {
                    'num_grades': model.num_grades if hasattr(model, 'num_grades') else 2,
                    'use_attention': model.use_attention if hasattr(model, 'use_attention') else True,
                    'learnable_thresholds': model.learnable_thresholds if hasattr(model, 'learnable_thresholds') else True,
                    'device': device,
                    'batch_size': train_loader.batch_size if hasattr(train_loader, 'batch_size') else 1,
                    'fold_info': fold_num,
                    'timestamp': datetime.now().isoformat()
                }
            }, f'best_mil_model_mps_{fold_num or "default"}.pth')
            print(f'✅ New best model saved! Val Acc: {best_val_acc:.2f}%')
        else:
            patience_counter += 1
        
        # Frequent checkpointing
        if (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = f'checkpoint_epoch_{epoch+1}_{fold_num or "default"}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, checkpoint_path)
            print(f'💾 Checkpoint saved: {checkpoint_path}')
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f'\n⏹️ Early stopping after {patience_counter} epochs without improvement')
            break
        
        history['train'].append({
            'epoch': epoch + 1,
            'loss': train_loss,
            'accuracy': train_acc,
            'lr': current_lr
        })
        history['val'].append({
            'epoch': epoch + 1,
            'loss': val_loss,
            'accuracy': val_acc
        })
        
        # Memory cleanup
        if device == 'mps':
            torch.mps.empty_cache()
        elif device == 'cuda':
            torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    print(f"\n🏆 Training Complete! Best Val Acc: {best_val_acc:.2f}% Time: {total_time/60:.1f}min")
    
    return model, history

if __name__ == "__main__":
    results = apple_mps_experiments()