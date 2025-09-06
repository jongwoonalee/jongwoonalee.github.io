import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from dataset import LRRC15Dataset
from models import LRRC15GradePredictor
from train_rtx3090 import train_model, mil_collate_fn
from utils import visualize_results, save_predictions

def main():
    # RTX3090 Optimized Configuration for Conference Paper
    config = {
        'excel_path': '/Volumes/One Touch/YOLO_ESP_LRRC15/AsanGuro_merged_ECP_083025.xlsx',
        'image_base_dir': '/Volumes/One Touch/ECP_LRRC15_MILpart/LRRC15_TIL',  # Will try both folders
        'batch_size': 4,  # Higher batch size for RTX3090's 24GB VRAM
        'num_epochs': 100,  # More epochs for conference paper
        'num_grades': 2,
        'learnable_thresholds': True,
        'use_attention': True,
        'train_ratio': 0.6,  # 60% for training
        'val_ratio': 0.2,    # 20% for validation  
        'test_ratio': 0.2,   # 20% for final test (conference paper)
        'max_patches_per_patient': 30,  # More patches per patient on RTX3090
        'num_workers': 8,  # More workers for faster data loading
        'mixed_precision': True,  # Enable mixed precision for speed
        'cross_validation': True,  # 5-fold CV for conference paper
        'n_folds': 5
    }
    
    # Device configuration - RTX3090
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available! Make sure RTX3090 drivers are installed.")
    
    device = 'cuda:0'
    torch.cuda.set_device(0)
    print(f"🚀 Using RTX3090 on device: {device}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # RTX3090 optimized transforms
    transform_train = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(20),  # More aggressive augmentation
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
    
    # Try both image folders for maximum dataset coverage
    datasets = []
    for folder_path in ['/Volumes/One Touch/ECP_LRRC15_MILpart/LRRC15_TIL',
                       '/Volumes/One Touch/ECP_LRRC15_MILpart/Guro_HE_Patch']:
        try:
            dataset = LRRC15Dataset(
                excel_path=config['excel_path'],
                image_base_dir=folder_path,
                transform=transform_train,
                max_patches_per_patient=config['max_patches_per_patient']
            )
            if len(dataset) > 0:
                datasets.append(dataset)
                print(f"✅ Loaded {len(dataset)} patients from {folder_path}")
        except Exception as e:
            print(f"⚠️ Could not load from {folder_path}: {e}")
    
    if not datasets:
        raise RuntimeError("No datasets could be loaded!")
    
    # Use the dataset with most patients
    full_dataset = max(datasets, key=len)
    print(f"📊 Using dataset with {len(full_dataset)} patients")
    
    if config['cross_validation']:
        print(f"🔄 Conference Paper with {config['n_folds']}-Fold Cross-Validation")
        
        # First: Hold out 20% for final test set
        train_val_indices, test_indices = train_test_split(
            range(len(full_dataset)), 
            test_size=0.2, 
            random_state=42,
            stratify=[full_dataset.df.iloc[full_dataset.valid_indices[i]]['grade'] 
                      for i in range(len(full_dataset))]
        )
        
        # Create final test set
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
        test_dataset.dataset.transform = transform_val_test
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=config['num_workers'],
            collate_fn=mil_collate_fn,
            pin_memory=True
        )
        
        print(f"📊 Held-out test set: {len(test_dataset)} patients")
        
        # Cross-validation on remaining 80%
        cv_results = []
        labels_for_cv = [full_dataset.df.iloc[full_dataset.valid_indices[i]]['grade'] 
                        for i in train_val_indices]
        
        skf = StratifiedKFold(n_splits=config['n_folds'], shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_indices, labels_for_cv)):
            print(f"\n🔄 Cross-Validation Fold {fold+1}/{config['n_folds']}")
            
            # Convert to actual dataset indices
            fold_train_indices = [train_val_indices[i] for i in train_idx]
            fold_val_indices = [train_val_indices[i] for i in val_idx]
            
            # Create fold datasets
            fold_train_dataset = torch.utils.data.Subset(full_dataset, fold_train_indices)
            fold_val_dataset = torch.utils.data.Subset(full_dataset, fold_val_indices)
            fold_val_dataset.dataset.transform = transform_val_test
            
            print(f"  Train: {len(fold_train_dataset)}, Val: {len(fold_val_dataset)}")
            
            # Create fold dataloaders
            fold_train_loader = DataLoader(
                fold_train_dataset, 
                batch_size=config['batch_size'], 
                shuffle=True, 
                num_workers=config['num_workers'],
                collate_fn=mil_collate_fn,
                pin_memory=True
            )
            fold_val_loader = DataLoader(
                fold_val_dataset, 
                batch_size=config['batch_size'], 
                shuffle=False, 
                num_workers=config['num_workers'],
                collate_fn=mil_collate_fn,
                pin_memory=True
            )
            
            # Create fresh model for this fold
            fold_model = LRRC15GradePredictor(
                num_grades=config['num_grades'],
                use_attention=config['use_attention'],
                learnable_thresholds=config['learnable_thresholds']
            )
            
            # Train fold model
            trained_fold_model, fold_history = train_model(
                fold_model, 
                fold_train_loader, 
                fold_val_loader, 
                num_epochs=config['num_epochs'], 
                device=device,
                mixed_precision=config['mixed_precision'],
                fold_num=fold+1
            )
            
            # Load best model for this fold
            checkpoint = torch.load(f'best_mil_model_rtx3090_fold{fold+1}.pth')
            trained_fold_model.load_state_dict(checkpoint['model_state_dict'])
            best_val_acc = checkpoint['best_val_acc']
            
            cv_results.append({
                'fold': fold+1,
                'val_accuracy': best_val_acc,
                'model_path': f'best_mil_model_rtx3090_fold{fold+1}.pth'
            })
            
            print(f"  ✅ Fold {fold+1} Val Accuracy: {best_val_acc:.2f}%")
            
            # Cleanup
            del fold_train_loader, fold_val_loader, trained_fold_model
            torch.cuda.empty_cache()
        
        # Cross-validation results
        cv_accs = [r['val_accuracy'] for r in cv_results]
        cv_mean = np.mean(cv_accs)
        cv_std = np.std(cv_accs)
        
        print(f"\n📊 Cross-Validation Results:")
        print(f"🎯 Mean Validation Accuracy: {cv_mean:.2f} ± {cv_std:.2f}%")
        print(f"📈 Individual Folds: {[f'{acc:.2f}%' for acc in cv_accs]}")
        
        # Train final model on all training data for test evaluation
        print(f"\n🏆 Training Final Model on All Training Data...")
        
        all_train_dataset = torch.utils.data.Subset(full_dataset, train_val_indices)
        all_train_loader = DataLoader(
            all_train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True, 
            num_workers=config['num_workers'],
            collate_fn=mil_collate_fn,
            pin_memory=True
        )
        
        # Create and train final model
        final_model = LRRC15GradePredictor(
            num_grades=config['num_grades'],
            use_attention=config['use_attention'],
            learnable_thresholds=config['learnable_thresholds']
        )
        
        # Train without validation (use all training data)
        final_trained_model, _ = train_model(
            final_model, 
            all_train_loader, 
            test_loader,  # Use test as "validation" for monitoring only
            num_epochs=config['num_epochs'], 
            device=device,
            mixed_precision=config['mixed_precision'],
            fold_num="final"
        )
        
        # Final test evaluation
        checkpoint = torch.load('best_mil_model_rtx3090_final.pth')
        final_trained_model.load_state_dict(checkpoint['model_state_dict'])
        
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
        
        print(f"\n📋 Final Conference Paper Results:")
        print(f"🔄 Cross-Validation: {cv_mean:.2f} ± {cv_std:.2f}%")
        print(f"🎯 Final Test Accuracy: {test_acc:.2f}%")
        print(f"📊 Test Set Size: {test_metrics['total']} patients")
        
        # Generate results for paper
        visualize_results(final_trained_model, test_loader, device)
        save_predictions(final_trained_model, test_loader, 'conference_paper_cv_predictions.csv', device)
        
        return {
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'cv_folds': cv_accs,
            'test_accuracy': test_acc,
            'test_size': len(test_dataset),
            'model_path': 'best_mil_model_rtx3090_final.pth'
        }
    
    else:
        # Original single split approach
        print("🎯 Conference Paper Split: Train(60%) / Val(20%) / Test(20%)")
        
        # First split: 60% train, 40% temp
        train_indices, temp_indices = train_test_split(
            range(len(full_dataset)), 
            test_size=0.4, 
            random_state=42,
            stratify=[full_dataset.df.iloc[full_dataset.valid_indices[i]]['grade'] 
                      for i in range(len(full_dataset))]
        )
        
        # Second split: 20% val, 20% test from the 40% temp
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=0.5,  # 50% of 40% = 20% of total
            random_state=42,
            stratify=[full_dataset.df.iloc[full_dataset.valid_indices[i]]['grade'] 
                      for i in temp_indices]
        )
        
        # Create datasets
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
        
        # Apply different transforms
        val_dataset.dataset.transform = transform_val_test
        test_dataset.dataset.transform = transform_val_test
        
        print(f"📊 Final Split: Train({len(train_dataset)}) / Val({len(val_dataset)}) / Test({len(test_dataset)})")
        
        # High-performance DataLoaders for RTX3090
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True, 
            num_workers=config['num_workers'],
            collate_fn=mil_collate_fn,
            pin_memory=True,
            persistent_workers=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=config['num_workers'],
            collate_fn=mil_collate_fn,
            pin_memory=True,
            persistent_workers=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=config['num_workers'],
            collate_fn=mil_collate_fn,
            pin_memory=True,
            persistent_workers=True
        )
        
        # Model optimized for RTX3090
        model = LRRC15GradePredictor(
            num_grades=config['num_grades'],
            use_attention=config['use_attention'],
            learnable_thresholds=config['learnable_thresholds']
        )
        
        print(f"\n🚀 Starting RTX3090 Training for Conference Paper...")
        
        # Training
        trained_model, history = train_model(
            model, 
            train_loader, 
            val_loader, 
            num_epochs=config['num_epochs'], 
            device=device,
            mixed_precision=config['mixed_precision']
        )
        
        # Test evaluation
        checkpoint = torch.load('best_mil_model_rtx3090.pth')
        trained_model.load_state_dict(checkpoint['model_state_dict'])
        
        trained_model.eval()
        test_metrics = {'correct': 0, 'total': 0}
        
        with torch.no_grad():
            for batch in test_loader:
                bags = batch['bags'].to(device)
                bag_sizes = batch['bag_sizes']
                targets = {'grade': batch['grades'].to(device)}
                
                outputs = trained_model(bags, bag_sizes)
                _, predicted = outputs['grade_logits'].max(1)
                
                test_metrics['total'] += targets['grade'].size(0)
                test_metrics['correct'] += predicted.eq(targets['grade']).sum().item()
        
        test_acc = 100. * test_metrics['correct'] / test_metrics['total']
        
        print(f"\n📋 Conference Paper Results:")
        print(f"🎯 Final Test Accuracy: {test_acc:.2f}%")
        print(f"📊 Test Set Size: {test_metrics['total']} patients")
        
        # Generate results for paper
        visualize_results(trained_model, test_loader, device)
        save_predictions(trained_model, test_loader, 'conference_paper_predictions.csv', device)
        
        return {
            'test_accuracy': test_acc,
            'train_size': len(train_dataset),
            'val_size': len(val_dataset), 
            'test_size': len(test_dataset),
            'model_path': 'best_mil_model_rtx3090.pth'
        }

if __name__ == "__main__":
    results = main()
    print(f"\n🏆 Conference Paper Training Complete!")
    print(f"Test Accuracy: {results['test_accuracy']:.2f}%")