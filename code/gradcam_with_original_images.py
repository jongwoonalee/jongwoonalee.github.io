#!/usr/bin/env python3
"""
GradCAM Visualization with Original Patient Images
Creates heatmap overlays on actual tissue patch images for all experiments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from dataset import LRRC15Dataset
from models import LRRC15GradePredictor
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

class GradCAMOriginalImageVisualizer:
    """GradCAM visualization with original patient tissue images"""
    
    def __init__(self, save_dir="gradcam_original_images"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for each experiment
        for exp in ['2class_no_attention_mps', '2class_no_thresholds_mps', 
                   '2class_quantile_split_mps', '2class_high_res_mps', 
                   '3class_quantile_mps', '4class_quantile_mps']:
            (self.save_dir / exp).mkdir(exist_ok=True)
        
        print(f"🎨 GradCAM Original Image Visualizer initialized")
        print(f"📁 Results will be saved to: {self.save_dir.absolute()}")

    def load_dataset(self):
        """Load the actual LRRC15 dataset with original images"""
        try:
            excel_path = "LRRC15_TIL/annotation/lrrc15_annotation.xlsx"
            image_base_dir = "."
            
            # Transform for loading original images (minimal processing)
            transform_original = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            dataset = LRRC15Dataset(
                excel_path=excel_path,
                image_base_dir=image_base_dir,
                transform=transform_original
            )
            
            # Use same test split as experiments
            full_indices = list(range(len(dataset.valid_indices)))
            train_val_indices, test_indices = train_test_split(
                full_indices, test_size=0.2, random_state=42,
                stratify=[dataset.df.iloc[dataset.valid_indices[i]]['grade'] for i in full_indices]
            )
            
            print(f"✅ Dataset loaded: {len(dataset.valid_indices)} valid patients")
            print(f"📊 Test patients: {len(test_indices)}")
            
            return dataset, test_indices
            
        except Exception as e:
            print(f"⚠️ Failed to load dataset: {e}")
            return None, None

    def load_model_with_config(self, experiment_name, model_path, device):
        """Load model with proper configuration"""
        try:
            # Determine model config from experiment name
            if 'no_attention' in experiment_name:
                use_attention = False
                learnable_thresholds = False
            elif 'no_thresholds' in experiment_name:
                use_attention = True
                learnable_thresholds = False
            else:
                use_attention = True  
                learnable_thresholds = True
            
            # Determine number of classes
            if '3class' in experiment_name:
                num_grades = 3
            elif '4class' in experiment_name:
                num_grades = 4
            else:
                num_grades = 2
            
            model = LRRC15GradePredictor(
                num_grades=num_grades,
                use_attention=use_attention,
                learnable_thresholds=learnable_thresholds
            ).to(device)
            
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            model.eval()
            
            return model
            
        except Exception as e:
            print(f"⚠️ Failed to load {experiment_name}: {e}")
            return None

    def generate_gradcam(self, model, input_tensor, target_class, device):
        """Generate GradCAM heatmap for a single patch"""
        model.eval()
        
        # Register hooks for feature extraction
        gradients = []
        activations = []
        
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])
        
        def forward_hook(module, input, output):
            activations.append(output)
        
        # Find the last convolutional layer
        target_layer = None
        for name, module in model.instance_extractor.named_modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module
        
        if target_layer is None:
            print("⚠️ No convolutional layer found for GradCAM")
            return None
        
        # Register hooks
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_backward_hook(backward_hook)
        
        try:
            # Forward pass
            input_tensor = input_tensor.unsqueeze(0).to(device)
            input_tensor.requires_grad_(True)
            
            # Get model prediction for this single patch
            bag_sizes = torch.tensor([1]).to(device)
            outputs = model(input_tensor, bag_sizes)
            
            if 'grade_logits' in outputs:
                logits = outputs['grade_logits']
                target_logit = logits[0, target_class] if target_class < logits.size(1) else logits[0, 0]
            else:
                # For models without grade_logits, use h_score or other outputs
                if 'h_score_norm' in outputs:
                    target_logit = outputs['h_score_norm'][0]
                else:
                    print("⚠️ No suitable output for GradCAM")
                    return None
            
            # Backward pass
            model.zero_grad()
            target_logit.backward()
            
            if gradients and activations:
                # Get gradients and activations
                gradient = gradients[0][0].cpu().data.numpy()  # [C, H, W]
                activation = activations[0][0].cpu().data.numpy()  # [C, H, W]
                
                # Calculate weights (global average pooling of gradients)
                weights = np.mean(gradient, axis=(1, 2))  # [C]
                
                # Generate CAM
                cam = np.zeros(activation.shape[1:], dtype=np.float32)  # [H, W]
                for i, w in enumerate(weights):
                    cam += w * activation[i, :, :]
                
                # Apply ReLU and normalize
                cam = np.maximum(cam, 0)
                if cam.max() > 0:
                    cam = cam / cam.max()
                
                return cam
            
        except Exception as e:
            print(f"⚠️ GradCAM generation failed: {e}")
            return None
        
        finally:
            forward_handle.remove()
            backward_handle.remove()
        
        return None

    def create_gradcam_overlay(self, original_image, gradcam_heatmap, alpha=0.6):
        """Create GradCAM overlay on original image"""
        if gradcam_heatmap is None:
            return original_image
        
        # Convert tensor to numpy if needed
        if torch.is_tensor(original_image):
            # Denormalize the image
            denorm_image = original_image.clone()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            denorm_image = denorm_image * std + mean
            denorm_image = torch.clamp(denorm_image, 0, 1)
            
            # Convert to numpy
            img_np = denorm_image.permute(1, 2, 0).cpu().numpy()
        else:
            img_np = original_image
        
        # Resize GradCAM to match image size
        h, w = img_np.shape[:2]
        gradcam_resized = cv2.resize(gradcam_heatmap, (w, h))
        
        # Apply colormap to GradCAM
        heatmap = cv2.applyColorMap(np.uint8(255 * gradcam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = heatmap.astype(np.float32) / 255.0
        
        # Convert original image to proper format
        if img_np.max() <= 1.0:
            img_np = img_np
        else:
            img_np = img_np / 255.0
        
        # Create overlay
        overlayed = heatmap * alpha + img_np * (1 - alpha)
        
        return overlayed

    def visualize_patient_gradcam(self, model, patient_data, patient_id, true_grade, 
                                predicted_grade, confidence, experiment_name, device):
        """Create GradCAM visualization for a patient"""
        
        patches = patient_data['patches']  # [N, 3, H, W]
        n_patches = patches.size(0)
        
        if n_patches == 0:
            return None
        
        # Create visualization grid
        cols = min(6, n_patches)
        rows = max(2, (n_patches + cols - 1) // cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if rows == 1:
            axes = axes.reshape(1, -1) if cols > 1 else [[axes]]
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        patches_to_show = min(n_patches, cols * rows)
        
        # Get attention weights if available
        attention_weights = None
        if hasattr(model, 'use_attention') and model.use_attention:
            try:
                with torch.no_grad():
                    bag_sizes = torch.tensor([n_patches]).to(device)
                    outputs = model(patches.to(device), bag_sizes)
                    if hasattr(model.mil_pooling, 'last_attention_weights'):
                        attention_weights = model.mil_pooling.last_attention_weights.cpu().numpy().flatten()
            except:
                attention_weights = None
        
        # Sort patches by attention if available, otherwise by index
        if attention_weights is not None:
            patch_indices = np.argsort(attention_weights)[::-1]  # Highest attention first
        else:
            patch_indices = range(patches_to_show)
        
        processed_patches = 0
        
        for i in range(rows):
            for j in range(cols):
                if processed_patches >= patches_to_show:
                    axes[i, j].axis('off')
                    continue
                
                patch_idx = patch_indices[processed_patches]
                patch = patches[patch_idx]
                ax = axes[i, j]
                
                # Generate GradCAM for this patch
                gradcam = self.generate_gradcam(model, patch, predicted_grade, device)
                
                # Create overlay
                overlayed_image = self.create_gradcam_overlay(patch, gradcam, alpha=0.5)
                
                # Display
                ax.imshow(overlayed_image)
                
                # Title with information
                title_parts = [f'Patch {patch_idx}']
                
                if attention_weights is not None:
                    att_val = attention_weights[patch_idx]
                    title_parts.append(f'Att: {att_val:.3f}')
                
                if gradcam is not None:
                    title_parts.append('+ GradCAM')
                
                ax.set_title(' '.join(title_parts), fontsize=10, fontweight='bold')
                ax.axis('off')
                
                # Color border based on attention or GradCAM intensity
                border_color = 'gray'
                if attention_weights is not None:
                    att_val = attention_weights[patch_idx]
                    if att_val > 0.15:
                        border_color = 'red'
                    elif att_val > 0.05:
                        border_color = 'orange'
                    else:
                        border_color = 'blue'
                elif gradcam is not None:
                    if gradcam.max() > 0.7:
                        border_color = 'red'
                    elif gradcam.max() > 0.3:
                        border_color = 'orange'
                
                for spine in ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(3)
                
                processed_patches += 1
        
        # Main title
        accuracy_info = f"Test Acc: {self.get_experiment_accuracy(experiment_name):.1f}%"
        plt.suptitle(f'{experiment_name} - Patient {patient_id}\\n'
                    f'True: {true_grade}, Pred: {predicted_grade} ({confidence:.1f}%) | {accuracy_info}', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save visualization
        save_path = self.save_dir / experiment_name / f"patient_{patient_id}_gradcam.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path

    def get_experiment_accuracy(self, experiment_name):
        """Get test accuracy for experiment from previous results"""
        accuracies = {
            '2class_no_attention_mps': 88.0,
            '2class_no_thresholds_mps': 76.0, 
            '2class_quantile_split_mps': 64.0,
            '2class_high_res_mps': 72.0,
            '3class_quantile_mps': 72.0,
            '4class_quantile_mps': 64.0
        }
        return accuracies.get(experiment_name, 0.0)

    def load_test_results(self, experiment_name):
        """Load existing test results for patient information"""
        results_file = f"conference_results/{experiment_name}_test_results.json"
        
        if Path(results_file).exists():
            with open(results_file, 'r') as f:
                return json.load(f)
        return None

    def run_gradcam_analysis_all_experiments(self, device='mps', max_patients_per_exp=3):
        """Run GradCAM analysis for all experiments with original images"""
        
        print("🔬 Starting GradCAM Analysis with Original Images")
        print("=" * 60)
        
        # Load dataset
        dataset, test_indices = self.load_dataset()
        if dataset is None:
            print("❌ Could not load dataset")
            return
        
        # Experiment configurations
        experiments = {
            '2class_no_attention_mps': 'best_mil_model_mps_2class_no_attention_mps_fold1.pth',
            '2class_no_thresholds_mps': 'best_mil_model_mps_2class_no_thresholds_mps_fold1.pth',
            '2class_quantile_split_mps': 'best_mil_model_mps_2class_quantile_split_mps_fold3.pth',
            '2class_high_res_mps': 'best_mil_model_mps_2class_high_res_mps_fold1.pth',
            '3class_quantile_mps': 'best_mil_model_mps_3class_quantile_mps_fold1.pth',
            '4class_quantile_mps': 'best_mil_model_mps_4class_quantile_mps_fold3.pth'
        }
        
        all_results = {}
        
        for exp_name, model_path in experiments.items():
            if not Path(model_path).exists():
                print(f"⚠️ Model not found: {model_path}")
                continue
            
            print(f"\\n🔍 Processing {exp_name} ({self.get_experiment_accuracy(exp_name):.1f}% accuracy)...")
            
            # Load model
            model = self.load_model_with_config(exp_name, model_path, device)
            if model is None:
                continue
            
            # Load test results for patient info
            test_results = self.load_test_results(exp_name)
            if test_results is None:
                print(f"⚠️ No test results found for {exp_name}")
                continue
            
            exp_results = []
            patients_processed = 0
            
            # Process patients using test indices
            for i, test_idx in enumerate(test_indices[:max_patients_per_exp]):
                if patients_processed >= max_patients_per_exp:
                    break
                
                try:
                    # Get patient data from dataset
                    actual_idx = dataset.valid_indices[test_idx]
                    patient_row = dataset.df.iloc[actual_idx]
                    patient_id = patient_row['환자ID'].replace('T-', '')
                    true_grade = patient_row['grade']
                    
                    # Get patient patches
                    patient_data = dataset[test_idx]
                    
                    if isinstance(patient_data, dict):
                        patches = patient_data['patches']
                        patient_id = patient_data.get('patient_id', patient_id)
                        true_grade = patient_data.get('grade', true_grade)
                    else:
                        # Assume tuple format (patches, grade, patient_id)
                        patches = patient_data[0]
                        true_grade = patient_data[1]
                        if len(patient_data) > 2:
                            patient_id = patient_data[2]
                    
                    # Get prediction info from test results
                    if i < len(test_results['patient_ids']) and test_results['patient_ids'][i] == patient_id:
                        predicted_grade = test_results['predictions'][i]
                        confidence = max(test_results['probabilities'][i]) * 100
                    else:
                        # Fallback: run prediction
                        with torch.no_grad():
                            patches_gpu = patches.unsqueeze(0) if patches.dim() == 3 else patches
                            bag_sizes = torch.tensor([patches_gpu.size(0)]).to(device)
                            outputs = model(patches_gpu.to(device), bag_sizes)
                            
                            if 'grade_logits' in outputs:
                                probs = torch.softmax(outputs['grade_logits'], dim=1)
                                predicted_grade = torch.argmax(probs, dim=1).item()
                                confidence = probs.max().item() * 100
                            else:
                                predicted_grade = 0
                                confidence = 50.0
                    
                    # Create GradCAM visualization
                    patient_data_dict = {'patches': patches}
                    save_path = self.visualize_patient_gradcam(
                        model, patient_data_dict, patient_id, true_grade,
                        predicted_grade, confidence, exp_name, device
                    )
                    
                    if save_path:
                        exp_results.append({
                            'patient_id': patient_id,
                            'true_grade': int(true_grade),
                            'predicted_grade': int(predicted_grade),
                            'confidence': float(confidence),
                            'visualization_path': str(save_path)
                        })
                        
                        print(f"  ✅ Patient {patient_id}: {true_grade}→{predicted_grade} ({confidence:.1f}%)")
                        patients_processed += 1
                
                except Exception as e:
                    print(f"  ⚠️ Failed patient {test_idx}: {e}")
                    continue
            
            all_results[exp_name] = exp_results
            print(f"  📊 Completed {len(exp_results)} patients")
        
        # Save summary
        summary_file = self.save_dir / "gradcam_analysis_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\\n🎯 GRADCAM ANALYSIS COMPLETE!")
        print(f"📁 Visualizations saved to: {self.save_dir}")
        print(f"📊 Summary: {summary_file}")
        
        # Print summary
        total_patients = sum(len(results) for results in all_results.values())
        print(f"\\n📋 Generated GradCAM visualizations:")
        for exp_name, results in all_results.items():
            print(f"  - {exp_name}: {len(results)} patients")
        print(f"📊 Total: {total_patients} patient visualizations")
        
        return all_results

def main():
    visualizer = GradCAMOriginalImageVisualizer()
    
    # Check device
    if torch.backends.mps.is_available():
        device = 'mps'
        print("🍎 Using Apple MPS")
    else:
        device = 'cpu'
        print("💻 Using CPU")
    
    # Run GradCAM analysis
    results = visualizer.run_gradcam_analysis_all_experiments(device=device, max_patients_per_exp=3)
    
    return results

if __name__ == "__main__":
    main()