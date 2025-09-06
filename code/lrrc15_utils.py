import json
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import torch

class LRRC15Visualizer:
    """검출 결과 시각화 클래스"""
    
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.colors = {
            0: (255, 0, 0),      # S0 - Blue
            1: (0, 255, 0),      # S1 - Green  
            2: (0, 255, 255),    # S2 - Yellow
            3: (255, 0, 255),    # S3 - Magenta
            4: (0, 0, 255)       # S4 - Red
        }
        self.class_names = {
            0: 'S0',
            1: 'S1',
            2: 'S2',
            3: 'S3',
            4: 'S4'
        }
    
    def visualize_predictions(self, image_path, save_path=None):
        """예측 결과 시각화"""
        # 이미지 로드
        img = cv2.imread(str(image_path))  # Convert Path to string for OpenCV compatibility
        
        # 예측
        results = self.model(image_path, imgsz=512, conf=0.25)
        
        # Bounding box 그리기
        for r in results:
            boxes = r.boxes  # Detected bounding boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Box coordinates
                    cls = int(box.cls)  # Class index
                    conf = float(box.conf)  # Confidence score
                    
                    # Box 그리기
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), 
                                self.colors[cls], 2)
                    
                    # 라벨 추가
                    label = f"{self.class_names[cls]} {conf:.2f}"
                    cv2.putText(img, label, (int(x1), int(y1)-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[cls], 2)
        
        if save_path:
            cv2.imwrite(str(save_path), img)
        
        return img
    
    def plot_hscore_distribution(self, excel_path):
        """H-score 분포 시각화"""
        df = pd.read_excel(excel_path)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # H-score 분포
        axes[0, 0].hist(df['h_score'], bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('H-score Distribution')
        axes[0, 0].set_xlabel('H-score')
        axes[0, 0].set_ylabel('Frequency')
        
        # Grade 분포
        grade_counts = df['h_score_grade'].value_counts().sort_index()
        axes[0, 1].bar(grade_counts.index, grade_counts.values)
        axes[0, 1].set_title('LRRC15 Grade Distribution')
        axes[0, 1].set_xlabel('Grade')
        axes[0, 1].set_ylabel('Count')
        
        # Cell count vs H-score
        axes[1, 0].scatter(df['cell_count'], df['h_score'], alpha=0.6)
        axes[1, 0].set_title('Cell Count vs H-score')
        axes[1, 0].set_xlabel('Cell Count')
        axes[1, 0].set_ylabel('H-score')
        
        # Grade별 H-score boxplot
        df.boxplot(column='h_score', by='h_score_grade', ax=axes[1, 1])
        axes[1, 1].set_title('H-score by Grade')
        axes[1, 1].set_xlabel('Grade')
        axes[1, 1].set_ylabel('H-score')
        
        plt.tight_layout()
        plt.savefig('hscore_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()


class DataAugmentationVisualizer:
    """데이터 증강 시각화"""
    
    @staticmethod
    def show_augmentations(image_path):
        """증강 예시 보여주기"""
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        # Horizontal Flip
        axes[0, 1].imshow(cv2.flip(img, 1))
        axes[0, 1].set_title('Horizontal Flip')
        axes[0, 1].axis('off')
        
        # Vertical Flip
        axes[0, 2].imshow(cv2.flip(img, 0))
        axes[0, 2].set_title('Vertical Flip')
        axes[0, 2].axis('off')
        
        # Rotation
        M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), 15, 1)
        rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        axes[1, 0].imshow(rotated)
        axes[1, 0].set_title('Rotation (15°)')
        axes[1, 0].axis('off')
        
        # Brightness adjustment
        bright = cv2.convertScaleAbs(img, alpha=1.2, beta=30)
        axes[1, 1].imshow(bright)
        axes[1, 1].set_title('Brightness Adjustment')
        axes[1, 1].axis('off')
        
        # Color jitter
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:,:,0] = (hsv[:,:,0] + 10) % 180
        hsv[:,:,1] = hsv[:,:,1] * 1.3
        hsv[:,:,2] = hsv[:,:,2] * 0.9
        color_jitter = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        axes[1, 2].imshow(color_jitter)
        axes[1, 2].set_title('Color Jitter')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('augmentation_examples.png', dpi=300, bbox_inches='tight')
        plt.show()


class ModelAnalyzer:
    """모델 성능 분석"""
    
    def __init__(self, results_path):
        self.results_path = Path(results_path)
    
    def plot_training_curves(self):
        """학습 곡선 시각화"""
        # results.csv 읽기
        results_df = pd.read_csv(self.results_path / 'results.csv')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(results_df['epoch'], results_df['train/box_loss'], label='Box Loss')
        axes[0, 0].plot(results_df['epoch'], results_df['train/cls_loss'], label='Class Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # mAP curves
        axes[0, 1].plot(results_df['epoch'], results_df['metrics/mAP50'], label='mAP50')
        axes[0, 1].plot(results_df['epoch'], results_df['metrics/mAP50-95'], label='mAP50-95')
        axes[0, 1].set_title('Validation mAP')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mAP')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision/Recall
        axes[1, 0].plot(results_df['epoch'], results_df['metrics/precision'], label='Precision')
        axes[1, 0].plot(results_df['epoch'], results_df['metrics/recall'], label='Recall')
        axes[1, 0].set_title('Precision & Recall')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate
        axes[1, 1].plot(results_df['epoch'], results_df['lr/pg0'], label='lr0')
        axes[1, 1].plot(results_df['epoch'], results_df['lr/pg1'], label='lr1')
        axes[1, 1].plot(results_df['epoch'], results_df['lr/pg2'], label='lr2')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_predictions(self, excel_path):
        """예측 결과 분석"""
        df = pd.read_excel(excel_path)
        
        # 환자별 평균 H-score
        patient_scores = df.groupby('patient_id')['h_score'].agg(['mean', 'std', 'count'])
        patient_scores = patient_scores.sort_values('mean', ascending=False)
        
        print("=== Top 10 Patients by H-score ===")
        print(patient_scores.head(10))
        
        # Grade 전환 분석
        grade_stats = df.groupby('h_score_grade').agg({
            'h_score': ['mean', 'std', 'min', 'max'],
            'cell_count': ['mean', 'std']
        })
        
        print("\n=== Grade Statistics ===")
        print(grade_stats)
        
        return patient_scores, grade_stats


# 빠른 추론을 위한 배치 처리 클래스
class BatchInference:
    """대량 이미지 빠른 처리"""
    
    def __init__(self, model_path, batch_size=32):
        self.model = YOLO(model_path)
        self.batch_size = batch_size
        
        # GPU 메모리 최적화
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def process_folder(self, folder_path, output_excel):
        """폴더 내 모든 이미지 처리"""
        folder_path = Path(folder_path)
        image_files = list(folder_path.glob('*.png'))
        
        results_list = []
        
        # 배치 단위로 처리
        for i in range(0, len(image_files), self.batch_size):
            batch_files = image_files[i:i+self.batch_size]
            
            # 배치 예측
            batch_results = self.model(
                batch_files, 
                imgsz=512, 
                conf=0.25,
                stream=True,  # 메모리 효율성
                device='mps'  # For M3, change to device=0 for RTX3090
            )
            
            # 결과 처리
            for img_path, result in zip(batch_files, batch_results):
                h_score = self._calculate_hscore(result)
                cell_count = len(result.boxes) if result.boxes is not None else 0
                
                results_list.append({
                    'patient_id': img_path.stem.split('_')[0],
                    'image': img_path.name,
                    'h_score': h_score,
                    'cell_count': cell_count
                })
        
        # 결과 저장
        df = pd.DataFrame(results_list)
        df['h_score_grade'] = pd.qcut(df['h_score'], q=4, labels=[1, 2, 3, 4])
        df.to_excel(output_excel, index=False)
        
        return df
    
    def _calculate_hscore(self, result):
        """단일 결과에서 H-score 계산"""
        h_score = 0
        total_cells = 0
        
        if result.boxes is not None:
            for box in result.boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                weight = cls + 1
                h_score += weight * conf
                total_cells += 1
        
        if total_cells > 0:
            h_score = min(5, (h_score / total_cells) * 2)
        
        return h_score


if __name__ == "__main__":
    # 사용 예시
    
    # 1. 시각화
    visualizer = LRRC15Visualizer('lrrc15_detection/yolo_lrrc15/weights/best.pt')
    visualizer.visualize_predictions('test_image.png', 'result_visualization.png')
    
    # 2. H-score 분석
    visualizer.plot_hscore_distribution('lrrc15_hscore_results.xlsx')
    
    # 3. 학습 곡선 분석
    analyzer = ModelAnalyzer('lrrc15_detection/yolo_lrrc15')
    analyzer.plot_training_curves()
    
    # 4. 배치 추론
    batch_processor = BatchInference('lrrc15_detection/yolo_lrrc15/weights/best.pt')
    results_df = batch_processor.process_folder('data/test_images', 'batch_results.xlsx')