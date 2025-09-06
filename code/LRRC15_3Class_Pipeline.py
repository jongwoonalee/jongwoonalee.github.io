# %%
import json
import shutil
import re
from pathlib import Path
import numpy as np
from PIL import Image
import yaml
from tqdm import tqdm
from collections import defaultdict
import torch

class LRRC15_3Class_DatasetPreparer:
    """QuPath GeoJSON을 3-class YOLO 형식으로 변환 (S2+S3 합침, S4 제외)"""
    
    def __init__(self, json_dir, images_base_dir, output_dir):
        self.json_dir = Path(json_dir)
        self.images_base_dir = Path(images_base_dir)
        self.output_dir = Path(output_dir)
        
        # 수정된 3-class 매핑 (S2+S3 합침, S4 제외)
        self.class_mapping = {
            'S0': 0,   # Background/Normal
            'S1': 1,   # Low expression  
            'S2': 2,   # Moderate-High (합친 클래스)
            'S3': 2,   # Moderate-High (S2와 동일)
            # S4는 라벨링 실수로 제외
        }
        self.class_names = ['S0', 'S1', 'S2-S3_Combined']  # 표시용 이름
        
        self.image_cache = {}
        self.device = self._detect_device()
        
    def _detect_device(self):
        """자동으로 사용 가능한 디바이스 감지"""
        if torch.backends.mps.is_available():
            print("Apple M3 MPS detected - using MPS acceleration")
            return 'mps'
        elif torch.cuda.is_available():
            print(f"CUDA detected - using GPU {torch.cuda.get_device_name()}")
            return 'cuda'
        else:
            print("No GPU acceleration available - using CPU")
            return 'cpu'
        
    def prepare_dataset(self):
        """전체 데이터셋 준비 프로세스 (3-class 버전)"""
        print("LRRC15 3-Class YOLO 데이터셋 준비 시작...")
        print("클래스 매핑: S0=0, S1=1, S2+S3=2 (S4 제외)")
        
        # 1. 디렉토리 생성
        self._create_directories()
        
        # 2. 모든 PNG 이미지 인덱싱
        print("이미지 파일 인덱싱 중...")
        self._build_image_cache()
        
        # 3. JSON-PNG 매칭 및 처리
        labeled_count, unlabeled_count, class_stats = self._process_all_data()
        
        # 4. YAML 파일 생성
        self._create_yaml()
        
        print(f"\n=== 3-Class 데이터셋 준비 완료! ===")
        print(f"라벨링된 이미지: {labeled_count}개 (train 폴더)")
        print(f"라벨링되지 않은 이미지: {unlabeled_count}개 (test 폴더)")
        print(f"\n클래스별 인스턴스 수:")
        for class_name, count in class_stats.items():
            print(f"  {class_name}: {count}개")
        
    def _create_directories(self):
        """필요한 디렉토리 생성"""
        (self.output_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'images' / 'test').mkdir(parents=True, exist_ok=True)
        print(f"출력 디렉토리 생성: {self.output_dir}")
            
    def _build_image_cache(self):
        """모든 PNG 이미지를 인덱싱하여 캐시 구성"""
        print("PNG 이미지 검색 중...")
        for png_path in self.images_base_dir.rglob('*.png'):
            normalized_name = self._normalize_filename(png_path.stem)
            if normalized_name in self.image_cache:
                print(f"Warning: 중복된 정규화 이름 발견: {normalized_name}")
                print(f"  기존: {self.image_cache[normalized_name]}")
                print(f"  새로운: {png_path}")
            self.image_cache[normalized_name] = png_path
        
        print(f"총 {len(self.image_cache)}개의 PNG 이미지를 인덱싱했습니다.")
    
    def _normalize_filename(self, filename):
        """파일명을 정규화하여 매칭에 사용 - 좌표 정보 포함"""
        bracket_match = re.search(r'\[(.*?)\]', filename)
        if bracket_match:
            bracket_content = bracket_match.group(1)
            x_match = re.search(r'x=(\d+)', bracket_content)
            y_match = re.search(r'y=(\d+)', bracket_content) 
            
            base_name = re.sub(r'\[.*?\]', '', filename)
            base_normalized = re.sub(r'[^a-zA-Z0-9]', '', base_name).lower()
            
            if x_match and y_match:
                x_coord = x_match.group(1)
                y_coord = y_match.group(1)
                return f"{base_normalized}x{x_coord}y{y_coord}"
            else:
                import hashlib
                bracket_hash = hashlib.md5(bracket_content.encode()).hexdigest()[:8]
                return f"{base_normalized}{bracket_hash}"
        else:
            normalized = re.sub(r'[^a-zA-Z0-9]', '', filename)
            return normalized.lower()
    
    def _find_matching_image(self, json_path):
        """JSON 파일에 대응하는 PNG 이미지를 찾기"""
        json_normalized = self._normalize_filename(json_path.stem)
        return self.image_cache.get(json_normalized)
        
    def _process_all_data(self):
        """모든 JSON과 이미지 데이터 처리 (3-class 버전)"""
        json_files = list(self.json_dir.rglob('*.geojson'))
        print(f"\n{len(json_files)}개의 GeoJSON 파일 발견")
        
        labeled_images = set()
        labeled_count = 0
        class_stats = {name: 0 for name in self.class_names}
        
        # JSON 파일 처리 (라벨링된 데이터)
        print("\n라벨링된 데이터 처리 중 (3-class)...")
        for json_path in tqdm(json_files):
            img_path = self._find_matching_image(json_path)
            if img_path is None:
                print(f"Warning: {json_path.name}에 대응하는 PNG를 찾을 수 없습니다.")
                continue
            
            # YOLO 라벨 생성 (3-class)
            yolo_labels, labels_class_count = self._convert_geojson_to_yolo_3class(img_path, json_path)
            
            if yolo_labels:
                # 이미지 복사
                dst_img = self.output_dir / 'images' / 'train' / img_path.name
                shutil.copy2(img_path, dst_img)
                
                # 라벨 저장
                label_path = self.output_dir / 'labels' / 'train' / f"{img_path.stem}.txt"
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_labels))
                
                labeled_images.add(self._normalize_filename(img_path.stem))
                labeled_count += 1
                
                # 클래스 통계 업데이트
                for class_idx, count in labels_class_count.items():
                    class_stats[self.class_names[class_idx]] += count
        
        # 라벨링되지 않은 이미지 처리 (테스트용)
        print("\n라벨링되지 않은 이미지를 test 폴더로 복사 중...")
        unlabeled_count = 0
        for normalized_name, img_path in tqdm(self.image_cache.items()):
            if normalized_name not in labeled_images:
                dst_img = self.output_dir / 'images' / 'test' / img_path.name
                shutil.copy2(img_path, dst_img)
                unlabeled_count += 1
        
        return labeled_count, unlabeled_count, class_stats
                
    def _convert_geojson_to_yolo_3class(self, img_path, json_path):
        """QuPath GeoJSON을 3-class YOLO 형식으로 변환 (S2+S3 합침, S4 제외)"""
        try:
            # 경로 검증
            if not img_path.exists():
                print(f"Warning: 이미지 파일이 존재하지 않습니다: {img_path}")
                return [], {}
                
            if not json_path.exists():
                print(f"Warning: JSON 파일이 존재하지 않습니다: {json_path}")
                return [], {}
            
            # 이미지 크기 가져오기
            try:
                img = Image.open(img_path)
                img_w, img_h = img.size
            except Exception as e:
                print(f"Error opening image {img_path}: {str(e)}")
                return [], {}
            
            # GeoJSON 로드
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            yolo_labels = []
            class_count = defaultdict(int)
            
            for feature in data.get('features', []):
                # 클래스 이름 가져오기
                properties = feature.get('properties', {})
                classification = properties.get('classification', {})
                class_name = classification.get('name', '')
                
                # S4는 제외 (라벨링 실수)
                if class_name == 'S4':
                    continue
                    
                if class_name not in self.class_mapping:
                    continue
                    
                class_id = self.class_mapping[class_name]
                
                # 좌표 가져오기
                geometry = feature.get('geometry', {})
                if geometry.get('type') != 'Polygon':
                    continue
                    
                coords = geometry.get('coordinates', [[]])[0]
                if len(coords) < 3:
                    continue
                
                # Bounding box 계산
                x_coords = [p[0] for p in coords[:-1]]
                y_coords = [p[1] for p in coords[:-1]]
                
                if not x_coords or not y_coords:
                    continue
                
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # YOLO 형식으로 변환
                x_center = (x_min + x_max) / 2 / img_w
                y_center = (y_min + y_max) / 2 / img_h
                width = (x_max - x_min) / img_w
                height = (y_max - y_min) / img_h
                
                # 유효성 검사
                if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1:
                    yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                    class_count[class_id] += 1
                
            return yolo_labels, dict(class_count)
            
        except Exception as e:
            print(f"Error processing {json_path}: {str(e)}")
            if self.device == 'mps':
                torch.mps.empty_cache()
            return [], {}
            
    def _create_yaml(self):
        """3-class YOLO 학습용 YAML 파일 생성"""
        yaml_content = {
            'train': str(self.output_dir / 'images' / 'train'),
            'val': str(self.output_dir / 'images' / 'train'),
            'test': str(self.output_dir / 'images' / 'test'),
            'nc': 3,  # 3개 클래스
            'names': self.class_names
        }
        
        yaml_path = self.output_dir / 'data_3class.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
            
        print(f"\n3-Class YAML 파일 생성: {yaml_path}")
        print(f"클래스: {self.class_names}")
        print(f"클래스 매핑: {self.class_mapping}")

# %%
# 사용 예시
if __name__ == "__main__":
    # 디렉토리 경로 설정
    json_dir = "/Users/jongwonlee/Downloads/LRRC15/StromalGrading_Boundingbox240730"
    images_base_dir = "/Users/jongwonlee/Downloads/LRRC15/"
    output_dir = "./lrrc15_3class_yolo_dataset"
    
    # 3-class 데이터셋 준비
    preparer = LRRC15_3Class_DatasetPreparer(json_dir, images_base_dir, output_dir)
    preparer.prepare_dataset()
    
    # 디바이스별 최적화된 훈련 명령어
    device = preparer.device
    batch_size = 16 if device == 'mps' else 32 if device == 'cuda' else 8
    
    print(f"\n=== 3-Class 훈련 명령어 ===")
    print(f"디바이스: {device} 감지됨")
    print(f"다음 명령어로 훈련을 시작하세요:")
    print(f"yolo train data={output_dir}/data_3class.yaml model=yolov8n.pt epochs=100 imgsz=512 device={device} batch={batch_size}")
    
    if device == 'mps':
        print("\n*** Apple M3 최적화 팁 ***")
        print("- 메모리 부족 시 batch size를 8로 줄이세요")
        print("- 이미지 크기를 416으로 줄일 수 있습니다")
    elif device == 'cuda':
        print(f"\n*** CUDA GPU 최적화 팁 ***")
        print("- 메모리가 충분하면 batch size를 64까지 늘릴 수 있습니다")
    else:
        print("\n*** CPU 사용 시 주의사항 ***")
        print("- 훈련 시간이 매우 오래 걸립니다")
        print("- epochs를 50 이하로 줄이는 것을 권장합니다")