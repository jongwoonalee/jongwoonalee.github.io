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

class LRRC15DatasetPreparer:
    """QuPath GeoJSON을 YOLO 형식으로 변환하고 데이터셋을 구성하는 클래스"""
    
    def __init__(self, json_dir, images_base_dir, output_dir):
        self.json_dir = Path(json_dir)
        self.images_base_dir = Path(images_base_dir)
        self.output_dir = Path(output_dir)
        self.class_mapping = {
            'S0': 0,
            'S1': 1,
            'S2': 2,
            'S3': 3,
            'S4': 4
        }
        self.image_cache = {}  # Cache for image paths to speed up lookups
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
        """전체 데이터셋 준비 프로세스"""
        print("LRRC15 YOLO 데이터셋 준비 시작...")
        
        # 1. 디렉토리 생성
        self._create_directories()
        
        # 2. 모든 PNG 이미지 인덱싱
        print("이미지 파일 인덱싱 중...")
        self._build_image_cache()
        
        # 3. JSON-PNG 매칭 및 처리
        labeled_count, unlabeled_count = self._process_all_data()
        
        # 4. YAML 파일 생성
        self._create_yaml()
        
        print(f"\n데이터셋 준비 완료!")
        print(f"라벨링된 이미지: {labeled_count}개 (train 폴더)")
        print(f"라벨링되지 않은 이미지: {unlabeled_count}개 (test 폴더)")
        
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
        # 대괄호 내부의 좌표 정보 추출
        bracket_match = re.search(r'\[(.*?)\]', filename)
        if bracket_match:
            # 대괄호 내용을 파싱하여 고유 식별자 생성
            bracket_content = bracket_match.group(1)
            # x, y 좌표 추출
            x_match = re.search(r'x=(\d+)', bracket_content)
            y_match = re.search(r'y=(\d+)', bracket_content) 
            
            base_name = re.sub(r'\[.*?\]', '', filename)
            base_normalized = re.sub(r'[^a-zA-Z0-9]', '', base_name).lower()
            
            if x_match and y_match:
                x_coord = x_match.group(1)
                y_coord = y_match.group(1)
                return f"{base_normalized}x{x_coord}y{y_coord}"
            else:
                # 좌표가 없으면 전체 브래킷 내용을 해시로 변환
                import hashlib
                bracket_hash = hashlib.md5(bracket_content.encode()).hexdigest()[:8]
                return f"{base_normalized}{bracket_hash}"
        else:
            # 대괄호가 없는 경우 기존 방식
            normalized = re.sub(r'[^a-zA-Z0-9]', '', filename)
            return normalized.lower()
    
    def _find_matching_image(self, json_path):
        """JSON 파일에 대응하는 PNG 이미지를 찾기"""
        json_normalized = self._normalize_filename(json_path.stem)
        return self.image_cache.get(json_normalized)
        
    def _process_all_data(self):
        """모든 JSON과 이미지 데이터 처리"""
        json_files = list(self.json_dir.rglob('*.geojson'))
        print(f"\n{len(json_files)}개의 GeoJSON 파일 발견")
        
        labeled_images = set()
        labeled_count = 0
        
        # JSON 파일 처리 (라벨링된 데이터)
        print("\n라벨링된 데이터 처리 중...")
        for json_path in tqdm(json_files):
            img_path = self._find_matching_image(json_path)
            if img_path is None:
                print(f"Warning: {json_path.name}에 대응하는 PNG를 찾을 수 없습니다.")
                continue
            
            # YOLO 라벨 생성
            yolo_labels = self._convert_geojson_to_yolo(img_path, json_path)
            
            if yolo_labels:  # 유효한 라벨이 있는 경우에만 처리
                # 이미지 복사
                dst_img = self.output_dir / 'images' / 'train' / img_path.name
                shutil.copy2(img_path, dst_img)
                
                # 라벨 저장
                label_path = self.output_dir / 'labels' / 'train' / f"{img_path.stem}.txt"
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_labels))
                
                labeled_images.add(self._normalize_filename(img_path.stem))
                labeled_count += 1
        
        # 라벨링되지 않은 이미지 처리 (테스트용)
        print("\n라벨링되지 않은 이미지를 test 폴더로 복사 중...")
        unlabeled_count = 0
        for normalized_name, img_path in tqdm(self.image_cache.items()):
            if normalized_name not in labeled_images:
                dst_img = self.output_dir / 'images' / 'test' / img_path.name
                shutil.copy2(img_path, dst_img)
                unlabeled_count += 1
        
        return labeled_count, unlabeled_count
                
    def _convert_geojson_to_yolo(self, img_path, json_path):
        """QuPath GeoJSON을 YOLO 형식으로 변환"""
        try:
            # 이미지 경로 검증
            if not img_path.exists():
                print(f"Warning: 이미지 파일이 존재하지 않습니다: {img_path}")
                return []
                
            if not json_path.exists():
                print(f"Warning: JSON 파일이 존재하지 않습니다: {json_path}")
                return []
            
            # 이미지 크기 가져오기
            try:
                img = Image.open(img_path)
                img_w, img_h = img.size
            except Exception as e:
                print(f"Error opening image {img_path}: {str(e)}")
                return []
            
            # GeoJSON 로드
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            yolo_labels = []
            
            for feature in data.get('features', []):
                # 클래스 이름 가져오기
                properties = feature.get('properties', {})
                classification = properties.get('classification', {})
                class_name = classification.get('name', '')
                
                if class_name not in self.class_mapping:
                    continue
                    
                class_id = self.class_mapping[class_name]
                
                # 좌표 가져오기
                geometry = feature.get('geometry', {})
                if geometry.get('type') != 'Polygon':
                    continue
                    
                coords = geometry.get('coordinates', [[]])[0]
                if len(coords) < 3:  # 최소 3개 점이 필요
                    continue
                
                # Bounding box 계산
                x_coords = [p[0] for p in coords[:-1]]  # 마지막 점은 첫 점과 동일
                y_coords = [p[1] for p in coords[:-1]]
                
                if not x_coords or not y_coords:
                    continue
                
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # YOLO 형식으로 변환 (정규화된 중심점, 너비, 높이)
                x_center = (x_min + x_max) / 2 / img_w
                y_center = (y_min + y_max) / 2 / img_h
                width = (x_max - x_min) / img_w
                height = (y_max - y_min) / img_h
                
                # 유효성 검사
                if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1:
                    yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                
            return yolo_labels
            
        except Exception as e:
            print(f"Error processing {json_path}: {str(e)}")
            return []
            
    def _create_yaml(self):
        """YOLO 학습용 YAML 파일 생성"""
        yaml_content = {
            'train': str(self.output_dir / 'images' / 'train'),
            'val': str(self.output_dir / 'images' / 'train'),  # train과 동일하게 설정, 필요시 수정
            'test': str(self.output_dir / 'images' / 'test'),
            'nc': len(self.class_mapping),
            'names': list(self.class_mapping.keys())
        }
        
        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
            
        print(f"\nYAML 파일 생성: {yaml_path}")
        print(f"클래스: {list(self.class_mapping.keys())}")
        print(f"클래스 매핑: {self.class_mapping}")

# %%
class LRRC15InferenceHelper:
    """테스트 이미지에 대한 간단한 추론 헬퍼"""
    
    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        self.test_images_dir = self.dataset_dir / 'images' / 'test'
        self.class_names = ['S0', 'S1', 'S2', 'S3', 'S4']
        
    def _detect_device(self):
        """디바이스 자동 감지"""
        if torch.backends.mps.is_available():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    
    def run_inference_ultralytics(self, model_path, conf_threshold=0.25, save_results=True):
        """
        Ultralytics YOLO를 사용한 추론 실행
        
        Args:
            model_path: 훈련된 YOLO 모델 경로 (.pt 파일)
            conf_threshold: 신뢰도 임계값
            save_results: 결과 저장 여부
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            print("Ultralytics가 설치되지 않았습니다. 'pip install ultralytics' 실행하세요.")
            return None
            
        # 디바이스 자동 감지
        device = self._detect_device()
        if device == 'mps':
            torch.mps.empty_cache()  # MPS 메모리 정리
        
        if not self.test_images_dir.exists():
            print(f"테스트 이미지 디렉토리를 찾을 수 없습니다: {self.test_images_dir}")
            return None
            
        if not Path(model_path).exists():
            print(f"모델 파일을 찾을 수 없습니다: {model_path}")
            return None
        
        test_images = list(self.test_images_dir.glob('*.png'))
        print(f"추론할 이미지 수: {len(test_images)}개")
        
        if len(test_images) == 0:
            print("추론할 이미지가 없습니다.")
            return None
        
        try:
            # 모델 로드
            model = YOLO(model_path)
            print(f"모델 로드 완료, 디바이스: {device}")
            
            # 추론 실행
            results = model.predict(
                source=str(self.test_images_dir),
                conf=conf_threshold,
                save=save_results,
                project=str(self.dataset_dir),
                name='inference_results',
                device=device
            )
        except Exception as e:
            print(f"추론 중 오류 발생: {str(e)}")
            return None
        
        return results
    
    def get_test_image_info(self):
        """테스트 이미지 정보 출력"""
        if not self.test_images_dir.exists():
            print(f"테스트 디렉토리가 존재하지 않습니다: {self.test_images_dir}")
            return
        
        test_images = list(self.test_images_dir.glob('*.png'))
        print(f"테스트 이미지 수: {len(test_images)}개")
        print(f"테스트 이미지 디렉토리: {self.test_images_dir}")
        
        if test_images:
            print(f"첫 번째 이미지 예시: {test_images[0].name}")

# %%
# 사용 예시
if __name__ == "__main__":
    # 디렉토리 경로 설정
    json_dir = "/Users/jongwonlee/Downloads/LRRC15/StromalGrading_Boundingbox240730"
    images_base_dir = "/Users/jongwonlee/Downloads/LRRC15/"
    output_dir = "./lrrc15_yolo_dataset"
    
    # 데이터셋 준비
    preparer = LRRC15DatasetPreparer(json_dir, images_base_dir, output_dir)
    preparer.prepare_dataset()
    
    # 추론 헬퍼 생성
    inference_helper = LRRC15InferenceHelper(output_dir)
    inference_helper.get_test_image_info()
    
    # 디바이스별 최적화된 훈련 명령어
    device = preparer.device
    batch_size = 16 if device == 'mps' else 32 if device == 'cuda' else 8
    
    print(f"\n디바이스: {device} 감지됨")
    print(f"다음 명령어로 훈련을 시작하세요:")
    print(f"yolo train data={output_dir}/data.yaml model=yolov8n.pt epochs=50 imgsz=512 device={device} batch={batch_size}")
    
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
        print("- epochs를 20 이하로 줄이는 것을 권장합니다")
# %%
