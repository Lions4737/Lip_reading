import cv2
import mediapipe as mp
import numpy as np
import os
import time
from typing import Optional, Tuple, List, Dict

class LipExtractor:
    def __init__(self):
        """
        MediaPipeを使用した唇領域抽出器の初期化
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False
        )
        
        self.lips_indices = list(range(61, 69)) + list(range(291, 299))
        self.lips_indices += list(range(0, 17))

    def get_lip_region_params(self, frame: np.ndarray) -> Optional[Dict]:
        """
        フレームから唇領域のパラメータを抽出
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            return None
            
        face_landmarks = results.multi_face_landmarks[0]
        
        lips_points = []
        for idx in self.lips_indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            lips_points.append([x, y])
            
        lips_points = np.array(lips_points)
        
        center = np.mean(lips_points, axis=0).astype(int)
        max_dim = max(
            np.max(lips_points[:, 0]) - np.min(lips_points[:, 0]),
            np.max(lips_points[:, 1]) - np.min(lips_points[:, 1])
        )
        
        margin_factor = 2.5
        size = int(max_dim * margin_factor)
        half_width = size // 2
        
        return {
            'center': center,
            'size': size,
            'half_width': half_width
        }

    def extract_lip_region(self, frame: np.ndarray, params: Dict) -> Optional[np.ndarray]:
        """
        パラメータを使用して唇領域を抽出
        """
        height, width = frame.shape[:2]
        center = params['center']
        half_width = params['half_width']
        
        x1 = max(0, center[0] - half_width)
        y1 = max(0, center[1] - half_width)
        x2 = min(width, center[0] + half_width)
        y2 = min(height, center[1] + half_width)
        
        lip_region = frame[y1:y2, x1:x2]
        if lip_region.size == 0:
            return None
            
        return cv2.resize(lip_region, (160, 80))

    def process_video(self, video_path: str, output_dir: str) -> None:
        """
        動画から唇領域を抽出して保存（双方向補間使用）
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 最初のパスで全フレームのパラメータを収集
        print(f"First pass: collecting face detection data...")
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        all_params = []
        all_frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
            params = self.get_lip_region_params(frame)
            all_params.append(params)
        
        cap.release()
        
        # 有効なパラメータのインデックスを見つける
        valid_indices = [i for i, params in enumerate(all_params) if params is not None]
        
        if not valid_indices:
            raise Exception("No face detected in the entire video")
        
        print(f"Processing video: {video_path}")
        print(f"Total frames: {total_frames}, FPS: {fps}")
        print(f"Frames with detected faces: {len(valid_indices)}/{total_frames}")
        
        # パラメータの補間
        interpolated_params = []
        for i in range(total_frames):
            if all_params[i] is not None:
                interpolated_params.append(all_params[i])
                continue
            
            # 前後の有効なパラメータを探す
            prev_valid = next((idx for idx in reversed(valid_indices) if idx < i), None)
            next_valid = next((idx for idx in valid_indices if idx > i), None)
            
            if prev_valid is None and next_valid is not None:
                # 前方に有効なフレームがない場合、次の有効なフレームを使用
                interpolated_params.append(all_params[next_valid])
            elif next_valid is None and prev_valid is not None:
                # 後方に有効なフレームがない場合、前の有効なフレームを使用
                interpolated_params.append(all_params[prev_valid])
            elif prev_valid is not None and next_valid is not None:
                # 線形補間
                prev_params = all_params[prev_valid]
                next_params = all_params[next_valid]
                weight = (i - prev_valid) / (next_valid - prev_valid)
                
                interpolated_center = prev_params['center'] * (1 - weight) + next_params['center'] * weight
                interpolated_size = prev_params['size'] * (1 - weight) + next_params['size'] * weight
                interpolated_half_width = int(interpolated_size / 2)
                
                interpolated_params.append({
                    'center': interpolated_center.astype(int),
                    'size': int(interpolated_size),
                    'half_width': interpolated_half_width
                })
            else:
                # ここには到達しないはず
                raise Exception("Unexpected interpolation case")
        
        # 第二パス: 補間されたパラメータを使用して唇領域を抽出
        print("\nSecond pass: extracting lip regions...")
        start_time = time.time()
        
        for i, (frame, params) in enumerate(zip(all_frames, interpolated_params)):
            lip_region = self.extract_lip_region(frame, params)
            
            if lip_region is not None:
                output_path = os.path.join(output_dir, f'frame_{i:04d}.jpg')
                cv2.imwrite(output_path, lip_region)
            
            if i % 30 == 0:
                elapsed_time = time.time() - start_time
                fps = i / elapsed_time if elapsed_time > 0 else 0
                eta = (total_frames - i) / fps if fps > 0 else 0
                print(f"Progress: {i}/{total_frames} frames "
                      f"({i/total_frames*100:.1f}%) "
                      f"FPS: {fps:.1f} "
                      f"ETA: {eta/60:.1f}min")
        
        print(f"\nProcessing completed:")
        print(f"Total frames processed: {total_frames}")
        print(f"Frames with face detection: {len(valid_indices)}")
        print(f"Frames using interpolation: {total_frames - len(valid_indices)}")
        print(f"Total time: {(time.time()-start_time)/60:.1f} minutes")

def main():
    input_dir = '../data/original_mov'
    output_base_dir = '../data/extracted_lip'
    
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"Created input directory: {input_dir}")
        print("Please place your MP4 files in this directory and run the script again.")
        return
    
    mp4_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
    
    if not mp4_files:
        print(f"No MP4 files found in {input_dir}")
        return
    
    print(f"Found {len(mp4_files)} MP4 files to process")
    
    extractor = LipExtractor()
    
    for video_file in mp4_files:
        video_path = os.path.join(input_dir, video_file)
        video_name = os.path.splitext(video_file)[0]
        output_dir = os.path.join(output_base_dir, video_name)
        
        try:
            extractor.process_video(video_path, output_dir)
        except Exception as e:
            print(f"Error processing {video_file}: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()