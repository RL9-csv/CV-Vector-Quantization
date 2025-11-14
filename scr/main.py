import os
import cv2
import glob
import numpy as np
from PIL import Image
import imagehash
import argparse

class ClassicCV:
    def __init__(self, n_features=2000, good_match_percent=0.05):
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.good_match_percent = good_match_percent

    def _match_pairs(self, v1_dir, v2_dir, hash_threshold=9):   # 최적의 임계값

        v1_files = sorted(glob.glob(os.path.join(v1_dir, '*.png')))
        v2_files = sorted(glob.glob(os.path.join(v2_dir, '*.png')))
        
        v1_hashes = {imagehash.phash(Image.open(f)): f for f in v1_files}

        matched_pairs = []
        for v2_file in v2_files:
            try:
                v2_hash = imagehash.phash(Image.open(v2_file))
            except Exception as e:
                print(f"[경고] {os.path.basename(v2_file)} 파일을 여는 데 실패: {e}")
                continue

            min_dist = float('inf')
            best_match_v1_file = None
            
            for v1_hash, v1_file in v1_hashes.items():
                dist = v1_hash - v2_hash
                if dist < min_dist:
                    min_dist = dist
                    best_match_v1_file = v1_file
                    
            if min_dist <= hash_threshold:
                print(f'  [매칭 성공] {os.path.basename(best_match_v1_file)} <-> {os.path.basename(v2_file)} (거리: {min_dist})')
                matched_pairs.append((best_match_v1_file, v2_file))
                
        print(f"--- 총 {len(matched_pairs)}개의 쌍을 찾음 ---\n")
        return matched_pairs
    
    def _align_and_diff(self, img1_path, img2_path):
        img1_color = cv2.imread(img1_path)
        img2_color = cv2.imread(img2_path)
        img1_gray = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
        
        kp1, des1 = self.orb.detectAndCompute(img1_gray, None)
        kp2, des2 = self.orb.detectAndCompute(img2_gray, None)

        if des1 is None or des2 is None:
            print(f"  [경고] {os.path.basename(img1_path)} 또는 {os.path.basename(img2_path)} 에서 특징점을 찾지 못함.")
            return None
        
        matches = self.matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        num_good_matches = int(len(matches) * self.good_match_percent)
        good_matches = matches[:num_good_matches]
        
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        if len(good_matches) < 4:
            print(f"  [경고] {os.path.basename(img1_path)} 와 {os.path.basename(img2_path)} 사이의 특징점이 부족하여 정렬 실패.")
            return None
        
        matrix, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
        
        if matrix is None:
            print(f"  [경고] {os.path.basename(img1_path)} 와 {os.path.basename(img2_path)} 의 변환 행렬 계산 실패.")
            return None
            
        h, w = img1_gray.shape
        aligned_img2_gray = cv2.warpPerspective(img2_gray, matrix, (w, h))
        
        diff = cv2.absdiff(img1_gray, aligned_img2_gray)
        blurred = cv2.GaussianBlur(diff, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result_image = img1_color.copy()
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
        return result_image
    
    def run(self, v1_dir, v2_dir, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        print(f"결과물은 '{out_dir}' 폴더에 저장됩니다.\n")
        matched_pairs = self._match_pairs(v1_dir, v2_dir)
        
        if not matched_pairs:
            print("매칭되는 도면 쌍을 찾지 못했습니다. 작전을 종료합니다.")
            return
        
        for img1_path, img2_path in matched_pairs:
            result = self._align_and_diff(img1_path, img2_path)
            
            if result is not None:
                v1_name = os.path.splitext(os.path.basename(img1_path))[0]
                v2_name = os.path.splitext(os.path.basename(img2_path))[0]
                output_filename = f'diff_{v1_name[:8]}_vs_{v2_name[:8]}.png'
                output_path = os.path.join(out_dir, output_filename)
                cv2.imwrite(output_path, result)
                print(f"  [분석 완료] {output_filename} 저장 완료.")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='두 도면 폴더를 비교하여 차이점을 찾아내는 파이프라인')
    parser.add_argument('--v1', required=True, help='원본 도면(v1)이 있는 폴더 경로')
    parser.add_argument('--v2', required=True, help='수정된 도면(v2)이 있는 폴더 경로')
    parser.add_argument('--out', required=True, help='결과 이미지를 저장할 폴더 경로')
    args = parser.parse_args()
    
    pipeline = ClassicCV()
    pipeline.run(v1_dir=args.v1, v2_dir=args.v2, out_dir=args.out)