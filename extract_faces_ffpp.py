#!/usr/bin/env python3
"""
High-quality face extraction from FaceForensics++ videos.
Extracts more frames from original sequences to balance the dataset.
"""

import cv2
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json


class FaceExtractor:
    """Extract faces from videos with high accuracy."""

    def __init__(self, output_dir, margin=0.3, min_face_size=80,
                 frames_original=128, frames_manipulated=32, target_size=224, use_alignment=True):
        """
        Args:
            output_dir: Directory to save extracted faces
            margin: Margin around detected face (0.3 = 30% padding)
            min_face_size: Minimum face size in pixels
            frames_original: Number of frames to extract per ORIGINAL video (more)
            frames_manipulated: Number of frames to extract per MANIPULATED video
            target_size: Target size for extracted faces (default: 224x224)
            use_alignment: Whether to use face alignment
        """
        self.output_dir = Path(output_dir)
        self.margin = margin
        self.min_face_size = min_face_size
        self.frames_original = frames_original  # Extract MORE from originals
        self.frames_manipulated = frames_manipulated
        self.target_size = target_size
        self.use_alignment = use_alignment

        # Initialize face detectors
        self.init_detectors()

        # Statistics
        self.stats = {
            'total_videos': 0,
            'successful_videos': 0,
            'total_faces_extracted': 0,
            'original_faces': 0,
            'manipulated_faces': 0,
            'failed_videos': []
        }

    def init_detectors(self):
        """Initialize face detection models."""
        # Primary: DNN-based detector (more accurate)
        model_path = "deploy.prototxt"
        weights_path = "res10_300x300_ssd_iter_140000.caffemodel"

        # Try to use DNN detector if available
        self.use_dnn = False
        if os.path.exists(model_path) and os.path.exists(weights_path):
            self.dnn_detector = cv2.dnn.readNetFromCaffe(model_path, weights_path)
            self.use_dnn = True
            print("Using DNN face detector (more accurate)")
        else:
            print("DNN model not found, using Haar Cascade")

        # Fallback: Haar Cascade detector
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.haar_detector = cv2.CascadeClassifier(cascade_path)

        if not self.use_dnn:
            print("Using Haar Cascade face detector")

    def detect_faces_dnn(self, frame, conf_threshold=0.8):
        """Detect faces using DNN detector (more accurate)."""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                      (104.0, 177.0, 123.0))
        self.dnn_detector.setInput(blob)
        detections = self.dnn_detector.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:  # Increased from 0.6 to 0.8 for higher quality
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                faces.append((x1, y1, x2 - x1, y2 - y1, confidence))

        return faces

    def detect_faces_haar(self, frame):
        """Detect faces using Haar Cascade."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.haar_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size)
        )

        # Convert to (x, y, w, h, confidence) format
        return [(x, y, w, h, 1.0) for (x, y, w, h) in faces]

    def detect_faces(self, frame):
        """Detect faces using the best available method."""
        if self.use_dnn:
            faces = self.detect_faces_dnn(frame, conf_threshold=0.8)
            # Only use Haar as fallback with strict threshold if DNN finds nothing
            if len(faces) == 0:
                haar_faces = self.detect_faces_haar(frame)
                # Only accept if Haar found exactly 1 large face (likely real)
                if len(haar_faces) == 1:
                    x, y, w, h, conf = haar_faces[0]
                    if w > 120 and h > 120:  # Must be reasonably large
                        faces = haar_faces
        else:
            faces = self.detect_faces_haar(frame)

        return faces

    def expand_bbox(self, x, y, w, h, img_h, img_w, margin=None):
        """Expand bounding box with margin, ensuring it stays within image."""
        if margin is None:
            margin = self.margin

        # Calculate expanded box
        margin_w = int(w * margin)
        margin_h = int(h * margin)

        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(img_w, x + w + margin_w)
        y2 = min(img_h, y + h + margin_h)

        return x1, y1, x2, y2

    def is_valid_face(self, x, y, w, h, img_h, img_w, frame=None):
        """Validate that detected region is a proper face (not ears/neck/background)."""
        # Check minimum size
        if w < self.min_face_size or h < self.min_face_size:
            return False

        # Check aspect ratio (faces are roughly 0.7 to 1.3 ratio)
        # This prevents extracting ears or neck regions
        aspect_ratio = w / h
        if aspect_ratio < 0.6 or aspect_ratio > 1.4:
            return False

        # Check if face is too close to edges (likely partial face/ear/neck)
        edge_threshold = 10
        if x < edge_threshold or y < edge_threshold:
            return False
        if x + w > img_w - edge_threshold or y + h > img_h - edge_threshold:
            return False

        # Check size relative to frame (faces shouldn't be too large - likely body/neck)
        face_area = w * h
        frame_area = img_h * img_w
        area_ratio = face_area / frame_area

        if area_ratio > 0.8:  # Face takes up more than 80% - likely not just face
            return False

        if area_ratio < 0.02:  # Face too small - likely false detection
            return False

        # NEW: Color variance check - faces have diverse colors, backgrounds are uniform
        if frame is not None:
            face_region = frame[y:y+h, x:x+w]
            if face_region.size > 0:
                # Convert to LAB color space for better skin tone detection
                try:
                    lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
                    l_channel = lab[:, :, 0]

                    # Calculate variance - faces should have reasonable variance
                    variance = np.var(l_channel)

                    # If variance is too low, it's likely a uniform background
                    if variance < 100:  # Backgrounds have low variance
                        return False

                    # Check if there's skin-like colors
                    # In LAB space, skin typically has specific ranges
                    a_channel = lab[:, :, 1]  # Green-Red
                    b_channel = lab[:, :, 2]  # Blue-Yellow

                    # Skin tones typically have higher a (red) values
                    mean_a = np.mean(a_channel)
                    if mean_a < 120 or mean_a > 150:  # Typical skin range in LAB
                        # Allow some flexibility but check if it's way off
                        if mean_a < 100 or mean_a > 170:
                            return False

                except Exception:
                    pass  # If color check fails, skip it

        return True

    def extract_frames_uniformly(self, video_path, num_frames):
        """Extract frames uniformly from video."""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < num_frames:
            num_frames = total_frames

        # Calculate frame indices to extract
        if total_frames > 0:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        else:
            cap.release()
            return []

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append((idx, frame))

        cap.release()
        return frames

    def process_video(self, video_path, video_id, category, is_original=False):
        """Extract faces from a single video."""
        # Use more frames for original videos
        num_frames = self.frames_original if is_original else self.frames_manipulated

        frames = self.extract_frames_uniformly(video_path, num_frames)

        if len(frames) == 0:
            self.stats['failed_videos'].append(str(video_path))
            return 0

        # Create output directory for this video
        output_subdir = self.output_dir / category / video_id
        output_subdir.mkdir(parents=True, exist_ok=True)

        faces_extracted = 0

        for frame_idx, frame in frames:
            h, w = frame.shape[:2]

            # Detect faces
            faces = self.detect_faces(frame)

            # Process each detected face
            for face_num, (x, y, fw, fh, conf) in enumerate(faces):
                # Validate face (ensures we don't extract ears/neck/background)
                if not self.is_valid_face(x, y, fw, fh, h, w, frame):
                    continue

                # Expand bounding box with margin
                x1, y1, x2, y2 = self.expand_bbox(x, y, fw, fh, h, w)

                # Extract face region
                face_img = frame[y1:y2, x1:x2]

                # Additional quality check - ensure extracted region has reasonable size
                if face_img.size > 0 and face_img.shape[0] > 50 and face_img.shape[1] > 50:
                    # Resize to target size (224x224)
                    face_resized = cv2.resize(face_img, (self.target_size, self.target_size),
                                             interpolation=cv2.INTER_LANCZOS4)

                    face_filename = f"frame_{frame_idx:04d}_face_{face_num}.png"
                    face_path = output_subdir / face_filename
                    cv2.imwrite(str(face_path), face_resized,
                               [cv2.IMWRITE_PNG_COMPRESSION, 3])
                    faces_extracted += 1

        # Save metadata
        metadata = {
            'video_path': str(video_path),
            'video_id': video_id,
            'category': category,
            'is_original': is_original,
            'frames_processed': len(frames),
            'faces_extracted': faces_extracted
        }

        metadata_path = output_subdir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Update stats
        if is_original:
            self.stats['original_faces'] += faces_extracted
        else:
            self.stats['manipulated_faces'] += faces_extracted

        return faces_extracted

    def process_directory(self, input_dir, category_name, is_original=False):
        """Process all videos in a directory."""
        input_path = Path(input_dir)

        # Find all video files
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov']:
            video_files.extend(list(input_path.rglob(ext)))

        video_files = sorted(video_files)  # Sort for consistency

        frames_info = self.frames_original if is_original else self.frames_manipulated
        print(f"\nProcessing {len(video_files)} videos from {category_name}")
        print(f"Extracting {frames_info} frames per video")

        for video_path in tqdm(video_files, desc=f"Extracting {category_name}"):
            self.stats['total_videos'] += 1

            # Get video ID (filename without extension)
            video_id = video_path.stem

            # Determine category from path
            if 'original_sequences' in str(video_path):
                category = 'original'
                is_orig = True
            else:
                # Extract manipulation type from path
                parts = video_path.parts
                for part in parts:
                    if part in ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']:
                        category = part
                        break
                else:
                    category = 'unknown'
                is_orig = False

            # Extract faces
            faces_count = self.process_video(video_path, video_id, category, is_orig)

            if faces_count > 0:
                self.stats['successful_videos'] += 1
                self.stats['total_faces_extracted'] += faces_count

    def save_stats(self):
        """Save extraction statistics."""
        stats_path = self.output_dir / 'extraction_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)

        print("\n" + "="*60)
        print("EXTRACTION STATISTICS")
        print("="*60)
        print(f"Total videos processed: {self.stats['total_videos']}")
        print(f"Successful videos: {self.stats['successful_videos']}")
        print(f"Total faces extracted: {self.stats['total_faces_extracted']}")
        print(f"  - Original faces: {self.stats['original_faces']}")
        print(f"  - Manipulated faces: {self.stats['manipulated_faces']}")
        print(f"Failed videos: {len(self.stats['failed_videos'])}")
        print(f"\nBalance ratio: {self.stats['original_faces'] / max(1, self.stats['manipulated_faces']):.2f}")
        print(f"Statistics saved to: {stats_path}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Extract faces from FaceForensics++ videos with balanced sampling'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Input directory containing videos'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for extracted faces'
    )
    parser.add_argument(
        '--frames_original',
        type=int,
        default=128,
        help='Number of frames to extract per ORIGINAL video (default: 128)'
    )
    parser.add_argument(
        '--frames_manipulated',
        type=int,
        default=32,
        help='Number of frames to extract per MANIPULATED video (default: 32)'
    )
    parser.add_argument(
        '--margin',
        type=float,
        default=0.3,
        help='Margin around face (default: 0.3 = 30%%)'
    )
    parser.add_argument(
        '--min_face_size',
        type=int,
        default=80,
        help='Minimum face size in pixels (default: 80)'
    )
    parser.add_argument(
        '--target_size',
        type=int,
        default=224,
        help='Target size for extracted faces (default: 224x224)'
    )

    args = parser.parse_args()

    # Create extractor
    extractor = FaceExtractor(
        output_dir=args.output_dir,
        margin=args.margin,
        min_face_size=args.min_face_size,
        frames_original=args.frames_original,
        frames_manipulated=args.frames_manipulated,
        target_size=args.target_size
    )

    # Process original sequences first (prioritize with MORE frames)
    original_dir = Path(args.input_dir) / 'original_sequences'
    if original_dir.exists():
        print("\n" + "="*60)
        print("PROCESSING ORIGINAL SEQUENCES (PRIORITY - MORE FRAMES)")
        print("="*60)
        extractor.process_directory(original_dir, 'original_sequences', is_original=True)

    # Process manipulated sequences
    manipulated_dir = Path(args.input_dir) / 'manipulated_sequences'
    if manipulated_dir.exists():
        print("\n" + "="*60)
        print("PROCESSING MANIPULATED SEQUENCES")
        print("="*60)

        # Process each manipulation type
        for manip_type in ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']:
            manip_path = manipulated_dir / manip_type
            if manip_path.exists():
                extractor.process_directory(manip_path, manip_type, is_original=False)

    # Save statistics
    extractor.save_stats()


if __name__ == '__main__':
    main()
