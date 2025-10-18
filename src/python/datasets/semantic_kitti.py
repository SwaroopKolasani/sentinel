import numpy as np
import torch
from torch.utils.data import Dataset
import os
from google.cloud import storage

class SemanticKITTIDataset(Dataset):
    def __init__(self, root_dir, split='train', use_gcs=False, bucket_name=None, use_intensity=True):
        self.root_dir = root_dir
        self.split = split
        self.use_gcs = use_gcs
        self.bucket_name = bucket_name
        self.use_intensity = use_intensity  # Add flag for intensity
        
        # SemanticKITTI class mapping
        self.learning_map = self._get_learning_map()
        self.num_classes = 20
        
        # Define train/val/test splits
        self.sequences = {
            'train': ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10'],
            'val': ['08'],
            'test': ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
        }
        
        self.files = self._load_file_list()
        
    def _get_learning_map(self):
        # Simplified SemanticKITTI learning map
        learning_map = {
            0: 0,     # "unlabeled"
            1: 0,     # "outlier" -> "unlabeled"
            10: 1,    # "car"
            11: 2,    # "bicycle"
            13: 5,    # "bus" -> "other-vehicle"
            15: 3,    # "motorcycle"
            16: 5,    # "on-rails" -> "other-vehicle"
            18: 4,    # "truck"
            20: 5,    # "other-vehicle"
            30: 6,    # "person"
            31: 7,    # "bicyclist"
            32: 8,    # "motorcyclist"
            40: 9,    # "road"
            44: 10,   # "parking"
            48: 11,   # "sidewalk"
            49: 12,   # "other-ground"
            50: 13,   # "building"
            51: 14,   # "fence"
            52: 0,    # "other-structure" -> "unlabeled"
            60: 9,    # "lane-marking" -> "road"
            70: 15,   # "vegetation"
            71: 16,   # "trunk"
            72: 17,   # "terrain"
            80: 18,   # "pole"
            81: 19,   # "traffic-sign"
            99: 0,    # "other-object" -> "unlabeled"
            252: 1,   # "moving-car" -> "car"
            253: 7,   # "moving-bicyclist" -> "bicyclist"
            254: 6,   # "moving-person" -> "person"
            255: 8,   # "moving-motorcyclist" -> "motorcyclist"
            256: 5,   # "moving-on-rails" -> "other-vehicle"
            257: 5,   # "moving-bus" -> "other-vehicle"
            258: 4,   # "moving-truck" -> "truck"
            259: 5,   # "moving-other-vehicle" -> "other-vehicle"
        }
        return learning_map
    
    def _load_file_list(self):
        files = []
        for seq in self.sequences[self.split]:
            if self.use_gcs:
                # Load from GCS
                files.extend(self._load_gcs_files(seq))
            else:
                # Load from local directory
                seq_dir = os.path.join(self.root_dir, 'sequences', seq, 'velodyne')
                if os.path.exists(seq_dir):
                    seq_files = sorted(os.listdir(seq_dir))
                    files.extend([(seq, f[:-4]) for f in seq_files if f.endswith('.bin')])
        return files
    
    def _load_gcs_files(self, sequence):
        # Implementation for GCS file loading
        client = storage.Client()
        bucket = client.bucket(self.bucket_name)
        prefix = f'sequences/{sequence}/velodyne/'
        blobs = bucket.list_blobs(prefix=prefix)
        files = []
        for blob in blobs:
            if blob.name.endswith('.bin'):
                filename = os.path.basename(blob.name)[:-4]
                files.append((sequence, filename))
        return sorted(files)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        seq, filename = self.files[idx]
        
        # Load point cloud
        if self.use_gcs:
            points = self._load_gcs_pointcloud(seq, filename)
            labels = self._load_gcs_labels(seq, filename)
        else:
            points = self._load_local_pointcloud(seq, filename)
            labels = self._load_local_labels(seq, filename)
        
        # Apply learning map to labels
        labels = self._apply_learning_map(labels)
        
        # Sample points if too many
        if len(points) > 50000:
            indices = np.random.choice(len(points), 50000, replace=False)
            points = points[indices]
            labels = labels[indices]
        
        # Convert to torch tensors
        points = torch.from_numpy(points).float()
        labels = torch.from_numpy(labels).long()
        
        return points, labels
    
    def _load_local_pointcloud(self, seq, filename):
        file_path = os.path.join(self.root_dir, 'sequences', seq, 'velodyne', f'{filename}.bin')
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        
        # Return XYZ + intensity or just XYZ
        if self.use_intensity:
            return points  # Return all 4 channels (X, Y, Z, intensity)
        else:
            # Return XYZ with dummy intensity channel
            xyz = points[:, :3]
            intensity = np.ones((xyz.shape[0], 1), dtype=np.float32)
            return np.hstack([xyz, intensity])
    
    def _load_local_labels(self, seq, filename):
        file_path = os.path.join(self.root_dir, 'sequences', seq, 'labels', f'{filename}.label')
        labels = np.fromfile(file_path, dtype=np.uint32).reshape(-1)
        labels = labels & 0xFFFF  # Get semantic labels (lower 16 bits)
        return labels
    
    def _load_gcs_pointcloud(self, seq, filename):
        client = storage.Client()
        bucket = client.bucket(self.bucket_name)
        blob = bucket.blob(f'sequences/{seq}/velodyne/{filename}.bin')
        content = blob.download_as_bytes()
        points = np.frombuffer(content, dtype=np.float32).reshape(-1, 4)
        
        # Return XYZ + intensity or just XYZ
        if self.use_intensity:
            return points  # Return all 4 channels (X, Y, Z, intensity)
        else:
            # Return XYZ with dummy intensity channel
            xyz = points[:, :3]
            intensity = np.ones((xyz.shape[0], 1), dtype=np.float32)
            return np.hstack([xyz, intensity])
    
    def _load_gcs_labels(self, seq, filename):
        client = storage.Client()
        bucket = client.bucket(self.bucket_name)
        blob = bucket.blob(f'sequences/{seq}/labels/{filename}.label')
        content = blob.download_as_bytes()
        labels = np.frombuffer(content, dtype=np.uint32).reshape(-1)
        labels = labels & 0xFFFF
        return labels
    
    def _apply_learning_map(self, labels):
        mapped_labels = np.zeros_like(labels)
        for key, value in self.learning_map.items():
            mapped_labels[labels == key] = value
        return mapped_labels