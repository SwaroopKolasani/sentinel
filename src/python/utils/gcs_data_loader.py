import numpy as np
import torch
from torch.utils.data import Dataset
from google.cloud import storage
from pathlib import Path
import pickle

# This is a standard label mapping for SemanticKITTI to 20 classes.
LABEL_MAP = {
    0: 0, 1: 0, 10: 1, 11: 2, 13: 3, 15: 4, 16: 5, 18: 6, 20: 7, 30: 8, 31: 9,
    32: 10, 40: 11, 44: 12, 48: 13, 49: 14, 50: 15, 51: 16, 52: 17, 60: 18,
    70: 19, 71: 19, 72: 14, 80: 17, 81: 17, 99: 0
}

class GCSKITTIDataset(Dataset):
    def __init__(self, bucket_name, sequences, num_points=8192, cache_dir='/content/kitti_cache'):
        self.bucket_name = bucket_name
        self.sequences = sequences
        self.num_points = num_points # <-- ADDED: Number of points to sample
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.file_list_cache_path = self.cache_dir / f'file_list_cache_{"_".join(sequences)}.pkl'

        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

        if self.file_list_cache_path.exists():
            print(f"Loading file list from cache: {self.file_list_cache_path}")
            with open(self.file_list_cache_path, 'rb') as f:
                self.files = pickle.load(f)
        else:
            print("Building file list from GCS bucket (this will happen only once per sequence set)...")
            self.files = self._build_and_cache_file_list()

        print(f"Dataset initialized with {len(self.files)} samples.")

    def _build_and_cache_file_list(self):
        files = []
        for seq in self.sequences:
            prefix = f'sequences/{seq}/velodyne/'
            blobs = self.bucket.list_blobs(prefix=prefix)
            for blob in blobs:
                if blob.name.endswith('.bin'):
                    filename = blob.name.split('/')[-1]
                    frame_num = filename.replace('.bin', '')
                    files.append({
                        'sequence': seq, 'frame': frame_num,
                        'velodyne_path': f'sequences/{seq}/velodyne/{filename}',
                        'label_path': f'sequences/{seq}/labels/{frame_num}.label'
                    })
        
        sorted_files = sorted(files, key=lambda x: (x['sequence'], x['frame']))
        with open(self.file_list_cache_path, 'wb') as f:
            pickle.dump(sorted_files, f)
        return sorted_files

    def _download_blob_to_cache(self, blob_path):
        cache_path = self.cache_dir / blob_path
        if not cache_path.exists():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            blob = self.bucket.blob(blob_path)
            if blob.exists():
                blob.download_to_filename(str(cache_path))
            else:
                return None
        return cache_path

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_info = self.files[idx]
        velodyne_cache_path = self._download_blob_to_cache(file_info['velodyne_path'])
        if velodyne_cache_path is None:
            raise FileNotFoundError(f"Velodyne file not found: {file_info['velodyne_path']}")
        
        points = np.fromfile(str(velodyne_cache_path), dtype=np.float32).reshape(-1, 4)
        
        label_cache_path = self._download_blob_to_cache(file_info['label_path'])
        if label_cache_path is not None:
            labels = np.fromfile(str(label_cache_path), dtype=np.uint32)
            semantic_labels = self._map_labels(labels & 0xFFFF)
        else:
            semantic_labels = np.zeros(points.shape[0], dtype=np.int64)

        # --- THIS IS THE NEW SAMPLING LOGIC ---
        num_available_points = len(points)
        if num_available_points >= self.num_points:
            # More points than needed, so we randomly sample
            choice_indices = np.random.choice(num_available_points, self.num_points, replace=False)
        else:
            # Fewer points than needed, so we sample with replacement (padding)
            choice_indices = np.random.choice(num_available_points, self.num_points, replace=True)
        
        points = points[choice_indices, :]
        semantic_labels = semantic_labels[choice_indices]
        # --- END OF NEW LOGIC ---

        points_tensor = torch.from_numpy(points).float()
        labels_tensor = torch.from_numpy(semantic_labels).long()

        return points_tensor, labels_tensor

    def _map_labels(self, labels):
        mapped_labels = np.copy(labels)
        for k, v in LABEL_MAP.items():
            mapped_labels[labels == k] = v
        return mapped_labels

def create_dataloaders(bucket_name, batch_size=8, num_workers=2, num_points=8192):
    train_sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
    val_sequences = ['08']

    train_dataset = GCSKITTIDataset(
        bucket_name=bucket_name,
        sequences=train_sequences,
        num_points=num_points # <-- Pass num_points
    )
    val_dataset = GCSKITTIDataset(
        bucket_name=bucket_name,
        sequences=val_sequences,
        num_points=num_points # <-- Pass num_points
    )
    # ... (rest of the function is the same)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader