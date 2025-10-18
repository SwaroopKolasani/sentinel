import yaml
import os

class TrainingConfig:
    def __init__(self, config_path=None):
        self.config = {
            'model': {
                'num_classes': 20,
                'architecture': 'pointnet2',
            },
            'training': {
                'batch_size': 2,
                'learning_rate': 0.001,
                'epochs': 100,
                'optimizer': 'adam',
                'weight_decay': 0.0001,
                'scheduler': {
                    'type': 'step',
                    'step_size': 20,
                    'gamma': 0.7
                }
            },
            'data': {
                'num_points': 50000,
                'augmentation': {
                    'random_rotate': True,
                    'random_scale': True,
                    'random_jitter': True,
                    'random_dropout': True
                }
            },
            'gcs': {
                'bucket_name': 'sentinel-kitti-data',
                'project_id': 'bug-sync-467815'
            },
            'paths': {
                'checkpoint_dir': '/content/drive/MyDrive/project-sentinel/models',
                'log_dir': '/content/drive/MyDrive/project-sentinel/logs',
                'data_dir': '/content/kitti_data'
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                custom_config = yaml.safe_load(f)
                self._update_config(self.config, custom_config)
    
    def _update_config(self, base, custom):
        for key, value in custom.items():
            if isinstance(value, dict) and key in base:
                self._update_config(base[key], value)
            else:
                base[key] = value
    
    def get(self, key_path, default=None):
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value