import json
from pathlib import Path
from datetime import datetime


class ModelVersioning:
    def __init__(self, version_file: Path = Path('models/saved_model/version.json')):
        self.version_file = version_file
        self.version_info = self._load_or_create_version_file()

    def _load_or_create_version_file(self):
        if self.version_file.exists():
            with open(self.version_file, 'r') as f:
                return json.load(f)
        else:
            initial_version = {
                'version': '1.0.0',
                'last_updated': datetime.now().isoformat(),
                'training_history': []
            }
            self._save_version_info(initial_version)
            return initial_version

    def _save_version_info(self, version_info):
        self.version_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.version_file, 'w') as f:
            json.dump(version_info, f, indent=2)

    def update_version(self, training_metrics=None):
        """Update version after training"""
        # Increment version
        major, minor, patch = map(int, self.version_info['version'].split('.'))
        patch += 1
        if patch >= 10:
            minor += 1
            patch = 0
        if minor >= 10:
            major += 1
            minor = 0

        self.version_info['version'] = f"{major}.{minor}.{patch}"
        self.version_info['last_updated'] = datetime.now().isoformat()

        # Add training metrics
        if training_metrics:
            self.version_info['training_history'].append({
                'version': self.version_info['version'],
                'timestamp': datetime.now().isoformat(),
                'metrics': training_metrics
            })

        self._save_version_info(self.version_info)
        return self.version_info['version']