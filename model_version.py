from clearml import Model

class ModelVersioning:
    def __init__(self, project_name, model_name):
        self.project_name = project_name
        self.model_name = model_name

    def _get_latest_version_from_clearml(self):
        """Fetch the latest version of the model from ClearML"""
        models = Model.query_models(project_name=self.project_name, model_name=self.model_name)
        if not models:
            return '1.0.0'  # Default version if no model exists
        latest_model = sorted(models, key=lambda m: m.published, reverse=True)[0]
        latest_version = latest_model.tags[-1] if latest_model.tags else '1.0.0'
        return latest_version

    def increment_version(self, current_version):
        """Increment the version number"""
        major, minor, patch = map(int, current_version.split('.'))
        patch += 1
        if patch >= 10:
            minor += 1
            patch = 0
        if minor >= 10:
            major += 1
            minor = 0
        return f"{major}.{minor}.{patch}"

    def get_next_version(self):
        """Fetch the latest version from ClearML and increment it"""
        latest_version = self._get_latest_version_from_clearml()
        return self.increment_version(latest_version)
