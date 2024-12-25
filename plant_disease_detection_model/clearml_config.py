from clearml import Task
from pathlib import Path
import os

def init_clearml():
    """Initialize ClearML configuration"""
    Task.set_credentials(
        api_host="https://api.clear.ml",
        web_host="https://app.clear.ml",
        files_host="https://files.clear.ml",
        key="835FC4669AGN89YFOB250Q0YPVQGPT",
        secret="r_2ii52kYIlU-LGe7wrj4OPQ4nCDiAilchZcAPsDft11mWTfNU-nMMaLUexuyeLhY74"
    )

def create_task(project_name: str, task_name: str, output_uri: str):
    """Create a new ClearML task"""
    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        output_uri=output_uri
    )
    return task

def log_model_artifacts(task, model_path: Path, class_names_path: Path):
    """Log model artifacts to ClearML"""
    task.upload_artifact("model", artifact_object=model_path)
    task.upload_artifact("class_names", artifact_object=class_names_path)
    task.close()