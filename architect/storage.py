import json
import os

DATA_DIR = "projects"

def get_project_path(project_name):
    """Get the full path for a project's JSON file."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    return os.path.join(DATA_DIR, f"{project_name}.json")

def save_project(project_name, data):
    """Save project data to a JSON file."""
    with open(get_project_path(project_name), "w") as f:
        json.dump(data, f, indent=4)

def load_project(project_name):
    """Load project data from a JSON file."""
    path = get_project_path(project_name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None

def list_projects():
    """List all available projects."""
    if not os.path.exists(DATA_DIR):
        return []
    return [f.replace(".json", "") for f in os.listdir(DATA_DIR) if f.endswith(".json")]
