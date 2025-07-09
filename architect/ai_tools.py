import json
import os
import shutil
from datetime import datetime

from .objects import ProjectObject
from .storage import save_project, load_project
from .mermaid_generator import generate_mermaid_diagram

# In a real app, this would be tied to a user session
PROJECT_NAME = "default_project"

def get_project_data():
    """Helper to load project data."""
    return load_project(PROJECT_NAME) or {"objects": {}}

def save_project_data(data):
    """Helper to save project data."""
    save_project(PROJECT_NAME, data)

def create_object(type: str, title: str, description: str, parent_id: str = None):
    """Create a new project object and assign unique ID"""
    project_data = get_project_data()
    new_obj = ProjectObject(type, title, description, parent_id)
    project_data["objects"][new_obj.id] = new_obj.to_dict()
    save_project_data(project_data)
    update_mermaid_diagram() # Automate diagram update
    return f"Created {type}: {title} (ID: {new_obj.id})"

def update_object(obj_id: str, new_info: str):
    """Update existing object with additional information"""
    project_data = get_project_data()
    if obj_id not in project_data["objects"]:
        return f"Error: Object with ID {obj_id} not found."
    
    project_data["objects"][obj_id]['description'] += "\n" + new_info
        
    project_data["objects"][obj_id]['updated_at'] = datetime.utcnow().isoformat()
    save_project_data(project_data)
    update_mermaid_diagram() # Automate diagram update
    return f"Updated object {obj_id}."

def delete_object(obj_id: str):
    """Delete an existing project object by its ID."""
    project_data = get_project_data()
    if obj_id in project_data["objects"]:
        del project_data["objects"][obj_id]
        # Also remove any relationships pointing to this object
        for obj in project_data["objects"].values():
            obj['relationships'] = [rel for rel in obj['relationships'] if rel['target_id'] != obj_id]
        save_project_data(project_data)
        update_mermaid_diagram() # Automate diagram update
        return f"Deleted object with ID: {obj_id}."
    else:
        return f"Error: Object with ID {obj_id} not found."

def link_objects(obj1_id: str, obj2_id: str, relationship: str):
    """Create relationship between objects (solves, depends_on, targets, requires, etc.)"""
    project_data = get_project_data()
    if obj1_id not in project_data["objects"] or obj2_id not in project_data["objects"]:
        return "Error: One or both object IDs not found."
    
    # Add relationship to obj1
    obj1 = project_data["objects"][obj1_id]
    if 'relationships' not in obj1:
        obj1['relationships'] = []
    obj1['relationships'].append({"target_id": obj2_id, "type": relationship})
    
    save_project_data(project_data)
    update_mermaid_diagram() # Automate diagram update
    return f"Linked {obj1_id} to {obj2_id} as '{relationship}'."

def delete_link(obj1_id: str, obj2_id: str, relationship: str):
    """Delete a specific relationship between two objects."""
    project_data = get_project_data()
    if obj1_id not in project_data["objects"]:
        return f"Error: Object with ID {obj1_id} not found."

    obj1 = project_data["objects"][obj1_id]
    original_relationships = list(obj1.get('relationships', []))
    
    obj1['relationships'] = [
        rel for rel in original_relationships 
        if not (rel['target_id'] == obj2_id and rel['type'] == relationship)
    ]

    if len(obj1['relationships']) < len(original_relationships):
        save_project_data(project_data)
        update_mermaid_diagram() # Automate diagram update
        return f"Deleted relationship '{relationship}' from {obj1_id} to {obj2_id}."
    else:
        return f"Error: Relationship '{relationship}' from {obj1_id} to {obj2_id} not found."


def update_mermaid_diagram():
    """Generate/update Mermaid diagram based on current objects and relationships"""
    project_data = get_project_data()
    mermaid_code = generate_mermaid_diagram(project_data.get('objects', {}))
    
    # Save to markdown file
    with open("project_diagram.md", "w") as f:
        f.write(f"```mermaid\n{mermaid_code}\n```")
    
    message = "Mermaid diagram updated in project_diagram.md."

    # Attempt to generate SVG if mmdc is available
    try:
        if shutil.which("mmdc"):
            # Write the mermaid code to a temporary .mmd file for mmdc
            with open("project_diagram.mmd", "w") as f:
                f.write(mermaid_code)
            os.system("mmdc -i project_diagram.mmd -o project_diagram.svg")
            os.remove("project_diagram.mmd") # Clean up temporary file
            message += " and project_diagram.svg"
        else:
            message += " (Install mermaid-cli for SVG export: npm install -g @mermaid-js/mermaid-cli)"
    except Exception as e:
        message += f" (Error generating SVG: {e})"

    return message

def show_project_summary():
    """Display current project objects and their relationships"""
    project_data = get_project_data()
    objects = project_data.get('objects', {})
    if not objects:
        return "No objects have been created yet."
    
    summary = []
    for obj_id, obj_data in objects.items():
        summary.append(f"- {obj_data['type']} '{obj_data['title']}' (ID: {obj_id})")
        for rel in obj_data.get('relationships', []):
            summary.append(f"  - {rel['type']}: {rel['target_id']}")
    return "\n".join(summary)

def customize_object_types(new_types: list):
    """Allow user to define custom object types beyond defaults (Conceptual)"""
    # This is a conceptual function. In a real app, this would modify a config file
    # or a user-specific setting. For now, it just returns a message.
    return f"Conceptual: Would update config with new types: {new_types}"