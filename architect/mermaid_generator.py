from .config import DEFAULT_OBJECT_TYPES

# Mermaid node shapes based on object type
NODE_SHAPES = {
    "Goal": ("(", ")"),
    "Problem": ("[", "]"),
    "User": ("((", "))"),
    "Solution": ("{{", "}}"),
    "Task": ("[[", "]]"),
    "Resource": ("{", "}"),
    "Risk": ("[/", "\\]"),
    "Metric": ("([", "])"),
}

def generate_mermaid_diagram(objects):
    """Generate a Mermaid.js diagram from project objects."""
    if not objects:
        return "graph TD\n    A[Start your project by creating an object!];"

    lines = ["graph TD"]

    # Define nodes
    for obj_id, obj_data in objects.items():
        shape = NODE_SHAPES.get(obj_data['type'], ("[", "]"))
        lines.append(f"    {obj_id}{shape[0]}{obj_data['title']}{shape[1]}")

    # Define relationships
    for obj_id, obj_data in objects.items():
        for rel in obj_data.get('relationships', []):
            lines.append(f"    {obj_id} -->|{rel['type']}| {rel['target_id']}")
            
    return "\n".join(lines)

