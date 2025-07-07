import uuid
from datetime import datetime

class ProjectObject:
    """Base class for all project objects."""
    def __init__(self, obj_type: str, title: str, description: str, parent_id: str = None):
        self.id = f"{obj_type.lower()}-{uuid.uuid4().hex[:6]}"
        self.type = obj_type
        self.title = title
        self.description = description
        self.parent_id = parent_id
        self.created_at = datetime.utcnow().isoformat()
        self.updated_at = self.created_at
        self.relationships = []

    def to_dict(self):
        """Convert object to a dictionary."""
        return self.__dict__

    @classmethod
    def from_dict(cls, data):
        """Create an object from a dictionary."""
        obj = cls(data['type'], data['title'], data['description'], data.get('parent_id'))
        obj.id = data['id']
        obj.created_at = data['created_at']
        obj.updated_at = data['updated_at']
        obj.relationships = data.get('relationships', [])
        return obj

    def add_relationship(self, target_id: str, relationship_type: str):
        """Add a relationship to another object."""
        self.relationships.append({
            "target_id": target_id,
            "type": relationship_type
        })
        self.updated_at = datetime.utcnow().isoformat()
