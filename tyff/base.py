"""Base pydantic models"""

import pydantic


class BaseModel(pydantic.BaseModel):
    """Base pydantic model for all other models to inherit from"""

    @classmethod
    def from_yaml(cls, yaml_str: str):
        """Create an instance of the model from a YAML string"""
        import yaml

        data = yaml.safe_load(yaml_str)
        return cls(**data)

    def to_yaml(self) -> str:
        """Convert the model instance to a YAML string"""
        import yaml

        return yaml.safe_dump(self.model_dump())
