from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def find_node(workflow: dict, title: str) -> dict:
    """Return the node dict whose `title` field matches `title`."""
    for node in workflow.get("nodes", []):
        if node.get("title") == title:
            return node
    raise KeyError(f"no node titled {title!r} in workflow")


class WorkflowTemplate:
    """In-memory mutable copy of a ComfyUI full-format workflow."""

    def __init__(self, data: dict) -> None:
        self._data = copy.deepcopy(data)

    @classmethod
    def from_file(cls, path: Path | str) -> WorkflowTemplate:
        return cls(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, data: dict) -> WorkflowTemplate:
        return cls(data)

    def to_dict(self) -> dict:
        return copy.deepcopy(self._data)

    def to_json(self) -> str:
        return json.dumps(self._data, ensure_ascii=False)

    def set_widget(self, title: str, index: int, value: Any) -> None:
        """Set widgets_values[index] on the node with the given title."""
        node = find_node(self._data, title)
        widgets = node.setdefault("widgets_values", [])
        while len(widgets) <= index:
            widgets.append(None)
        widgets[index] = value

    def set_property(self, title: str, key: str, value: Any) -> None:
        """Set node['properties'][key] on the node with the given title."""
        node = find_node(self._data, title)
        node.setdefault("properties", {})[key] = value

    # --- API-format (ComfyUI /prompt) helpers ---
    # In API format, the workflow is a flat {node_id: {class_type, inputs}} dict
    # keyed by node name. Presets patch inputs via set_input.

    def is_api_format(self) -> bool:
        """True if top-level keys look like {name: {class_type, inputs}}."""
        if not isinstance(self._data, dict) or not self._data:
            return False
        first = next(iter(self._data.values()))
        return isinstance(first, dict) and "class_type" in first

    def set_input(self, node_name: str, key: str, value: Any) -> None:
        """Set inputs[key] on the API-format node with the given name."""
        if node_name not in self._data:
            raise KeyError(f"no node named {node_name!r} in workflow")
        self._data[node_name].setdefault("inputs", {})[key] = value
