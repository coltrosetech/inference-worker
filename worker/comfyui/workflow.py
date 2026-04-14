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
