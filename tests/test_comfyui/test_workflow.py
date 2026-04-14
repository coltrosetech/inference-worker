import json
from pathlib import Path

import pytest

from worker.comfyui.workflow import WorkflowTemplate, find_node


FIXTURE = Path(__file__).parent.parent / "fixtures" / "workflows" / "sample.json"


def test_load_template_from_file():
    t = WorkflowTemplate.from_file(FIXTURE)
    assert t.to_dict()["last_node_id"] == 5


def test_find_node_by_title():
    t = WorkflowTemplate.from_file(FIXTURE)
    n = find_node(t.to_dict(), "positive_prompt")
    assert n["id"] == 1


def test_find_node_raises_when_missing():
    t = WorkflowTemplate.from_file(FIXTURE)
    with pytest.raises(KeyError):
        find_node(t.to_dict(), "missing_title")


def test_set_widget_changes_value():
    t = WorkflowTemplate.from_file(FIXTURE)
    t.set_widget("positive_prompt", 0, "a dog on a skateboard")
    assert find_node(t.to_dict(), "positive_prompt")["widgets_values"][0] == "a dog on a skateboard"


def test_set_widget_does_not_mutate_original_file():
    t1 = WorkflowTemplate.from_file(FIXTURE)
    t2 = WorkflowTemplate.from_file(FIXTURE)
    t1.set_widget("positive_prompt", 0, "new")
    assert find_node(t2.to_dict(), "positive_prompt")["widgets_values"][0] == "a cat"


def test_set_many_widgets_chain():
    t = WorkflowTemplate.from_file(FIXTURE)
    t.set_widget("sampler", 0, 42)
    t.set_widget("sampler", 2, 12)
    t.set_widget("latent_dims", 0, 512)
    t.set_widget("latent_dims", 1, 768)
    d = t.to_dict()
    assert find_node(d, "sampler")["widgets_values"][0] == 42
    assert find_node(d, "sampler")["widgets_values"][2] == 12
    assert find_node(d, "latent_dims")["widgets_values"][0] == 512
    assert find_node(d, "latent_dims")["widgets_values"][1] == 768


def test_to_json_roundtrip():
    t = WorkflowTemplate.from_file(FIXTURE)
    raw = t.to_json()
    parsed = json.loads(raw)
    assert parsed["nodes"][0]["title"] == "positive_prompt"
