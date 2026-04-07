from pathlib import Path

from opensportslib.core.utils.config import (
    dict_to_namespace,
    expand,
    load_classes,
    load_gz_json,
    load_json,
    store_gz_json,
    store_json,
)


def test_json_store_and_load_roundtrip(tmp_path):
    payload = {"a": 1, "b": ["x", "y"]}

    json_path = tmp_path / "payload.json"
    gz_path = tmp_path / "payload.json.gz"

    store_json(str(json_path), payload, pretty=True)
    store_gz_json(str(gz_path), payload)

    assert load_json(str(json_path)) == payload
    assert load_gz_json(str(gz_path)) == payload


def test_expand_returns_absolute_path(tmp_path):
    rel_path = Path("relative") / "path"
    expanded = expand(str(rel_path))

    assert Path(expanded).is_absolute()


def test_load_classes_supports_list():
    classes = ["PASS", "SHOT", "GOAL"]
    mapping = load_classes(classes)

    assert mapping == {"PASS": 1, "SHOT": 2, "GOAL": 3}


def test_dict_to_namespace_preserves_classes_shape():
    data = {"DATA": {"classes": ["A", "B"], "num_classes": 2}}
    ns = dict_to_namespace(data)

    assert ns.DATA.classes == ["A", "B"]
    assert ns.DATA.num_classes == 2
