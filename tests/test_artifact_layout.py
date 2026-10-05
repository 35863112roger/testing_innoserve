from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_model_links_resolve() -> None:
    expected = {
        "bert-context": "model.safetensors",
        "summary-adapter": "adapter_model.safetensors",
        "summary-base": "model.safetensors.index.json",
    }
    for directory, required_file in expected.items():
        assert (ROOT / "model_artifacts" / directory / required_file).is_file()

