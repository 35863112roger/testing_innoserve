import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_env() -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def describe(name: str, path_text: str) -> None:
    path = Path(os.path.expandvars(path_text)).expanduser()
    resolved = path.resolve() if path.exists() else path
    print(f"{name}: {'OK' if path.exists() else 'MISSING'}")
    print(f"  configured: {path}")
    print(f"  resolved:   {resolved}")
    for config_name in ("config.json", "adapter_config.json"):
        config_path = path / config_name
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            print(f"  {config_name}: {json.dumps(config, ensure_ascii=False)[:500]}")


if __name__ == "__main__":
    env = load_env()
    for variable in (
        "BERT_MODEL_PATH",
        "SUMMARY_ADAPTER_PATH",
        "SUMMARY_BASE_MODEL_PATH",
    ):
        describe(variable, env[variable])

