from pathlib import Path
from p2i.ir import ModelIR

def load(path):
    return ModelIR.model_validate_json(Path(path).read_text(encoding="utf-8"))
