from .ir import ModelIR
from .analyze.tracer import trace
from .serialization.json import load

from .architecture import ArchitectureIR, build, register_module_adapter, ModuleAdapter, ModuleAdapterRegistry
from .harness import Harness

__version__ = "0.4.0"

def serve(ir, *, host="127.0.0.1", port=8000):
    """Serve with the optional GUI package, retaining the Phase 1 API."""
    try:
        from p2i_gui import serve as gui_serve
    except ImportError as exc:
        raise RuntimeError("Install p2i-gui to use p2i.serve()") from exc
    return gui_serve(ir, host=host, port=port)

__all__ = ["ModelIR", "trace", "load", "serve", "Harness", "ArchitectureIR", "build", "register_module_adapter", "ModuleAdapter", "ModuleAdapterRegistry"]

from .skills import SkillRegistry, SkillSpec, TensorContract, ingest_skill, extract_skill, skill_from_subgraph, compare_evaluations
from .actions import ReplaceWithSkill, InsertSkill
__all__ += ['SkillRegistry','SkillSpec','TensorContract','ingest_skill','extract_skill','skill_from_subgraph','compare_evaluations','ReplaceWithSkill','InsertSkill']
