import inspect
from functools import lru_cache
import importlib.metadata
from pathlib import Path
from p2i.ir import CodeOrigin

@lru_cache(maxsize=1)
def distributions_by_package():
    return importlib.metadata.packages_distributions()

def origin(cls):
    module = cls.__module__
    package = module.split(".")[0]
    file = None
    start = end = None
    try:
        file = inspect.getsourcefile(cls)
        lines, start = inspect.getsourcelines(cls)
        end = start + len(lines) - 1
    except (TypeError, OSError):
        pass
    kind = "unknown"
    version = None
    distributions = distributions_by_package().get(package, [])
    if module.startswith("p2i.examples."):
        # Bundled model definitions stand in for user-authored models.
        kind = "user"
    elif package == "torch":
        kind = "framework"
    elif distributions or (file and any(p in Path(file).parts for p in ("site-packages", "dist-packages"))):
        kind = "dependency"
    elif file and not file.startswith("<"):
        kind = "user"
    # __main__ with no inspectable source is intentionally unknown.
    if distributions:
        try:
            version = importlib.metadata.version(distributions[0])
        except importlib.metadata.PackageNotFoundError:
            pass
    return CodeOrigin(kind=kind, package=package, package_version=version, file=file,
                      line_start=start, line_end=end, module_name=module, class_name=cls.__qualname__)
