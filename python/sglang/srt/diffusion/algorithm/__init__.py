import importlib
import logging
import pkgutil

logger = logging.getLogger(__name__)

def import_algorithms():
    mapping = {}
    package_name = "sglang.srt.diffusion.algorithm"
    package = importlib.import_module(package_name)
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        if ispkg:
            continue
        try:
            module = importlib.import_module(name)
        except Exception as e:
            logger.warning(f"Ignore import error when loading {name}: {e}")
            continue
        if not hasattr(module, "Algorithm"):
            continue

        algo = module.Algorithm
        mapping[algo.__name__] = algo

    return mapping

def get_algorithm(
    name: str,
    block_size: int,
):
    try:
        return algo_name_to_cls[name](block_size)
    except:
        raise RuntimeError(f"Unkown diffusion algorithm: {name}")

algo_name_to_cls = import_algorithms()
