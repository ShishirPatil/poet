import os
import pathlib
from typing import Union, Iterable, Tuple, Dict, List

PathLike = Union[pathlib.Path, str]

# graph defs
Vertex = int
EdgeList = Iterable[Tuple[Vertex, Vertex]]
AdjList = Dict[Vertex, List[Vertex]]


# environment variables
ENV_VAR_FLAGS = ["DEBUG_SCHEDULER_RAM"]
active_env_var_flags = {key for key in ENV_VAR_FLAGS if key in os.environ and os.environ[key].lower() in ("true", "t", "1")}
