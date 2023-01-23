import pathlib
from typing import Union, Iterable, Tuple, Dict, List

PathLike = Union[pathlib.Path, str]

# graph defs
Vertex = int
EdgeList = Iterable[Tuple[Vertex, Vertex]]
AdjList = Dict[Vertex, List[Vertex]]
