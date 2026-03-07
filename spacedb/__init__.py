# theSpaceDB — A self-evolving cognitive database.
# Blocks in free space. Distance evolves. Clusters emerge. Personalities form.
# Nothing fixed. Everything evolving. We chase infinity.

from .client import SpaceClient
from .space  import Space
from .query  import QueryBuilder
from .drift  import DriftController

__version__ = "0.1.0"
__all__ = ["SpaceClient", "Space", "QueryBuilder", "DriftController"]
