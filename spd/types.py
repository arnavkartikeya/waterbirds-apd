from pathlib import Path
from typing import Annotated

from pydantic import BeforeValidator, Field, PlainSerializer

from spd.utils import from_root_path, to_root_path

WANDB_PATH_PREFIX = "wandb:"


def validate_path(v: str | Path) -> str | Path:
    """Check if wandb path. If not, convert to relative to repo root."""
    if isinstance(v, str) and v.startswith(WANDB_PATH_PREFIX):
        return v
    return to_root_path(v)


# Type for paths that can either be wandb paths (starting with "wandb:")
# or regular paths (converted to be relative to repo root)
ModelPath = Annotated[
    str | Path,
    BeforeValidator(validate_path),
    PlainSerializer(lambda x: str(from_root_path(x)) if isinstance(x, Path) else x),
]

# This is a type for pydantic configs that will convert all relative paths
# to be relative to the root of this repository
RootPath = Annotated[
    Path, BeforeValidator(to_root_path), PlainSerializer(lambda x: str(from_root_path(x)))
]

TrigParams = tuple[float, float, float, float, float, float, float]

Probability = Annotated[float, Field(strict=True, ge=0, le=1)]
