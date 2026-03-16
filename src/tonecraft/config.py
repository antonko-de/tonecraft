"""Load and validate tonecraft.toml project configuration."""

import logging
import sys
from pathlib import Path

from tonecraft.schemas import ProjectConfig

logger = logging.getLogger(__name__)

# tomllib is stdlib on Python 3.11+. On 3.10 we fall back to the
# third-party `tomli` package (same API, aliased to the same name so
# the rest of this module doesn't need to care which one was imported).
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def load_config(path: str | Path) -> ProjectConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Copy examples/tonecraft.example.toml to tonecraft.toml and edit it to get started."
        )
    logger.debug("Loading config from %s", path)
    with open(path, "rb") as f:
        data = tomllib.load(f)
    config = ProjectConfig.model_validate(data)
    logger.info("Config loaded: provider=%s, slm=%s", config.generation.provider, config.target.slm)
    return config
