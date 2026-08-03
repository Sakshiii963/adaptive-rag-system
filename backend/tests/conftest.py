"""Shared test configuration."""

import os

os.environ["ENVIRONMENT"] = "test"
os.environ["DEBUG"] = "true"

from backend.app.core.config import get_settings

get_settings.cache_clear()
