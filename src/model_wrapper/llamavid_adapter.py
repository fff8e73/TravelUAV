# src/model_wrapper/llamavid_adapter.py
"""
Adapter for llamavid package.
If llamavid is installed/available, this module forwards to it.
Otherwise provides clear stubs that raise informative errors or minimal placeholders.
"""

from types import SimpleNamespace
import importlib

INSTALLED = False

# Attempt to import real llamavid
try:
    llamavid = importlib.import_module("llamavid")
    # forward commonly-used symbols
    try:
        from llamavid.model.builder import load_pretrained_model  # type: ignore
    except Exception:
        load_pretrained_model = None

    try:
        from llamavid.model.vis_traj_arch import VisionTrajectoryGenerator  # type: ignore
    except Exception:
        VisionTrajectoryGenerator = None

    try:
        from llamavid import conversation as conversation_lib  # type: ignore
    except Exception:
        conversation_lib = None

    # try to import a few constants; fallback to None if missing
    try:
        from llamavid.constants import (
            DEFAULT_IMAGE_PATCH_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
            WAYPOINT_LABEL_TOKEN,
        )
    except Exception:
        DEFAULT_IMAGE_PATCH_TOKEN = None
        DEFAULT_IM_START_TOKEN = None
        DEFAULT_IM_END_TOKEN = None
        WAYPOINT_LABEL_TOKEN = None

    INSTALLED = True
except Exception:
    # Provide stubs
    load_pretrained_model = None

    class VisionTrajectoryGenerator:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("VisionTrajectoryGenerator unavailable: llamavid not installed.")

    conversation_lib = SimpleNamespace()
    DEFAULT_IMAGE_PATCH_TOKEN = None
    DEFAULT_IM_START_TOKEN = None
    DEFAULT_IM_END_TOKEN = None
    WAYPOINT_LABEL_TOKEN = None

# Utility helpers
def require_llamavid(func_name: str):
    raise RuntimeError(
        f"'{func_name}' requires the 'llamavid' package. "
        "Either install llamavid or restore the archived source. "
        "If you intentionally removed llamavid, implement an alternative."
    )