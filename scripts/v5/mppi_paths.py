"""Locations of rendering code and the external hose-routing runtime/assets.

Set HOSE_ROUTING_ROOT if hose-routing is not a sibling of this checkout.
"""
import os
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
RENDER_REPO = Path(os.environ.get('BLENDER_ROBOT_RENDER', SCRIPTS.parents[1])).expanduser().resolve()
HOSE_ROUTING_ROOT = Path(os.environ.get('HOSE_ROUTING_ROOT', RENDER_REPO.parent / 'hose-routing')).expanduser().resolve()
