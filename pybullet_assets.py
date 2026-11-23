"""Asset loading for the PyBullet GUI, using gym-pybullet-drones meshes.

Pure-pybullet helper (no NumPy/Torch) usable both inside the isolated render
server process and by the in-process PyBulletViewer. It loads:

  - drones      : assets/cf2x.urdf (+ cf2.dae mesh) from gym-pybullet-drones,
                  tinted per role (3 pursuers + evader)
  - obstacles   : translucent grey cylinders matching the 6V2 SDGCR GUI
  - ground      : plane.urdf from pybullet_data (grid-textured ground plane)
  - fence       : four low blue walls matching the 6V2 SDGCR GUI

Mesh location defaults to /media/yudong/data/gym-pybullet-drones-main but can
be overridden with the MTV2_PYBULLET_ASSETS env var or --assets-path so the
GUI does not depend on a hard-coded absolute path.

Coordinate convention (same as the sim env): arena centred at origin, drones
fly at flight_height above the plane, obstacles are vertical cylinders of
radius r and height >= flight_height*1.8 (matching the old debug-line look).
"""
import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

DEFAULT_ASSETS_ROOT = Path("/media/yudong/data/gym-pybullet-drones-main")
_GPB_SUBDIR = "gym_pybullet_drones/assets"   # package layout


def assets_root() -> Path:
    override = os.environ.get("MTV2_PYBULLET_ASSETS")
    if override:
        return Path(override)
    # accept either the top-level dir or the assets dir itself
    candidate = DEFAULT_ASSETS_ROOT / _GPB_SUBDIR
    if candidate.is_dir():
        return DEFAULT_ASSETS_ROOT
    if DEFAULT_ASSETS_ROOT.is_dir():
        return DEFAULT_ASSETS_ROOT
    return DEFAULT_ASSETS_ROOT


def _asset_path(name: str) -> str:
    root = assets_root()
    for sub in ("", _GPB_SUBDIR, "assets"):
        p = root / sub / name
        if p.exists():
            return str(p)
    return str(root / name)


# ---------------------------------------------------------------------------
# Loaders (each returns the created body id or list of ids)
# ---------------------------------------------------------------------------


def load_drone(bullet, client, color=None, base_position=(0.0, 0.0, 0.0)):
    """Load the Crazyflie 2X URDF mesh as a kinematic visual body.

    cf2x.urdf is a collision+visual model with an implicit body frame that
    roughly matches the sim's flight_height convention once we offset it.
    We strip dynamics (mass 0) so the GUI only ever repositions it, never
    simulates it -- consistent with the rest of the viewer.
    """
    flags = bullet.URDF_USE_INERTIA_FROM_FILE if hasattr(
        bullet, "URDF_USE_INERTIA_FROM_FILE") else 0
    uid = bullet.loadURDF(
        _asset_path("cf2x.urdf"),
        basePosition=list(base_position),
        useFixedBase=True,
        flags=flags,
        physicsClientId=client,
    )
    if color is not None:
        # tint every visual link of the loaded drone
        _recolor_body(bullet, client, uid, color)
    return uid


def _recolor_body(bullet, client, uid, color):
    data = bullet.getVisualShapeData(uid, physicsClientId=client)
    for shape in data:
        link_index = shape[1]
        bullet.changeVisualShape(
            uid,
            linkIndex=link_index,
            rgbaColor=list(color),
            physicsClientId=client,
        )


def load_obstacle(bullet, client, x, y, radius, height,
                  model="box.urdf", color=None):
    """Build the 6V2-style cylinder at the obstacle's exact footprint.

    Returns the created body id. The pillar is a fixed-base visual body only:
    the GUI never simulates collisions, consistent with the rest of the
    viewer. ``model`` remains accepted for compatibility; ``color`` overrides
    the reference GUI's translucent grey material.
    """
    del model
    body_rgba = list(color) if color is not None else [0.55, 0.55, 0.58, 0.85]
    height = max(float(height), 0.6)
    radius = max(float(radius), 0.05)
    body_vis = bullet.createVisualShape(
        bullet.GEOM_CYLINDER,
        radius=radius,
        length=height,
        rgbaColor=body_rgba,
        physicsClientId=client,
    )
    uid = bullet.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=body_vis,
        basePosition=[float(x), float(y), height / 2.0],
        physicsClientId=client,
    )
    return uid


def load_goal(bullet, client, position, radius, color=None):
    """Create the small translucent goal sphere used by the 6V2 GUI."""
    rgba = list(color) if color is not None else [0.90, 0.15, 0.10, 0.45]
    marker_radius = max(0.12, float(radius) * 0.65)
    visual = bullet.createVisualShape(
        bullet.GEOM_SPHERE,
        radius=marker_radius,
        rgbaColor=rgba,
        physicsClientId=client,
    )
    return bullet.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=visual,
        basePosition=[float(position[0]), float(position[1]), marker_radius],
        physicsClientId=client,
    )


def load_ground(bullet, client, half_extent):
    """Load plane.urdf as a large static ground visual.

    plane.urdf ships with pybullet_data, not with gym-pybullet-drones, so we
    add pybullet_data to the search path when available.
    """
    del half_extent
    try:
        import pybullet_data
        plane_path = os.path.join(pybullet_data.getDataPath(), "plane.urdf")
    except Exception:
        plane_path = "plane.urdf"
    uid = bullet.loadURDF(
        plane_path,
        basePosition=[0.0, 0.0, -0.02],
        useFixedBase=True,
        physicsClientId=client,
    )
    return uid


def load_fence(bullet, client, half_extent, height=0.35, color=None):
    """Draw the four low blue walls used by the 6V2 SDGCR GUI.

    Returns the four collision-free visual body ids.
    """
    color = color or [0.15, 0.35, 0.85, 0.65]
    ids = []
    thickness = 0.04
    specs = (
        ([0.0, -half_extent, height / 2.0],
         [half_extent, thickness, height / 2.0]),
        ([0.0, half_extent, height / 2.0],
         [half_extent, thickness, height / 2.0]),
        ([-half_extent, 0.0, height / 2.0],
         [thickness, half_extent, height / 2.0]),
        ([half_extent, 0.0, height / 2.0],
         [thickness, half_extent, height / 2.0]),
    )
    for position, half_extents in specs:
        visual = bullet.createVisualShape(
            bullet.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=color,
            physicsClientId=client,
        )
        body = bullet.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual,
            basePosition=position,
            physicsClientId=client,
        )
        ids.append(body)
    return ids


def config_from_env():
    """Optional JSON config passed by the parent (e.g. asset override)."""
    raw = os.environ.get("MTV2_PYBULLET_ASSETS_JSON")
    if raw:
        try:
            return json.loads(raw)
        except Exception:
            return {}
    return {}


if __name__ == "__main__":
    print("assets root:", assets_root())
    print("cf2x exists:", Path(_asset_path("cf2x.urdf")).exists())
    print("architrave exists:", Path(_asset_path("architrave.urdf")).exists())
    print("box exists:", Path(_asset_path("box.urdf")).exists())
