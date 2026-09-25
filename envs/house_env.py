"""House grid-world environment with partial observability.

Grid layout (10 × 10; row = y increases downward, col = x increases rightward):

    col:  0  1  2  3  4  5  6  7  8  9
row 0:    ■  ■  ■  ■  ■  ■  ■  ■  ■  ■
row 1:    ■  K  K  K  ■  L  L  L  L  ■
row 2:    ■  K  K  K  □  L  L  L  L  ■   □ = doorway
row 3:    ■  K  K  K  ■  L  L  L  L  ■
row 4:    ■  ■  □  ■  ■  ■  ■  □  ■  ■
row 5:    ■  B  B  B  ■  D  D  D  D  ■
row 6:    ■  B  B  B  ■  D  D  D  D  ■
row 7:    ■  B  B  B  □  D  D  D  D  ■
row 8:    ■  B  B  B  ■  D  D  D  D  ■
row 9:    ■  ■  ■  ■  ■  ■  ■  ■  ■  ■

K = Kitchen, L = Living Room, B = Bedroom, D = Dining Room
Doorways: (2,4) K↔L,  (4,2) K↔B,  (4,7) L↔D,  (7,4) B↔D
"""
from __future__ import annotations

import io
from collections import deque

import gymnasium
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from PIL import Image

# ── Grid constants ─────────────────────────────────────────────────────────────
GRID_SIZE = 10
G = GRID_SIZE

# Wall map: 1 = wall, 0 = passable floor
_W = np.ones((G, G), dtype=np.int8)
for _iy in range(1, 4):                        # rows 1–3
    for _ix in range(1, 4):  _W[_iy, _ix] = 0  # Kitchen
    for _ix in range(5, 9):  _W[_iy, _ix] = 0  # Living Room
for _iy in range(5, 9):                        # rows 5–8
    for _ix in range(1, 4):  _W[_iy, _ix] = 0  # Bedroom
    for _ix in range(5, 9):  _W[_iy, _ix] = 0  # Dining Room
# Doorways
_W[2, 4] = 0   # Kitchen ↔ Living Room
_W[4, 2] = 0   # Kitchen ↔ Bedroom
_W[4, 7] = 0   # Living Room ↔ Dining Room
_W[7, 4] = 0   # Bedroom ↔ Dining Room

WALLS    = _W
PASSABLE = WALLS == 0

# Room interior cells (iy, ix)
ROOM_CELLS = {
    'kitchen': [(iy, ix) for iy in range(1, 4) for ix in range(1, 4)],
    'living':  [(iy, ix) for iy in range(1, 4) for ix in range(5, 9)],
    'bedroom': [(iy, ix) for iy in range(5, 9) for ix in range(1, 4)],
    'dining':  [(iy, ix) for iy in range(5, 9) for ix in range(5, 9)],
}

ROOM_BOUNDS = {           # (row_min, col_min, row_max, col_max)  – inclusive
    'kitchen': (1, 1, 3, 3),
    'living':  (1, 5, 3, 8),
    'bedroom': (5, 1, 8, 3),
    'dining':  (5, 5, 8, 8),
}

# Goals are always placed in the kitchen; used to normalise goal XY positions.
GOAL_ROOM_X_MIN, GOAL_ROOM_X_MAX = 1.0, 4.0   # col range [1, 4)
GOAL_ROOM_Y_MIN, GOAL_ROOM_Y_MAX = 1.0, 4.0   # row range [1, 4)

# Required pickup order: dining-room obj (idx 2) → living-room obj (idx 0) → bedroom obj (idx 1).
# Violating this order terminates the episode immediately with reward -3.
DELIVERY_ORDER = (2, 0, 1)

ROOM_COLORS = {
    'kitchen': (0.82, 0.98, 0.82),
    'living':  (1.00, 0.94, 0.82),
    'bedroom': (0.82, 0.90, 1.00),
    'dining':  (0.96, 0.86, 1.00),
}

OBJ_COLORS = [
    (0.90, 0.20, 0.20),   # red
    (0.20, 0.50, 0.90),   # blue
    (0.90, 0.70, 0.10),   # yellow
    (0.50, 0.90, 0.30),   # green
    (0.90, 0.40, 0.70),   # pink
]

# ── Image observation rendering ─────────────────────────────────────────────────

IMG_OBS_SIZE = 80   # pixels; 10 grid cells × 8 px/cell
IMG_CELL_PX  = 8    # pixels per grid cell

# Precomputed uint8 colour palettes
_OBJ_U8 = np.array(
    [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in OBJ_COLORS],
    dtype=np.uint8,
)
# Goals: blend obj colour 35% + 65% white  →  distinctly lighter pastel
_GOAL_U8 = np.array(
    [(int((r * 0.35 + 0.65) * 255), int((g * 0.35 + 0.65) * 255), int((b * 0.35 + 0.65) * 255))
     for r, g, b in OBJ_COLORS],
    dtype=np.uint8,
)
_AGENT_U8 = np.array([255, 255, 255], dtype=np.uint8)     # white agent (distinct from all object colours)
_ARROW_U8 = np.array([80, 80, 80], dtype=np.uint8)        # dark-gray direction marker


def _build_house_bg() -> np.ndarray:
    """Build the static 80×80 background once: room fills + wall tiles."""
    img = np.full((IMG_OBS_SIZE, IMG_OBS_SIZE, 3), 130, dtype=np.uint8)
    for room, (r0, c0, r1, c1) in ROOM_BOUNDS.items():
        rc = tuple(int(v * 255) for v in ROOM_COLORS[room])
        img[r0 * IMG_CELL_PX:(r1 + 1) * IMG_CELL_PX,
            c0 * IMG_CELL_PX:(c1 + 1) * IMG_CELL_PX] = rc
    for iy in range(G):
        for ix in range(G):
            if WALLS[iy, ix]:
                img[iy * IMG_CELL_PX:(iy + 1) * IMG_CELL_PX,
                    ix * IMG_CELL_PX:(ix + 1) * IMG_CELL_PX] = (55, 55, 55)
    return img


_HOUSE_BG = _build_house_bg()   # computed once at import time


def _stamp(img: np.ndarray, cy: int, cx: int, r: int, color) -> None:
    """Fill a (2r+1)×(2r+1) square centred at pixel (cx, cy) with *color*."""
    H, W = img.shape[:2]
    img[max(0, cy - r):min(H, cy + r + 1),
        max(0, cx - r):min(W, cx + r + 1)] = color


def _make_fov_mask(agent_pos: np.ndarray, agent_theta: float,
                   fov_radius: float, fov_angle: float) -> np.ndarray:
    """Return a (IMG_OBS_SIZE, IMG_OBS_SIZE) bool mask: True where pixel is in the FOV cone."""
    rows = np.arange(IMG_OBS_SIZE, dtype=np.float32)
    cols = np.arange(IMG_OBS_SIZE, dtype=np.float32)
    world_x = cols[None, :] / IMG_CELL_PX   # (1, W)  — col maps to x
    world_y = rows[:, None] / IMG_CELL_PX   # (H, 1)  — row maps to y
    dx = world_x - agent_pos[0]
    dy = world_y - agent_pos[1]
    in_radius = (dx * dx + dy * dy) <= fov_radius ** 2
    diff = (np.arctan2(dy, dx) - agent_theta + np.pi) % (2 * np.pi) - np.pi
    in_cone = np.abs(diff) <= fov_angle / 2.0
    return in_radius & in_cone


def make_house_obs_image(
    agent_pos:        np.ndarray,   # (2,)  [x, y]
    agent_theta:      float,
    object_pos:       np.ndarray,   # (N, 2)
    object_held:      int,          # -1 = empty
    delivered:        np.ndarray,   # (N,) bool
    goal_pos:         np.ndarray,   # (N, 2)
    num_objects:      int,
    fov_radius:       float = None,
    fov_angle:        float = None,
    fully_observable: bool  = True,
) -> np.ndarray:
    """Return an (80, 80, 3) uint8 image observation.

    When *fully_observable* is False the image shows a flashlight cone:
      - Background (rooms + walls), objects, and goals are blacked out outside the FOV.
      - The agent is always rendered.

    Colour convention:
        Objects  → full OBJ_COLOR   (darker, vivid)
        Goals    → pastel tint      (lighter, same hue)
        Agent    → white square with dark-gray direction dot
        Held obj → small colour dot at agent centre
    """
    # ── Background (rooms + walls) ────────────────────────────────────────────
    if fully_observable or fov_radius is None:
        img = _HOUSE_BG.copy()
        fov_mask = None
    else:
        fov_mask = _make_fov_mask(agent_pos, agent_theta, fov_radius, fov_angle)
        img = np.zeros((IMG_OBS_SIZE, IMG_OBS_SIZE, 3), dtype=np.uint8)
        img[fov_mask] = _HOUSE_BG[fov_mask]   # reveal room/wall only inside cone

    # ── Objects (FOV-gated) ───────────────────────────────────────────────────
    for i in range(num_objects):
        if delivered[i] or i == object_held:
            continue
        ox, oy = object_pos[i]
        opx, opy = int(ox * IMG_CELL_PX), int(oy * IMG_CELL_PX)
        if fov_mask is not None:
            opy_c = int(np.clip(opy, 0, IMG_OBS_SIZE - 1))
            opx_c = int(np.clip(opx, 0, IMG_OBS_SIZE - 1))
            if not fov_mask[opy_c, opx_c]:
                continue
        _stamp(img, opy, opx, 3, _OBJ_U8[i % len(_OBJ_U8)])

    # ── Goals (FOV-gated) ─────────────────────────────────────────────────────
    for i in range(num_objects):
        if delivered[i]:
            continue
        gx, gy = goal_pos[i]
        gpx, gpy = int(gx * IMG_CELL_PX), int(gy * IMG_CELL_PX)
        if fov_mask is not None:
            gpy_c = int(np.clip(gpy, 0, IMG_OBS_SIZE - 1))
            gpx_c = int(np.clip(gpx, 0, IMG_OBS_SIZE - 1))
            if not fov_mask[gpy_c, gpx_c]:
                continue
        _stamp(img, gpy, gpx, 3, _GOAL_U8[i % len(_GOAL_U8)])

    # ── Agent (always visible) ────────────────────────────────────────────────
    ax, ay = agent_pos
    apx, apy = int(ax * IMG_CELL_PX), int(ay * IMG_CELL_PX)
    _stamp(img, apy, apx, 3, _AGENT_U8)

    # Direction indicator
    ddx = int(round(4.0 * np.cos(agent_theta)))
    ddy = int(round(4.0 * np.sin(agent_theta)))
    dpx = int(np.clip(apx + ddx, 0, IMG_OBS_SIZE - 1))
    dpy = int(np.clip(apy + ddy, 0, IMG_OBS_SIZE - 1))
    img[dpy, dpx] = _ARROW_U8

    # Held-object indicator: tiny colour dot at agent centre
    if object_held >= 0:
        _stamp(img, apy, apx, 1, _OBJ_U8[object_held % len(_OBJ_U8)])

    return img


# ── BFS path planning ───────────────────────────────────────────────────────────

def bfs_path(start_cell: tuple, goal_cell: tuple) -> list | None:
    """BFS on the passable grid.  Returns [(iy,ix)…] from start to goal, or None."""
    sy, sx = start_cell
    gy, gx = goal_cell
    if (sy, sx) == (gy, gx):
        return [(sy, sx)]
    visited: dict = {(sy, sx): None}
    queue   = deque([(sy, sx)])
    while queue:
        cy, cx = queue.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < G and 0 <= nx < G and PASSABLE[ny, nx] and (ny, nx) not in visited:
                visited[(ny, nx)] = (cy, cx)
                if (ny, nx) == (gy, gx):
                    path = []
                    cur: tuple | None = (gy, gx)
                    while cur is not None:
                        path.append(cur)
                        cur = visited[cur]
                    path.reverse()
                    return path
                queue.append((ny, nx))
    return None


def _compute_grid_dist() -> np.ndarray:
    """All-pairs BFS distance on the passable grid.

    Returns a (G*G, G*G) float32 array where entry [s, t] is the shortest
    path distance (in grid cells) from cell s to cell t, navigating only
    through passable cells.  Unreachable pairs get distance G*G (large but finite).
    """
    n = G * G
    INF = float(G * G)
    dist = np.full((n, n), INF, dtype=np.float32)
    for start_iy in range(G):
        for start_ix in range(G):
            if WALLS[start_iy, start_ix]:
                continue
            s = start_iy * G + start_ix
            dist[s, s] = 0.0
            queue = deque([(start_iy, start_ix)])
            visited = set()
            visited.add((start_iy, start_ix))
            while queue:
                cy, cx = queue.popleft()
                d = dist[s, cy * G + cx]
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = cy + dy, cx + dx
                    if (0 <= ny < G and 0 <= nx < G
                            and PASSABLE[ny, nx]
                            and (ny, nx) not in visited):
                        visited.add((ny, nx))
                        dist[s, ny * G + nx] = d + 1.0
                        queue.append((ny, nx))
    return dist


# Precomputed at import time (10×10 grid → 100×100 = 10 K entries, instant).
GRID_DIST = _compute_grid_dist()
# Maximum finite BFS distance (used for normalisation).
GRID_DIST_MAX = float(np.max(GRID_DIST[GRID_DIST < G * G]))


def grid_dist(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
    """Shortest-path distance between two continuous (x, y) positions."""
    iy_a = int(np.clip(pos_a[1], 0, G - 1))
    ix_a = int(np.clip(pos_a[0], 0, G - 1))
    iy_b = int(np.clip(pos_b[1], 0, G - 1))
    ix_b = int(np.clip(pos_b[0], 0, G - 1))
    return float(GRID_DIST[iy_a * G + ix_a, iy_b * G + ix_b])


# ── Environment ─────────────────────────────────────────────────────────────────

class HouseEnv(gymnasium.Env):
    """Grid-world house with partial observability – move objects to the kitchen.

    Observation (flat float32, length = 4 + 2·G² + N+1):
        [pose(4) | visible_grid(G²) | goal_grid(G²) | inventory(N+1)]

        pose:          (x/G, y/G, sin θ, cos θ)   — normalised continuous pose
        visible_grid:  per-cell float; -1=outside FOV, 0=empty, 1..N=object id
        goal_grid:     per-cell float;  0=no goal,      1..N=goal for object i
                       (always fully visible – the agent knows target locations)
        inventory:     one-hot; index 0 = holding nothing, i+1 = holding object i

    Action (continuous, all ∈ [-1, 1]):
        [dx, dy, dtheta, hold]

        (dx, dy)  clipped to unit circle → at most 1 grid-unit translation per step
        dtheta    scaled by max_rotation → heading change (rad)
        hold > 0  → pick up nearby object (if empty-handed); no-op if already holding
        hold < 0  → drop held object (if holding); no-op if empty-handed
    """

    metadata = {'render_modes': ['rgb_array'], 'render_fps': 10}

    def __init__(
        self,
        num_objects:    int   = 3,
        max_steps:      int   = 70,
        fov_radius:     float = 3.5,
        fov_angle:      float = 2.0 * np.pi / 3,   # 120° cone
        max_rotation:   float = np.pi / 4,          # 45° per step
        pickup_radius:  float = 0.7,
        drop_goal_radius: float = 1.0,
        render_mode:    str   = 'rgb_array',
        render_size:    int   = 500,
        seed:           int | None = None,
        count_reward:      bool  = False,
        dense_reward:      bool  = False,
        fully_observable:  bool  = False,
        small_obs:         bool  = False,
        randomization:     str   = 'large',
        image_obs:         bool  = False,
    ):
        super().__init__()
        self.num_objects      = num_objects
        self.max_steps        = max_steps
        self.fov_radius       = fov_radius
        self.fov_angle        = fov_angle
        self.max_rotation     = max_rotation
        self.pickup_radius    = pickup_radius
        self.drop_goal_radius = drop_goal_radius
        self.render_mode      = render_mode
        self.render_size      = render_size
        self.count_reward     = count_reward
        self.dense_reward     = dense_reward
        self.fully_observable = fully_observable
        self.small_obs        = small_obs
        self.randomization    = randomization
        self.image_obs        = image_obs
        self._max_dist        = GRID_DIST_MAX     # max BFS path length in grid

        if image_obs:
            # proprio: pose only (x, y, sin_theta, cos_theta)
            self.observation_space = spaces.Dict({
                'image':  spaces.Box(low=0, high=255,
                                     shape=(IMG_OBS_SIZE, IMG_OBS_SIZE, 3), dtype=np.uint8),
                'proprio': spaces.Box(low=-np.inf, high=np.inf,
                                      shape=(4,), dtype=np.float32),
            })
        else:
            state_dim = 2 * num_objects if small_obs else G * G
            self.observation_space = spaces.Dict({
                'proprio': spaces.Box(low=-np.inf, high=np.inf,
                                      shape=(4,), dtype=np.float32),
                'state':   spaces.Box(low=-1.0, high=float(num_objects + 1),
                                      shape=(state_dim,), dtype=np.float32),
            })
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32,
        )

        self.np_random       = np.random.default_rng(seed)
        self.agent_pos       = np.zeros(2, dtype=np.float32)
        self.agent_theta     = 0.0
        self.object_pos      = np.zeros((num_objects, 2), dtype=np.float32)
        self.object_held     = -1
        self.goal_pos        = np.zeros((num_objects, 2), dtype=np.float32)
        self.delivered       = np.zeros(num_objects, dtype=bool)
        self.step_count      = 0
        self.next_pickup_idx = 0   # index into DELIVERY_ORDER
        self.wrong_pickup    = False

    # ── Private helpers ──────────────────────────────────────────────────────────

    def _rand_in_room(self, room: str) -> np.ndarray:
        row_min, col_min, row_max, col_max = ROOM_BOUNDS[room]
        cx = (col_min + col_max + 1) / 2.0
        cy = (row_min + row_max + 1) / 2.0
        if self.randomization == 'low':
            return np.array([cx, cy], dtype=np.float32)
        elif self.randomization == 'medium':
            half_w = (col_max - col_min + 1) / 4.0
            half_h = (row_max - row_min + 1) / 4.0
            x = cx + self.np_random.uniform(-half_w, half_w)
            y = cy + self.np_random.uniform(-half_h, half_h)
        else:  # 'large'
            cells = ROOM_CELLS[room]
            iy, ix = cells[self.np_random.integers(len(cells))]
            x = ix + self.np_random.uniform(0.2, 0.8)
            y = iy + self.np_random.uniform(0.2, 0.8)
        return np.array([x, y], dtype=np.float32)

    def _cell(self, pos: np.ndarray) -> tuple:
        """Discrete grid cell (iy, ix) for a continuous position."""
        return int(pos[1]), int(pos[0])

    def _is_passable(self, x: float, y: float) -> bool:
        ix, iy = int(x), int(y)
        return (0 <= ix < G) and (0 <= iy < G) and bool(PASSABLE[iy, ix])

    def _fov_cells(self) -> set:
        """Set of (iy, ix) cells within the agent's field-of-view."""
        if self.fully_observable:
            return {(iy, ix) for iy in range(G) for ix in range(G)}
        px, py = self.agent_pos
        theta  = self.agent_theta
        half   = self.fov_angle / 2.0
        r_sq   = self.fov_radius ** 2
        visible = set()
        for iy in range(G):
            for ix in range(G):
                cx, cy  = ix + 0.5, iy + 0.5
                dx, dy  = cx - px,  cy - py
                if dx * dx + dy * dy > r_sq:
                    continue
                diff = (np.arctan2(dy, dx) - theta + np.pi) % (2 * np.pi) - np.pi
                if abs(diff) <= half:
                    visible.add((iy, ix))
        return visible

    def render_obs_image(self) -> np.ndarray:
        """Return an (80, 80, 3) uint8 image observation of the current state."""
        return make_house_obs_image(
            self.agent_pos, self.agent_theta,
            self.object_pos, self.object_held,
            self.delivered, self.goal_pos,
            self.num_objects,
            fov_radius=self.fov_radius,
            fov_angle=self.fov_angle,
            fully_observable=self.fully_observable,
        )

    def _get_proprio(self) -> np.ndarray:
        """Proprioceptive state: agent position and orientation only."""
        return np.array(
            [self.agent_pos[0] / G, self.agent_pos[1] / G,
             np.sin(self.agent_theta), np.cos(self.agent_theta)],
            dtype=np.float32,
        )

    def _get_obs(self):
        if self.image_obs:
            return {'image': self.render_obs_image(), 'proprio': self._get_proprio()}
        # ── state: FOV-gated object positions ────────────────────────────────────
        fov = self._fov_cells()
        if self.small_obs:
            # Compact: 2*num_objects floats — [x/G, y/G] if visible, [-1,-1] if not.
            state_parts = []
            for i in range(self.num_objects):
                if not self.delivered[i] and i != self.object_held:
                    if self.fully_observable or self._cell(self.object_pos[i]) in fov:
                        state_parts.extend([self.object_pos[i][0] / G, self.object_pos[i][1] / G])
                    else:
                        state_parts.extend([-1.0, -1.0])
                else:
                    state_parts.extend([-1.0, -1.0])
            state = np.array(state_parts, dtype=np.float32)
        else:
            state = np.full(G * G, -1.0, dtype=np.float32)
            for iy, ix in fov:
                state[iy * G + ix] = 0.0
            for i in range(self.num_objects):
                if not self.delivered[i] and i != self.object_held:
                    iy, ix = self._cell(self.object_pos[i])
                    if (iy, ix) in fov:
                        state[iy * G + ix] = float(i + 1)
        return {'proprio': self._get_proprio(), 'state': state}

    def _compute_dense_reward(self, prev_held: int) -> float:
        """Shaped reward:
          - Not holding: (1 - dist_agent→nearest_obj / max_dist)
                       + (1 - dist_nearest_obj→its_goal / max_dist)
          - Pickup event: +1 bonus
          - Holding:     (1 - dist_agent→goal / max_dist)
                       + (1 - dist_held_obj→goal / max_dist)
        """
        reward = 0.0
        if (prev_held == -1) and (self.object_held >= 0):
            reward += 1.0

        # +2 per already-delivered object
        reward += 2.0 * float(np.sum(self.delivered))

        if self.object_held >= 0:
            dist_to_goal = grid_dist(self.agent_pos, self.goal_pos[self.object_held])
            reward += (1.0 - dist_to_goal / self._max_dist)   # agent → goal
            reward += (1.0 - dist_to_goal / self._max_dist)   # held obj → goal (same pos)
        else:
            undelivered = [i for i in range(self.num_objects) if not self.delivered[i]]
            if undelivered:
                nearest = min(undelivered,
                              key=lambda i: grid_dist(self.agent_pos, self.object_pos[i]))
                dist_agent_obj = grid_dist(self.agent_pos, self.object_pos[nearest])
                dist_obj_goal  = grid_dist(self.object_pos[nearest], self.goal_pos[nearest])
                reward += (1.0 - dist_agent_obj / self._max_dist)   # agent → object
                reward += (1.0 - dist_obj_goal  / self._max_dist)   # object → its goal

        return reward

    # ── Gymnasium API ────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.step_count      = 0
        self.object_held     = -1
        self.next_pickup_idx = 0
        self.delivered[:] = False
        self.wrong_pickup    = False

        self.agent_pos   = self._rand_in_room('kitchen')
        self.agent_theta = float(self.np_random.uniform(-np.pi, np.pi))

        src_rooms = ['living', 'bedroom', 'dining']
        for i in range(self.num_objects):
            self.object_pos[i] = self._rand_in_room(src_rooms[i % len(src_rooms)])
        for i in range(self.num_objects):
            self.goal_pos[i] = self._rand_in_room('kitchen')

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        dx, dy, dtheta, hold = (
            float(action[0]), float(action[1]),
            float(action[2]), float(action[3]),
        )

        # Rotate
        self.agent_theta = (
            self.agent_theta + dtheta * self.max_rotation + np.pi
        ) % (2 * np.pi) - np.pi

        # Translate – clamp to unit circle
        mag = np.sqrt(dx * dx + dy * dy)
        if mag > 1.0:
            dx /= mag; dy /= mag

        new_x = float(np.clip(self.agent_pos[0] + dx, 0.05, G - 0.05))
        new_y = float(np.clip(self.agent_pos[1] + dy, 0.05, G - 0.05))

        if self._is_passable(new_x, new_y):
            self.agent_pos[0] = new_x
            self.agent_pos[1] = new_y
        elif self._is_passable(new_x, self.agent_pos[1]):
            self.agent_pos[0] = new_x
        elif self._is_passable(self.agent_pos[0], new_y):
            self.agent_pos[1] = new_y

        # Carried object follows the agent
        if self.object_held >= 0:
            self.object_pos[self.object_held] = self.agent_pos.copy()

        # Interaction
        prev_delivered = int(np.sum(self.delivered))
        prev_held      = self.object_held
        if hold > 0.0:
            # Pick up nearest reachable undelivered object (no-op if already holding)
            if self.object_held == -1:
                for i in range(self.num_objects):
                    if not self.delivered[i]:
                        if np.linalg.norm(self.agent_pos - self.object_pos[i]) < self.pickup_radius:
                            self.object_held = i
                            break
        elif hold < 0.0:
            # Drop held object; check whether it lands at its goal
            if self.object_held >= 0:
                h = self.object_held
                self.object_pos[h] = self.agent_pos.copy()
                self.object_held   = -1
                if np.linalg.norm(self.agent_pos - self.goal_pos[h]) < self.drop_goal_radius:
                    self.delivered[h]  = True
                    self.object_pos[h] = self.goal_pos[h].copy()

        # Enforce delivery order: wrong pickup → penalty for remainder of episode.
        self.step_count += 1
        if prev_held == -1 and self.object_held >= 0:
            required = DELIVERY_ORDER[self.next_pickup_idx] if self.next_pickup_idx < len(DELIVERY_ORDER) else -1
            if self.object_held != required:
                self.wrong_pickup = True
            else:
                self.next_pickup_idx += 1

        if self.wrong_pickup:
            reward = -5.0
        elif self.dense_reward:
            reward = self._compute_dense_reward(prev_held)
        elif self.count_reward:
            cur_delivered = int(np.sum(self.delivered))
            reward = float(cur_delivered - self.num_objects)
        else:
            cur_delivered = int(np.sum(self.delivered))
            reward = float(cur_delivered - prev_delivered)

        terminated = bool(np.all(self.delivered))
        truncated  = self.step_count >= self.max_steps

        info = {
            'objects_delivered': int(np.sum(self.delivered)),
            'total_objects':     self.num_objects,
            'success':           terminated,
            'wrong_pickup':      self.wrong_pickup,
        }
        return self._get_obs(), reward, terminated, truncated, info

    # ── Rendering ────────────────────────────────────────────────────────────────

    def render(self) -> np.ndarray:
        sz  = self.render_size
        obs = self._get_obs()

        pose = obs['proprio']  # (4,): x/G, y/G, sin_theta, cos_theta

        if self.image_obs or self.small_obs:
            # Reconstruct full vis grid from physical state for display
            fov     = self._fov_cells()
            fov_set = set(fov)
            vis_grid = np.full((G, G), -1.0, dtype=np.float32)
            for iy, ix in fov:
                vis_grid[iy, ix] = 0.0
            for i in range(self.num_objects):
                if not self.delivered[i] and i != self.object_held:
                    iy, ix = self._cell(self.object_pos[i])
                    if (iy, ix) in fov_set:
                        vis_grid[iy, ix] = float(i + 1)
            vis_flat = vis_grid.flatten()
        else:
            vis_flat = obs['state']

        # Build goal grid directly from env state.
        goal_grid = np.zeros((G, G), dtype=np.float32)
        for i in range(self.num_objects):
            gx, gy = self.goal_pos[i]
            ix = int(np.clip(gx, 0, G - 1))
            iy = int(np.clip(gy, 0, G - 1))
            goal_grid[iy, ix] = float(i + 1)

        # Inventory from env state.
        inventory = np.zeros(self.num_objects + 1, dtype=np.float32)
        if self.object_held == -1:
            inventory[0] = 1.0
        else:
            inventory[self.object_held + 1] = 1.0
        goal_title = 'goal XY→grid\n0=no goal  1..N=object i'

        # Layout: [World | vis_grid | goal_grid] on top, pose/inv strip on bottom
        fig = plt.figure(figsize=(sz * 3.2 / 100, sz / 100), dpi=100)
        gs  = fig.add_gridspec(
            nrows=2, ncols=3,
            width_ratios=[3, 2, 2],
            height_ratios=[5, 1],
            hspace=0.40, wspace=0.28,
        )
        ax_world = fig.add_subplot(gs[0, 0])
        ax_vis   = fig.add_subplot(gs[0, 1])
        ax_goal  = fig.add_subplot(gs[0, 2])
        ax_pose  = fig.add_subplot(gs[1, :])

        self._draw_world(ax_world)
        self._draw_obs_grid(ax_vis, vis_flat.reshape(G, G),
                            f'obs[4:104]  visible_grid\n'
                            f'−1=unknown  0=empty  1..N=object',
                            show_agent=True)
        self._draw_obs_grid(ax_goal, goal_grid, goal_title, show_agent=False)
        self._draw_pose_inv(ax_pose, pose, inventory)

        fig.tight_layout(pad=0.3)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img = np.array(Image.open(buf))
        return img[:, :, :3]

    def _draw_obs_grid(self, ax, grid: np.ndarray, title: str,
                       show_agent: bool = False) -> None:
        """Render a 10×10 obs slice as a colour-coded grid."""
        ax.set_xlim(0, G); ax.set_ylim(G, 0)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=6.5, pad=3)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.tick_params(left=False, bottom=False,
                       labelleft=False, labelbottom=False)

        for iy in range(G):
            for ix in range(G):
                v = float(grid[iy, ix])
                if v < 0:                          # unknown (−1)
                    fc = (0.28, 0.28, 0.28)
                    label, lc = '?', (0.6, 0.6, 0.6)
                elif v == 0:                       # empty / no goal
                    fc = (0.93, 0.93, 0.93)
                    label, lc = '0', (0.75, 0.75, 0.75)
                else:                              # object i (value = i+1)
                    fc = OBJ_COLORS[int(v) - 1 % len(OBJ_COLORS)]
                    label, lc = str(int(v)), 'white'

                ax.add_patch(patches.Rectangle(
                    (ix, iy), 1, 1,
                    facecolor=fc, linewidth=0.4,
                    edgecolor=(0.65, 0.65, 0.65), zorder=1,
                ))
                ax.text(ix + 0.5, iy + 0.5, label,
                        ha='center', va='center', fontsize=5,
                        color=lc, zorder=2)

        if show_agent:
            ax_x, ax_y = self.agent_pos
            ax.add_patch(patches.Circle(
                (ax_x, ax_y), 0.28,
                facecolor=(0.15, 0.30, 0.90), edgecolor='white',
                linewidth=1, zorder=4,
            ))
            adx = 0.28 * np.cos(self.agent_theta)
            ady = 0.28 * np.sin(self.agent_theta)
            ax.annotate('', xy=(ax_x + adx, ax_y + ady), xytext=(ax_x, ax_y),
                        arrowprops=dict(arrowstyle='->', color='white', lw=1),
                        zorder=5)

    def _draw_pose_inv(self, ax, pose: np.ndarray, inventory: np.ndarray) -> None:
        """Render pose values as horizontal bar + inventory as coloured boxes."""
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('off')

        labels  = ['x/G', 'y/G', 'sin θ', 'cos θ']
        bar_h   = 0.18
        bar_gap = 0.26
        bar_x0  = 0.06
        bar_w   = 0.38

        for i, (lbl, val) in enumerate(zip(labels, pose)):
            yc = 0.85 - i * bar_gap
            # background track
            ax.add_patch(patches.Rectangle(
                (bar_x0, yc - bar_h / 2), bar_w, bar_h,
                facecolor=(0.88, 0.88, 0.88), linewidth=0, zorder=1,
            ))
            # value fill (centred at 0.5 for [-1,1] range)
            fill_x  = bar_x0 + bar_w * 0.5
            fill_len = bar_w * 0.5 * float(val)
            ax.add_patch(patches.Rectangle(
                (min(fill_x, fill_x + fill_len), yc - bar_h / 2),
                abs(fill_len), bar_h,
                facecolor=(0.25, 0.50, 0.85), linewidth=0, zorder=2,
            ))
            # zero line
            ax.plot([fill_x, fill_x], [yc - bar_h / 2, yc + bar_h / 2],
                    color='white', lw=0.8, zorder=3)
            ax.text(bar_x0 - 0.01, yc, lbl, ha='right', va='center',
                    fontsize=6.5, family='monospace')
            ax.text(bar_x0 + bar_w + 0.01, yc, f'{val:+.3f}',
                    ha='left', va='center', fontsize=6.5, family='monospace')

        # Inventory boxes
        inv_x0 = 0.56
        n_slots = len(inventory)
        box_w = min(0.10, 0.40 / n_slots)
        inv_labels = ['∅'] + [str(i + 1) for i in range(self.num_objects)]
        inv_colors = [(0.85, 0.85, 0.85)] + list(OBJ_COLORS[:self.num_objects])

        ax.text(inv_x0, 0.92, f'obs[204:]  inventory (one-hot)',
                ha='left', va='center', fontsize=6.5, family='monospace')

        for j in range(n_slots):
            xc   = inv_x0 + j * (box_w + 0.01)
            val  = float(inventory[j])
            fc   = inv_colors[j % len(inv_colors)]
            edge = 'black' if val > 0.5 else (0.75, 0.75, 0.75)
            lw   = 2.0    if val > 0.5 else 0.5
            ax.add_patch(patches.FancyBboxPatch(
                (xc, 0.50), box_w, 0.30,
                boxstyle='round,pad=0.01',
                facecolor=fc, edgecolor=edge, linewidth=lw, zorder=1,
            ))
            ax.text(xc + box_w / 2, 0.65, inv_labels[j],
                    ha='center', va='center', fontsize=7,
                    fontweight='bold' if val > 0.5 else 'normal',
                    color='white' if j > 0 else 'black', zorder=2)
            ax.text(xc + box_w / 2, 0.42, f'{val:.1f}',
                    ha='center', va='center', fontsize=5.5,
                    color='black', zorder=2)

        ax.text(inv_x0, 0.15,
                f'obs[0:4]  pose: x/G={pose[0]:+.3f}  y/G={pose[1]:+.3f}'
                f'  sinθ={pose[2]:+.3f}  cosθ={pose[3]:+.3f}',
                ha='left', va='center', fontsize=6.5, family='monospace')

    def _draw_world(self, ax):
        ax.set_xlim(0, G); ax.set_ylim(G, 0)   # y=0 at top
        ax.set_aspect('equal'); ax.axis('off')
        obs_label = 'full' if self.fully_observable else 'partial'
        ax.set_title(
            f'Step {self.step_count}  |  Delivered: '
            f'{int(np.sum(self.delivered))}/{self.num_objects}  |  {obs_label}',
            fontsize=9, pad=3,
        )

        fov_set = self._fov_cells()

        room_labels = {
            'kitchen': 'Kitchen',
            'living':  'Living\nRoom',
            'bedroom': 'Bedroom',
            'dining':  'Dining\nRoom',
        }
        for room, (r0, c0, r1, c1) in ROOM_BOUNDS.items():
            ax.add_patch(patches.Rectangle(
                (c0, r0), c1 - c0 + 1, r1 - r0 + 1,
                facecolor=ROOM_COLORS[room], linewidth=0, zorder=0,
            ))
            ax.text(
                (c0 + c1 + 1) / 2, (r0 + r1 + 1) / 2,
                room_labels[room], ha='center', va='center',
                fontsize=7, color='gray', alpha=0.55, zorder=1,
            )

        # Walls
        for iy in range(G):
            for ix in range(G):
                if WALLS[iy, ix]:
                    ax.add_patch(patches.Rectangle(
                        (ix, iy), 1, 1, facecolor=(0.25, 0.25, 0.25),
                        linewidth=0, zorder=2,
                    ))

        # FOV highlight
        for iy, ix in fov_set:
            ax.add_patch(patches.Rectangle(
                (ix, iy), 1, 1, facecolor=(1, 1, 0.4), alpha=0.28, zorder=3,
            ))


        # Goal markers (pentagons) — goals are always known from obs
        for i in range(self.num_objects):
            if self.delivered[i]:
                continue
            gx, gy = self.goal_pos[i]
            c = OBJ_COLORS[i % len(OBJ_COLORS)]
            ax.add_patch(patches.RegularPolygon(
                (gx, gy), 5, radius=0.30, orientation=np.pi / 2,
                facecolor=(*c, 0.28), edgecolor=c, linewidth=1.5, zorder=7,
            ))
            ax.text(gx, gy, f'G{i+1}', ha='center', va='center',
                    fontsize=5.5, color=c, fontweight='bold', zorder=8)

        # Objects — always show all in visualization
        for i in range(self.num_objects):
            if self.delivered[i] or i == self.object_held:
                continue
            ox, oy = self.object_pos[i]
            c = OBJ_COLORS[i % len(OBJ_COLORS)]
            ax.add_patch(patches.FancyBboxPatch(
                (ox - 0.22, oy - 0.22), 0.44, 0.44,
                boxstyle='round,pad=0.03',
                facecolor=c, edgecolor='black', linewidth=0.8, zorder=7,
            ))
            ax.text(ox, oy, str(i + 1), ha='center', va='center',
                    fontsize=7, fontweight='bold', color='white', zorder=8)

        # Delivered (checkmark at goal) — always shown (goals always known)
        for i in range(self.num_objects):
            if not self.delivered[i]:
                continue
            gx, gy = self.goal_pos[i]
            c = OBJ_COLORS[i % len(OBJ_COLORS)]
            ax.add_patch(patches.Circle(
                (gx, gy), 0.28, facecolor=c, edgecolor='green',
                linewidth=2, zorder=9,
            ))
            ax.text(gx, gy, '✓', ha='center', va='center',
                    fontsize=7, color='white', zorder=10)

        # Agent body — always shown above fog
        ax_x, ax_y = self.agent_pos
        ax.add_patch(patches.Circle(
            (ax_x, ax_y), 0.32,
            facecolor=(0.15, 0.30, 0.90), edgecolor='navy',
            linewidth=1.5, zorder=9,
        ))
        # Direction arrow
        adx = 0.40 * np.cos(self.agent_theta)
        ady = 0.40 * np.sin(self.agent_theta)
        ax.annotate(
            '', xy=(ax_x + adx, ax_y + ady), xytext=(ax_x, ax_y),
            arrowprops=dict(arrowstyle='->', color='white', lw=1.5),
            zorder=10,
        )
        # Held-object indicator (small coloured dot on agent)
        if self.object_held >= 0:
            c = OBJ_COLORS[self.object_held % len(OBJ_COLORS)]
            ax.add_patch(patches.Circle(
                (ax_x, ax_y), 0.14, facecolor=c, edgecolor='white',
                linewidth=1, zorder=11,
            ))

    def close(self):
        plt.close('all')

    def get_normalized_score(self, total_reward: float) -> float:
        return total_reward / max(self.num_objects, 1)


# ── Obs-based expert ───────────────────────────────────────────────────────────

class ObsExpertAgent:
    """Stateful expert that acts only on the observation vector (no ground-truth state access).

    Uses:
      - pose   : own position + orientation
      - vis    : FOV grid (objects visible in cone)
      - goal   : goal grid (always fully visible; delivered goals are removed)
      - inv    : inventory one-hot (what the agent is currently holding)

    Maintains a memory of last-seen object positions for navigation.
    Explores room waypoints when a target object has not been seen yet.
    """

    # Centres of the four rooms — used for exploration when object not yet seen.
    EXPLORE_WAYPOINTS = [
        np.array([2.0, 2.0], dtype=np.float32),   # kitchen
        np.array([6.5, 2.0], dtype=np.float32),   # living room
        np.array([2.0, 6.5], dtype=np.float32),   # bedroom
        np.array([6.5, 6.5], dtype=np.float32),   # dining room
    ]

    def __init__(
        self,
        num_objects:      int   = 3,
        noise_scale:      float = 0.0,
        pickup_radius:    float = 0.7,
        drop_goal_radius: float = 1.0,
        max_rotation:     float = np.pi / 4,
        small_obs:        bool  = False,
    ):
        self.num_objects      = num_objects
        self.noise_scale      = noise_scale
        self.pickup_radius    = pickup_radius
        self.drop_goal_radius = drop_goal_radius
        self.max_rotation     = max_rotation
        self.small_obs        = small_obs
        self.reset()

    def reset(self, seed=None):
        self.known_obj_pos: dict = {}   # obj_id -> np.array([x, y])
        self.explore_idx        = 0
        self.next_delivery_idx  = 0     # index into DELIVERY_ORDER
        self._prev_held         = -1
        # 50% chance to route through living room, 50% through bedroom on first pickup.
        # Use a dedicated RNG seeded per-episode so the split is truly 50/50.
        rng = np.random.default_rng(seed)
        if rng.random() < 0.5:
            self.routing_waypoint = np.array([6.5, 2.0], dtype=np.float32)   # living room
        else:
            self.routing_waypoint = np.array([2.0, 6.5], dtype=np.float32)   # bedroom

    def _parse_obs(self, obs: dict, env: 'HouseEnv'):
        pose  = obs['proprio']  # (4,): x/G, y/G, sin_theta, cos_theta
        state = obs['state']

        ax    = float(pose[0]) * G
        ay    = float(pose[1]) * G
        theta = float(np.arctan2(pose[2], pose[3]))
        agent_pos = np.array([ax, ay], dtype=np.float32)

        object_held = int(env.object_held)

        # Update memory: record positions of objects currently in FOV.
        if self.small_obs:
            for i in range(self.num_objects):
                x_norm = float(state[2 * i])
                y_norm = float(state[2 * i + 1])
                if x_norm >= 0.0:
                    self.known_obj_pos[i] = np.array([x_norm * G, y_norm * G], dtype=np.float32)
        else:
            vis_grid = state.reshape(G, G)
            for iy in range(G):
                for ix in range(G):
                    v = float(vis_grid[iy, ix])
                    if v > 0.5:
                        obj_id = int(round(v)) - 1
                        if 0 <= obj_id < self.num_objects:
                            self.known_obj_pos[obj_id] = np.array(
                                [ix + 0.5, iy + 0.5], dtype=np.float32
                            )

        goal_positions = {i: env.goal_pos[i].copy() for i in range(self.num_objects)}

        return agent_pos, theta, object_held, goal_positions

    def _navigate(self, agent_pos: np.ndarray, theta: float, target: np.ndarray) -> np.ndarray:
        dx_raw = target[0] - agent_pos[0]
        dy_raw = target[1] - agent_pos[1]
        dist   = float(np.sqrt(dx_raw ** 2 + dy_raw ** 2))
        if dist < 1e-3:
            return np.zeros(4, dtype=np.float32)

        start = (int(agent_pos[1]), int(agent_pos[0]))
        tg    = (int(np.clip(target[1], 0, G - 1)), int(np.clip(target[0], 0, G - 1)))
        path  = bfs_path(start, tg)

        if path is not None and len(path) >= 2:
            ny, nx  = path[1]
            nav_dx  = (nx + 0.5) - agent_pos[0]
            nav_dy  = (ny + 0.5) - agent_pos[1]
            nd = float(np.sqrt(nav_dx ** 2 + nav_dy ** 2))
            if nd > 1e-3:
                nav_dx /= nd; nav_dy /= nd
        else:
            nav_dx, nav_dy = dx_raw / dist, dy_raw / dist

        desired = float(np.arctan2(nav_dy, nav_dx))
        diff    = (desired - theta + np.pi) % (2 * np.pi) - np.pi
        dtheta  = float(np.clip(diff / self.max_rotation, -1.0, 1.0))

        move = min(1.0, dist)
        return np.array([nav_dx * move, nav_dy * move, dtheta, 0.0], dtype=np.float32)

    def act(self, obs: dict, env: 'HouseEnv') -> np.ndarray:
        agent_pos, theta, object_held, goal_positions = self._parse_obs(obs, env)

        # Detect delivery: held → not held means we just dropped (assume delivered).
        if self._prev_held >= 0 and object_held == -1:
            self.next_delivery_idx = min(self.next_delivery_idx + 1, self.num_objects)
            self.routing_waypoint  = None   # routing only applies to first pickup
        self._prev_held = object_held

        if object_held >= 0:
            # Carrying something → navigate to its goal and drop.
            target = goal_positions[object_held]
            d = float(np.linalg.norm(agent_pos - target))
            if d < self.drop_goal_radius * 0.75:
                action = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)  # drop
            else:
                action = self._navigate(agent_pos, theta, target)
                action[3] = 1.0
        elif self.next_delivery_idx < self.num_objects:
            target_obj = DELIVERY_ORDER[self.next_delivery_idx]

            # On the first pickup, visit the routing waypoint first.
            if self.next_delivery_idx == 0 and self.routing_waypoint is not None:
                wp = self.routing_waypoint
                if float(np.linalg.norm(agent_pos - wp)) < 1.5:
                    self.routing_waypoint = None   # waypoint reached
                    # Jump exploration directly to the target object's room so we
                    # don't loop back through kitchen/other rooms after the detour.
                    # EXPLORE_WAYPOINTS: 0=kitchen,1=living,2=bedroom,3=dining
                    # Object ids: 0=living,1=bedroom,2=dining → room index = obj_id+1
                    if target_obj not in self.known_obj_pos:
                        self.explore_idx = target_obj + 1
                else:
                    action = self._navigate(agent_pos, theta, wp)
                    action[3] = -1.0
                    if self.noise_scale > 0.0:
                        action[:3] += np.random.randn(3).astype(np.float32) * self.noise_scale
                        action = np.clip(action, -1.0, 1.0)
                    return action

            # Navigate to the required object.
            if target_obj in self.known_obj_pos:
                target = self.known_obj_pos[target_obj]
                d = float(np.linalg.norm(agent_pos - target))
                if d < self.pickup_radius * 0.85:
                    action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # pick up
                else:
                    action = self._navigate(agent_pos, theta, target)
                    action[3] = -1.0
            else:
                # Explore to find the object.
                wp = self.EXPLORE_WAYPOINTS[self.explore_idx % len(self.EXPLORE_WAYPOINTS)]
                if float(np.linalg.norm(agent_pos - wp)) < 0.5:
                    self.explore_idx += 1
                    wp = self.EXPLORE_WAYPOINTS[self.explore_idx % len(self.EXPLORE_WAYPOINTS)]
                action = self._navigate(agent_pos, theta, wp)
                action[3] = -1.0
        else:
            action = np.zeros(4, dtype=np.float32)  # all delivered

        if self.noise_scale > 0.0:
            action[:3] += np.random.randn(3).astype(np.float32) * self.noise_scale
            action = np.clip(action, -1.0, 1.0)
        return action


# ── Dataset generation ─────────────────────────────────────────────────────────

def expert_action(env: HouseEnv, noise_scale: float = 0.0) -> np.ndarray:
    """Omniscient expert policy with BFS path planning.

    Args:
        env:         HouseEnv instance (state accessed directly).
        noise_scale: Gaussian std added to the action (0 = perfect expert).

    Returns:
        action: float32[4] array  [dx, dy, dtheta, hold]
    """
    pos   = env.agent_pos.copy()
    theta = env.agent_theta

    def _navigate(target_pos: np.ndarray) -> np.ndarray:
        """Return a movement action toward target_pos, using BFS waypoints."""
        dx_raw = target_pos[0] - pos[0]
        dy_raw = target_pos[1] - pos[1]
        dist   = float(np.sqrt(dx_raw ** 2 + dy_raw ** 2))
        if dist < 1e-3:
            return np.zeros(4, dtype=np.float32)

        # BFS to get next grid waypoint
        start = (int(pos[1]), int(pos[0]))
        tg    = (
            int(np.clip(target_pos[1], 0, G - 1)),
            int(np.clip(target_pos[0], 0, G - 1)),
        )
        path = bfs_path(start, tg)

        if path is not None and len(path) >= 2:
            # Navigate toward centre of the next grid cell on the path
            ny, nx  = path[1]
            nav_dx  = (nx + 0.5) - pos[0]
            nav_dy  = (ny + 0.5) - pos[1]
            nd = float(np.sqrt(nav_dx ** 2 + nav_dy ** 2))
            if nd > 1e-3:
                nav_dx /= nd; nav_dy /= nd
        else:
            # Fallback: move directly
            nav_dx, nav_dy = dx_raw / dist, dy_raw / dist

        # Rotate toward movement direction
        desired = float(np.arctan2(nav_dy, nav_dx))
        diff    = (desired - theta + np.pi) % (2 * np.pi) - np.pi
        dtheta  = float(np.clip(diff / env.max_rotation, -1.0, 1.0))

        move = min(1.0, dist)
        return np.array([nav_dx * move, nav_dy * move, dtheta, 0.0], dtype=np.float32)

    # --- policy logic ---
    if env.object_held == -1:
        # Find nearest undelivered object
        best_i, best_d = -1, float('inf')
        for i in range(env.num_objects):
            if not env.delivered[i]:
                d = float(np.linalg.norm(pos - env.object_pos[i]))
                if d < best_d:
                    best_d, best_i = d, i
        if best_i == -1:
            return np.zeros(4, dtype=np.float32)
        if best_d < env.pickup_radius * 0.85:
            action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)   # hold=1: pick up
        else:
            action = _navigate(env.object_pos[best_i])
            action[3] = -1.0   # hold=-1: don't pick up while navigating
    else:
        h      = env.object_held
        target = env.goal_pos[h]
        d      = float(np.linalg.norm(pos - target))
        if d < env.drop_goal_radius * 0.75:
            action = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)  # hold=-1: drop
        else:
            action = _navigate(target)
            action[3] = 1.0    # hold=1: keep holding while navigating

    if noise_scale > 0.0:
        action[:3] += np.random.randn(3).astype(np.float32) * noise_scale
        action = np.clip(action, -1.0, 1.0)
    return action


def generate_house_dataset(
    env_name: str,
    num_expert_episodes:     int = 500,
    num_suboptimal_episodes: int = 0,
    seed:                    int = 0,
    count_reward:            bool = False,
    fully_observable:        bool = False,
    small_obs:               bool = False,
    max_steps:               int = 70,
    randomization:           str = 'large',
    image_obs:               bool = False,
    cache_dir:               str | None = None,
):
    """Collect an offline dataset using the expert (+ noisy) policy.

    When *image_obs* is True the stored observations are (80,80,3) uint8 images
    rendered from the ground-truth state (the expert still navigates using the
    compact flat obs so it works correctly).

    Results are cached to *cache_dir* as an .npz file keyed by all parameters.
    Pass cache_dir=None to disable caching.

    Returns:
        (train_dataset, val_dataset) as Dataset objects compatible with the
        rest of the MTQL training pipeline.
    """
    import hashlib, json, os
    from utils.datasets import Dataset

    # ── Cache lookup ────────────────────────────────────────────────────────────
    if cache_dir is not None:
        cache_key = json.dumps(dict(
            env_name=env_name,
            num_expert=num_expert_episodes,
            num_suboptimal=num_suboptimal_episodes,
            seed=seed,
            count_reward=count_reward,
            fully_observable=fully_observable,
            small_obs=small_obs,
            max_steps=max_steps,
            randomization=randomization,
            image_obs=image_obs,
            obs_format='v2',  # v2: image_obs stores dict {image, proprio}
        ), sort_keys=True)
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:16]
        cache_file = os.path.join(cache_dir, f'house_{cache_hash}.npz')

        if os.path.exists(cache_file):
            print(f'[generate_house_dataset] loading cached dataset from {cache_file}')
            d = np.load(cache_file)
            act_arr      = d['actions']
            rew_arr      = d['rewards']
            terminal_arr = d['terminals']
            mask_arr     = d['masks']
            if image_obs:
                obs_arr  = {'image': d['obs_image'],      'proprio': d['obs_proprio']}
                nobs_arr = {'image': d['next_obs_image'], 'proprio': d['next_obs_proprio']}
            else:
                obs_arr  = {'proprio': d['obs_proprio'],      'state': d['obs_state']}
                nobs_arr = {'proprio': d['next_obs_proprio'], 'state': d['next_obs_state']}
            n       = len(act_arr)
            n_val   = max(1, n // 10)
            n_train = n - n_val

            def _make_cached(sl):
                return Dataset.create(
                    observations={k: v[sl] for k, v in obs_arr.items()},
                    actions=act_arr[sl],
                    rewards=rew_arr[sl],
                    next_observations={k: v[sl] for k, v in nobs_arr.items()},
                    terminals=terminal_arr[sl],
                    masks=mask_arr[sl],
                )
            return _make_cached(slice(None, n_train)), _make_cached(slice(n_train, None))
    else:
        cache_file = None

    # ── Collection ──────────────────────────────────────────────────────────────
    num_objects = _parse_num_objects(env_name)
    rng = np.random.default_rng(seed)

    print(f'[generate_house_dataset] collecting {num_expert_episodes} expert + {num_suboptimal_episodes} suboptimal episodes for {env_name}')

    obs_buf, act_buf, rew_buf, next_obs_buf, terminal_buf, mask_buf = (
        [] for _ in range(6)
    )

    ep_returns   = {'expert': [], 'suboptimal': []}
    ep_successes = {'expert': 0,  'suboptimal': 0}
    ep_lengths   = {'expert': [], 'suboptimal': []}

    for ep_idx in range(num_expert_episodes + num_suboptimal_episodes):
        is_expert  = ep_idx < num_expert_episodes
        split      = 'expert' if is_expert else 'suboptimal'
        noise      = 0.05 if is_expert else 0.35
        ep_seed    = int(rng.integers(0, 2 ** 31))

        # Always use flat obs env for the expert (ObsExpertAgent needs flat obs).
        env   = HouseEnv(num_objects=num_objects, seed=ep_seed, count_reward=count_reward,
                          fully_observable=fully_observable, small_obs=small_obs, max_steps=max_steps,
                          randomization=randomization)
        agent = ObsExpertAgent(num_objects=num_objects, noise_scale=noise, small_obs=small_obs)
        flat_obs, _ = env.reset(seed=ep_seed)
        agent.reset(seed=ep_seed)
        done   = False
        ep_ret = 0.0
        ep_len = 0

        while not done:
            # Build current obs from pre-step state.
            if image_obs:
                cur_obs = {'image': env.render_obs_image(), 'proprio': env._get_proprio()}
            else:
                cur_obs = flat_obs

            action = agent.act(flat_obs, env)
            flat_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if image_obs:
                next_obs = {'image': env.render_obs_image(), 'proprio': env._get_proprio()}
            else:
                next_obs = flat_obs

            obs_buf.append(cur_obs)
            act_buf.append(action)
            rew_buf.append(reward)
            next_obs_buf.append(next_obs)
            terminal_buf.append(1.0 if terminated else 0.0)
            mask_buf.append(0.0 if terminated else 1.0)

            ep_ret += reward
            ep_len += 1

        ep_returns[split].append(ep_ret)
        ep_lengths[split].append(ep_len)
        if terminated:
            ep_successes[split] += 1
        env.close()

    for split, n_eps in [('expert', num_expert_episodes), ('suboptimal', num_suboptimal_episodes)]:
        if n_eps == 0:
            continue
        rets = ep_returns[split]
        lens = ep_lengths[split]
        print(
            f'  [{split}] episodes: {n_eps} | '
            f'success: {ep_successes[split]}/{n_eps} ({100*ep_successes[split]/n_eps:.1f}%) | '
            f'return: {np.mean(rets):.2f} ± {np.std(rets):.2f} (min {np.min(rets):.1f}, max {np.max(rets):.1f}) | '
            f'length: {np.mean(lens):.1f} ± {np.std(lens):.1f}'
        )

    act_arr      = np.array(act_buf,      dtype=np.float32)
    rew_arr      = np.array(rew_buf,      dtype=np.float32)
    terminal_arr = np.array(terminal_buf, dtype=np.float32)
    mask_arr     = np.array(mask_buf,     dtype=np.float32)

    # All obs are now dicts; pack them into per-key arrays.
    if image_obs:
        obs_arr  = {'image':  np.array([o['image']  for o in obs_buf],       dtype=np.uint8),
                    'proprio': np.array([o['proprio'] for o in obs_buf],      dtype=np.float32)}
        nobs_arr = {'image':  np.array([o['image']  for o in next_obs_buf],  dtype=np.uint8),
                    'proprio': np.array([o['proprio'] for o in next_obs_buf], dtype=np.float32)}
    else:
        obs_arr  = {'proprio': np.array([o['proprio'] for o in obs_buf],      dtype=np.float32),
                    'state':   np.array([o['state']   for o in obs_buf],      dtype=np.float32)}
        nobs_arr = {'proprio': np.array([o['proprio'] for o in next_obs_buf], dtype=np.float32),
                    'state':   np.array([o['state']   for o in next_obs_buf], dtype=np.float32)}

    # ── Cache save ──────────────────────────────────────────────────────────────
    if cache_file is not None:
        os.makedirs(cache_dir, exist_ok=True)
        save_kwargs = dict(actions=act_arr, rewards=rew_arr,
                           terminals=terminal_arr, masks=mask_arr)
        if image_obs:
            save_kwargs.update(obs_image=obs_arr['image'], obs_proprio=obs_arr['proprio'],
                               next_obs_image=nobs_arr['image'], next_obs_proprio=nobs_arr['proprio'])
        else:
            save_kwargs.update(obs_proprio=obs_arr['proprio'], obs_state=obs_arr['state'],
                               next_obs_proprio=nobs_arr['proprio'], next_obs_state=nobs_arr['state'])
        np.savez_compressed(cache_file, **save_kwargs)
        print(f'[generate_house_dataset] cached dataset to {cache_file}')

    # 90/10 train/val split
    n      = len(act_arr)
    n_val  = max(1, n // 10)
    n_train = n - n_val
    print(f'[generate_house_dataset] collected {n} transitions -> train: {n_train}, val: {n_val}')

    def _make(sl):
        obs_sl  = {k: v[sl] for k, v in obs_arr.items()}
        nobs_sl = {k: v[sl] for k, v in nobs_arr.items()}
        return Dataset.create(
            observations=obs_sl,
            actions=act_arr[sl],
            rewards=rew_arr[sl],
            next_observations=nobs_sl,
            terminals=terminal_arr[sl],
            masks=mask_arr[sl],
        )

    return _make(slice(None, n_train)), _make(slice(n_train, None))


def _parse_num_objects(env_name: str) -> int:
    """Extract num_objects from env names like 'house-n3-v0'."""
    if '-n' in env_name:
        try:
            return int(env_name.split('-n')[1].split('-')[0])
        except (ValueError, IndexError):
            pass
    return 3


def make_house_env(env_name: str, count_reward: bool = False, dense_reward: bool = False,
                   fully_observable: bool = False, small_obs: bool = False,
                   max_steps: int = 70, randomization: str = 'large',
                   image_obs: bool = False) -> HouseEnv:
    return HouseEnv(num_objects=_parse_num_objects(env_name), count_reward=count_reward,
                    dense_reward=dense_reward, fully_observable=fully_observable,
                    small_obs=small_obs, max_steps=max_steps, randomization=randomization,
                    image_obs=image_obs)


# ── Eval / video script ────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse, os, imageio

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name',     default='house-n3-v0')
    parser.add_argument('--num_episodes', type=int, default=20)
    parser.add_argument('--noise',        type=float, default=0.05)
    parser.add_argument('--num_videos',   type=int, default=3)
    parser.add_argument('--out_dir',      default='expert_videos')
    parser.add_argument('--fps',          type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    num_objects = _parse_num_objects(args.env_name)
    rng = np.random.default_rng(0)

    returns, lengths, successes, objects_delivered = [], [], [], []

    for ep in range(args.num_episodes):
        ep_seed   = int(rng.integers(0, 2 ** 31))
        env       = HouseEnv(num_objects=num_objects, seed=ep_seed)
        agent     = ObsExpertAgent(num_objects=num_objects, noise_scale=args.noise)
        obs, _    = env.reset(seed=ep_seed)
        agent.reset(seed=ep_seed)

        record  = ep < args.num_videos
        frames  = []
        ep_ret, ep_len = 0.0, 0
        done = False

        while not done:
            if record:
                frames.append(env.render())
            action = agent.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done    = terminated or truncated
            ep_ret += reward
            ep_len += 1

        returns.append(ep_ret)
        lengths.append(ep_len)
        successes.append(int(terminated))
        objects_delivered.append(int(ep_ret))   # reward = 1 per delivered object

        if record:
            path = os.path.join(args.out_dir, f'ep{ep:03d}.gif')
            imageio.mimsave(path, frames, fps=args.fps)
            print(f'  saved {path}  (ret={ep_ret:.0f}, len={ep_len}, success={bool(terminated)})')

        env.close()

    print(f'\n── ObsExpert stats over {args.num_episodes} episodes (noise={args.noise}) ──')
    print(f'  success rate : {np.mean(successes)*100:.1f}%  ({sum(successes)}/{args.num_episodes})')
    print(f'  return       : {np.mean(returns):.2f} ± {np.std(returns):.2f}  '
          f'(min {np.min(returns):.1f}, max {np.max(returns):.1f})')
    print(f'  ep length    : {np.mean(lengths):.1f} ± {np.std(lengths):.1f}')
    print(f'  objects/ep   : {np.mean(objects_delivered):.2f} / {num_objects}')
