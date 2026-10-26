"""The recurrent activations shared by simulation, rendering, and the executor."""

from dataclasses import dataclass
import math

import torch


(X, Y, ANGLE, Z, HEALTH, ARMOR, BULLETS, SHELLS, WEAPON, SHOTGUN_OWNED,
 COOLDOWN, DAMAGE_FLASH, PICKUP_FLASH, TIME, KILLS, WON, FIRE_FLASH, SECRETS,
 USE_HELD, ARMOR_TYPE, RNG) = range(21)
SCALAR_COUNT = 21
CLOCK_PERIOD = 96.0
PROJECTILE_SPEED = 350.0
# Original MT_TROOPSHOT: https://github.com/id-Software/DOOM/blob/master/linuxdoom-1.10/info.c#L1874-L1897
PROJECTILE_RADIUS = 6.0
PROJECTILE_HEIGHT = 8.0
PROJECTILE_LAUNCH_HEIGHT = 40.0
AIM_HEIGHT = 28.0
IMP_MELEE_RANGE = 64.0
PLAYER_RADIUS = 16.0
ENEMY_RADIUS = 20.0
ENEMY_TYPES = (3004, 9, 3001)

ENTITY_FIELDS = (
    "entity_x", "entity_y", "entity_hp", "entity_active", "entity_cooldown",
    "entity_alive", "entity_drop", "entity_angle", "projectile_x",
    "projectile_y", "projectile_z", "projectile_dx", "projectile_dy",
    "projectile_dz", "projectile_alive",
)
SECTOR_FIELDS = (
    "sector_floor", "sector_ceiling", "sector_motion", "sector_timer",
    "sector_floor_motion", "sector_floor_timer", "sector_secret",
)


def moving_actors(types):
    return torch.isin(types, torch.tensor(ENEMY_TYPES, device=types.device))


def sector_height_bounds(data):
    # Turbo lowering may target an adjacent floor plus eight units.
    return (data["sector_floor"].min(), data["sector_floor"].max() + 8,
            data["sector_ceiling"].max())


@dataclass
class StateLayout:
    entity_count: int
    sector_count: int
    line_count: int

    def __post_init__(self):
        if min(self.entity_count, self.sector_count, self.line_count) <= 0:
            raise ValueError("a supported level must contain entities, sectors, and lines")
        offset = SCALAR_COUNT
        for fields, count in ((ENTITY_FIELDS, self.entity_count),
                              (SECTOR_FIELDS, self.sector_count),
                              (("line_used",), self.line_count)):
            for field in fields:
                setattr(self, field, slice(offset, offset + count))
                offset += count
        self.size = offset

    @classmethod
    def from_data(cls, data):
        return cls(len(data["entity_type"]), len(data["sector_floor"]),
                   len(data["lines"]))


def initial_state(data):
    """Construct spawn activations once; no host game decisions occur during play."""
    from .geometry import Geometry

    layout = StateLayout.from_data(data)
    state = torch.zeros(layout.size, dtype=torch.float32)
    state[X:Y + 1] = data["player_start"][:2]
    state[ANGLE] = data["player_start"][2]
    support, _ = Geometry(data).body_opening(state[X:Y + 1].reshape(1, 2),
                                            data["sector_floor"], data["sector_ceiling"],
                                            torch.tensor([PLAYER_RADIUS]))
    state[Z] = support[0]
    state[HEALTH] = 100
    state[BULLETS] = 50
    state[WEAPON] = 1
    state[RNG] = 1
    state[layout.entity_x] = data["entity_pos"][:, 0]
    state[layout.entity_y] = data["entity_pos"][:, 1]
    thing = data["entity_type"]
    state[layout.entity_hp] = (
        (thing == 3004).float() * 20 + (thing == 9).float() * 30
        + (thing == 3001).float() * 60 + (thing == 2035).float() * 20
    )
    state[layout.entity_alive] = 1
    state[layout.entity_angle] = data["entity_angle"]
    state[layout.sector_floor] = data["sector_floor"]
    state[layout.sector_ceiling] = data["sector_ceiling"]
    return state


def state_bounds(data):
    """Construct finite float32 validation intervals for source E1M1 state.

    Actors start inside the closed WAD boundary and cannot tunnel through it:
    their maximum step of 300/35 units is below the smallest moving-body radius
    of 16 units.
    Radius-6 endpoint clearance prevents wall tunneling: the horizontal step
    is below the diameter of 12. Inclusive sweeps and floor/ceiling impacts
    allow only one horizontal or vertical death-step outside the map envelope.
    Inactive slots include their initial zero coordinates and retain their
    final collision point.
    Projectile launches are at least 64 units from the player; their vertical
    slope follows the floor-height difference plus the two aim offsets.
    The velocity domains round outward with one unit of headroom for source
    float32 normalization; death-step coordinates add another unit. The strict
    compiler must compose its arithmetic error bounds and validate the actual
    neural activations against these domains. They are metadata, never output
    repair clamps.
    """
    from .runtime import MAX_DT

    layout = StateLayout.from_data(data)
    spawn = initial_state(data)
    lower, upper = torch.empty_like(spawn), torch.empty_like(spawn)
    assigned = torch.zeros(layout.size, dtype=torch.bool)

    def bound(field, low, high):
        lower[field], upper[field] = low, high
        assigned[field] = True

    points = torch.cat((data["vertices"], data["entity_pos"], data["player_start"][:2][None, :]))
    xy_low, xy_high = points.amin(0), points.amax(0)
    floor_low, floor_high, ceiling_high = sector_height_bounds(data)
    enemy = (spawn[layout.entity_hp] > 0) & (data["entity_type"] != 2035)
    horizontal_speed = PROJECTILE_SPEED + 1
    vz_low = math.floor(min(0., float(PROJECTILE_SPEED * (floor_low - floor_high + AIM_HEIGHT
                                                        - PROJECTILE_LAUNCH_HEIGHT) / IMP_MELEE_RANGE))) - 1
    vz_high = math.ceil(max(0., float(PROJECTILE_SPEED * (floor_high - floor_low + AIM_HEIGHT
                                                       - PROJECTILE_LAUNCH_HEIGHT) / IMP_MELEE_RANGE))) + 1
    for field, low, high in (
        (X, xy_low[0], xy_high[0]), (Y, xy_low[1], xy_high[1]),
        (ANGLE, 0, 2 * math.pi), (Z, floor_low, floor_high),
        (HEALTH, 0, 200), (ARMOR, 0, 200), (BULLETS, 0, 200), (SHELLS, 0, 50),
        (WEAPON, 0, 2), (SHOTGUN_OWNED, 0, 1), (COOLDOWN, 0, .9),
        (DAMAGE_FLASH, 0, 1), (PICKUP_FLASH, 0, .65), (TIME, 0, CLOCK_PERIOD),
        (KILLS, 0, int(enemy.sum())), (WON, 0, 1), (FIRE_FLASH, 0, .12),
        (SECRETS, 0, int((data["sector_special"] == 9).sum())),
        (USE_HELD, 0, 1), (ARMOR_TYPE, 0, 2), (RNG, 0, 32748),
    ):
        bound(field, low, high)
    bound(layout.entity_x, xy_low[0], xy_high[0])
    bound(layout.entity_y, xy_low[1], xy_high[1])
    bound(layout.entity_hp, 0, spawn[layout.entity_hp])
    for field in (layout.entity_active, layout.entity_alive, layout.entity_drop,
                  layout.projectile_alive, layout.sector_secret, layout.line_used):
        bound(field, 0, 1)
    bound(layout.entity_cooldown, 0, 2)
    # Original angles are nonnegative; awakened enemies use atan2 in [-pi,pi].
    bound(layout.entity_angle, -math.pi, 2 * math.pi)
    for field, axis in ((layout.projectile_x, 0), (layout.projectile_y, 1)):
        bound(field, min(0., float(xy_low[axis] - horizontal_speed * MAX_DT - 1)),
               max(0., float(xy_high[axis] + horizontal_speed * MAX_DT + 1)))
    bound(layout.projectile_z, min(0., float(floor_low) + vz_low * MAX_DT - 1),
           max(0., float(ceiling_high), float(floor_high) + PROJECTILE_LAUNCH_HEIGHT)
           + vz_high * MAX_DT + 1)
    bound(layout.projectile_dx, -horizontal_speed, horizontal_speed)
    bound(layout.projectile_dy, -horizontal_speed, horizontal_speed)
    bound(layout.projectile_dz, vz_low, vz_high)
    bound(layout.sector_floor, floor_low, floor_high)
    bound(layout.sector_ceiling, floor_low, max(float(ceiling_high), float(floor_high)))
    bound(layout.sector_motion, -1, 2)
    bound(layout.sector_timer, 0, 4)
    bound(layout.sector_floor_motion, -1, 2)
    bound(layout.sector_floor_timer, 0, 3)
    assert bool(assigned.all()), "every recurrent field must have declared bounds"
    assert bool(torch.isfinite(lower).all() & torch.isfinite(upper).all()), "state bounds must be finite"
    assert bool((lower <= upper).all()), "state lower bounds must not exceed upper bounds"
    assert bool(((spawn >= lower) & (spawn <= upper)).all()), "original spawn violates declared state bounds"
    return lower, upper
