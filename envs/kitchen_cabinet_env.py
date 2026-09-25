"""
Kitchen Cabinet Search environment.

A mobile manipulator (PandaOmron) is placed in front of three adjacent
wall-mounted cabinets. A single graspable object is hidden in one of the
three cabinets (chosen uniformly at random). All cabinet doors start closed.
The episode succeeds as soon as the robot's gripper makes contact with the
target object.

Episode metadata includes which cabinet index (0/1/2) holds the object so
that scripted / oracle policies can use it.
"""

from __future__ import annotations

import numpy as np

from robocasa.environments.kitchen.kitchen import Kitchen
from robocasa.models.fixtures import FixtureType
from robocasa.models.fixtures.fixture_utils import fixture_is_type
import robocasa.utils.env_utils as EnvUtils
import robocasa.utils.object_utils as OU


class KitchenCabinetSearch(Kitchen):
    """
    Find-and-grasp task across three adjacent wall-mounted cabinets.

    Args:
        num_cabs (int): Number of cabinets to use (default 3).  The
            environment will try to find this many adjacent cabinets; if the
            layout has fewer it will use however many are available.
    """

    def __init__(self, num_cabs: int = 3, *args, **kwargs):
        self.num_cabs = num_cabs
        # Will be set during _setup_kitchen_references
        self.cabs: list = []
        self.target_cab_idx: int = 0
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        # Collect all wall-mounted cabinets that have doors.
        all_cabs = self.get_fixture(
            FixtureType.CABINET_WITH_DOOR, return_all=True
        )
        if all_cabs is None or len(all_cabs) == 0:
            raise RuntimeError(
                "KitchenCabinetSearch: no CABINET_WITH_DOOR fixtures found "
                "in the current layout."
            )

        # Filter to top-row wall cabinets: their absolute z > 0.8 m.
        top_cabs = [c for c in all_cabs if c.pos[2] > 0.8]
        if len(top_cabs) == 0:
            top_cabs = all_cabs  # fall back if none qualify

        # Group by wall: cabinets on the same wall share a similar y-coord
        # (within 0.3 m).  Pick the largest group.
        top_cabs_sorted = sorted(top_cabs, key=lambda c: c.pos[1])
        best_group: list = []
        current_group: list = [top_cabs_sorted[0]]
        for cab in top_cabs_sorted[1:]:
            if abs(cab.pos[1] - current_group[0].pos[1]) < 0.4:
                current_group.append(cab)
            else:
                if len(current_group) > len(best_group):
                    best_group = current_group
                current_group = [cab]
        if len(current_group) > len(best_group):
            best_group = current_group

        # Sort selected group along x and take up to num_cabs.
        best_group.sort(key=lambda c: c.pos[0])
        self.cabs = best_group[: self.num_cabs]

        # Register each cabinet so its state is serialised in ep_meta.
        for i, cab in enumerate(self.cabs):
            self.register_fixture_ref(f"cab_{i}", dict(id=cab))

        # Place the robot in front of the middle cabinet.
        mid_idx = len(self.cabs) // 2
        self.init_robot_base_ref = self.cabs[mid_idx]

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        # Which cabinet (0-indexed) holds the object.
        ep_meta["target_cab_idx"] = int(self.target_cab_idx)
        ep_meta["num_cabs"] = len(self.cabs)
        cab_names = [c.name for c in self.cabs]
        ep_meta["cab_names"] = cab_names
        ep_meta["lang"] = (
            "Open the cabinets to find the object and pick it up."
        )
        return ep_meta

    def _setup_scene(self):
        # Ensure all cabinets start closed.
        for cab in self.cabs:
            cab.close_door(env=self)
        super()._setup_scene()

    # ------------------------------------------------------------------
    # Object configuration
    # ------------------------------------------------------------------

    def _get_obj_cfgs(self):
        if len(self.cabs) == 0:
            return []

        # Choose which cabinet gets the object.
        self.target_cab_idx = int(self.rng.integers(len(self.cabs)))
        target_cab = self.cabs[self.target_cab_idx]

        cfgs = [
            dict(
                name="obj",
                obj_groups="all",
                graspable=True,
                placement=dict(
                    fixture=target_cab,
                    size=(0.30, 0.20),
                    pos=(None, -1.0),  # front of shelf, closest to robot
                ),
            )
        ]
        return cfgs

    # ------------------------------------------------------------------
    # Success condition
    # ------------------------------------------------------------------

    def _check_success(self):
        """
        Episode succeeds as soon as the gripper makes contact with the object.
        """
        obj = self.objects.get("obj", None)
        if obj is None:
            return False
        gripper = self.robots[0].gripper
        return self._check_grasp(gripper, obj)
