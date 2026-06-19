#!/usr/bin/env python3
import numpy as np
from enum import Enum, auto
from typing import Dict, Tuple


class Phase(Enum):
    ACQUIRE_MOVE = auto()
    PICK_HOVER = auto()
    PICK_PREGRASP = auto()
    PICK_DESCEND = auto()
    CLOSE_GRIPPER = auto()
    LIFT_TO_PREGRASP = auto()
    LIFT_TO_HOVER = auto()
    POST_LIFT_SETTLE = auto()
    PLACE_HOVER = auto()
    PLACE_PREPLACE = auto()
    PLACE_DESCEND = auto()
    OPEN_GRIPPER = auto()
    POST_PLACE_SETTLE = auto()
    RETREAT_FROM_PLACE = auto()
    DONE = auto()


class HanoiFSM:
    """
    Adapted FrankaPolicy for real robot.
    Pure Python — no ROS2 dependencies.

    Input: obs dict from RealObsBridge
    Output: (Phase, target_pos) where target_pos is numpy xyz
            HanoiNode converts to PoseStamped and executes via MoveIt2.
    """

    def __init__(self, n_disks: int = 4):
        self.n_disks = int(n_disks)

        self.safe_clearance = 0.120
        self.pick_hover_clearance = 0.100
        self.pick_pregrasp_clearance = 0.090
        self.place_hover_clearance = 0.100
        self.place_preplace_clearance = 0.045

        self.grasp_depth_offset = 0.045
        self.pick_grasp_z_safety = 0.000
        self.place_contact_z_safety = -0.004
        # self.place_contact_z_safety = 0.0000
        

        self.size_margin = 0.003
        self.peg_assign_xy = 0.075

        # Generic real-robot z safety.
        # These are absolute lower bounds in panda_link0 frame.
        self.min_hover_z = 0.140
        self.min_pregrasp_z = 0.110

        # Minimum extra height above each cube half-size.
        # This is generic for red/green/blue/brown.
        self.min_cube_clearance = 0.005

        self.last_debug = {}
        self.last_place_debug = {}

        # Hanoi plan: left -> right.
        # peg 0 = left, peg 1 = middle, peg 2 = right
        self.move_plan = self._generate_hanoi_moves(
            self.n_disks, src=0, aux=1, dst=2
        )
        self.move_idx = 0
        self.phase = Phase.ACQUIRE_MOVE
        self.current_move = None
        self.done = False
        self.support_z_est = None

    def step(self, observation: Dict) -> Tuple:
        if self.done:
            return Phase.DONE, None

        parsed_obs = self._parse_observation(observation)
        self._update_support_estimate(parsed_obs)

        if self.move_idx >= len(self.move_plan):
            if self._is_goal_state(parsed_obs):
                self.done = True
                return Phase.DONE, None

            hole_center = self._hole_center(
                parsed_obs["boxes_pos"][2], parsed_obs["boxes_quat"][2]
            )
            hover = hole_center + np.array([0.0, 0.0, self.safe_clearance])
            hover[2] = max(float(hover[2]), self.min_hover_z)
            return Phase.ACQUIRE_MOVE, hover

        return self._step_fsm(parsed_obs)

    def replan(self, phase: Phase):
        self.phase = phase

    def _step_fsm(self, parsed_obs) -> Tuple:
        if self.phase == Phase.ACQUIRE_MOVE:
            src, distance = self.move_plan[self.move_idx]
            disk = self._top_disk_on_peg(parsed_obs, src)
            if disk is None:
                return Phase.ACQUIRE_MOVE, None

            self.current_move = {
                "src": src,
                "distance": distance,
                "disk": disk,
            }
            self.phase = Phase.PICK_HOVER

        disk = self.current_move["disk"]
        distance = self.current_move["distance"]

        cube_pos = parsed_obs["cube_pos"][disk].copy()
        cube_half = float(parsed_obs["cube_half"][disk])

        # Generic z repair:
        # Perception may return a visible mask/depth point near table height.
        # The center of any cube cannot be below cube_half + clearance.
        min_cube_center_z = cube_half + self.min_cube_clearance
        cube_pos[2] = max(float(cube_pos[2]), min_cube_center_z)

        cube_top_z = float(cube_pos[2] + cube_half)
        cube_bottom_z = float(cube_pos[2] - cube_half)

        raw_pick_grasp = np.array(
            [
                cube_pos[0],
                cube_pos[1],
                cube_pos[2] - self.grasp_depth_offset,
            ],
            dtype=float,
        )

        pick_grasp = raw_pick_grasp + np.array(
            [0.0, 0.0, self.pick_grasp_z_safety],
            dtype=float,
        )

        pick_hover = np.array(
            [
                cube_pos[0],
                cube_pos[1],
                cube_top_z + self.pick_hover_clearance,
            ],
            dtype=float,
        )

        pick_pregrasp = np.array(
            [
                cube_pos[0],
                cube_pos[1],
                cube_top_z + self.pick_pregrasp_clearance,
            ],
            dtype=float,
        )

        place_contact_raw = self._compute_place_target(parsed_obs, disk, distance)
        place_contact = place_contact_raw + np.array(
            [0.0, 0.0, self.place_contact_z_safety],
            dtype=float,
        )

        # Final real-robot z clamps.
        # These are generic and apply to all cubes.
        pick_hover[2] = max(float(pick_hover[2]), self.min_hover_z)
        pick_pregrasp[2] = max(float(pick_pregrasp[2]), self.min_pregrasp_z)
        pick_grasp[2] = max(float(pick_grasp[2]), cube_half + self.min_cube_clearance)
        place_contact[2] = max(
            float(place_contact[2]),
            cube_half + self.min_cube_clearance,
        )

        place_preplace = place_contact + np.array(
            [0.0, 0.0, self.place_preplace_clearance],
            dtype=float,
        )
        place_hover = place_contact + np.array(
            [0.0, 0.0, self.place_hover_clearance],
            dtype=float,
        )

        place_preplace[2] = max(float(place_preplace[2]), self.min_pregrasp_z)
        place_hover[2] = max(float(place_hover[2]), self.min_hover_z)

        assign, holes = self._assign_cubes_to_pegs(parsed_obs)
        self.last_debug = {
            "move_idx": int(self.move_idx),
            "phase_before_return": self.phase.name,
            "current_move": dict(self.current_move) if self.current_move else None,
            "disk": int(disk),
            "dst": int(distance),
            "cube_pos": cube_pos.copy(),
            "cube_half": float(cube_half),
            "cube_top_z": cube_top_z,
            "cube_bottom_z": cube_bottom_z,
            "min_cube_center_z": float(min_cube_center_z),
            "raw_pick_grasp": raw_pick_grasp.copy(),
            "pick_grasp": pick_grasp.copy(),
            "pick_pregrasp": pick_pregrasp.copy(),
            "pick_hover": pick_hover.copy(),
            "place_contact_raw": place_contact_raw.copy(),
            "place_contact": place_contact.copy(),
            "place_preplace": place_preplace.copy(),
            "place_hover": place_hover.copy(),
            "holes": [h.copy() for h in holes],
            "assign": {k: list(v) for k, v in assign.items()},
            "support_z_est": self.support_z_est,
        }

        # ── Pick sequence ─────────────────────────────────────────────────
        if self.phase == Phase.PICK_HOVER:
            self.phase = Phase.PICK_PREGRASP
            return Phase.PICK_HOVER, pick_hover

        if self.phase == Phase.PICK_PREGRASP:
            self.phase = Phase.PICK_DESCEND
            return Phase.PICK_PREGRASP, pick_pregrasp

        if self.phase == Phase.PICK_DESCEND:
            self.phase = Phase.CLOSE_GRIPPER
            return Phase.PICK_DESCEND, pick_grasp

        if self.phase == Phase.CLOSE_GRIPPER:
            self.phase = Phase.LIFT_TO_PREGRASP
            return Phase.CLOSE_GRIPPER, None

        if self.phase == Phase.LIFT_TO_PREGRASP:
            self.phase = Phase.LIFT_TO_HOVER
            return Phase.LIFT_TO_PREGRASP, pick_pregrasp

        if self.phase == Phase.LIFT_TO_HOVER:
            self.phase = Phase.POST_LIFT_SETTLE
            return Phase.LIFT_TO_HOVER, pick_hover

        if self.phase == Phase.POST_LIFT_SETTLE:
            self.phase = Phase.PLACE_HOVER
            return Phase.POST_LIFT_SETTLE, None

        # ── Place sequence ────────────────────────────────────────────────
        if self.phase == Phase.PLACE_HOVER:
            self.phase = Phase.PLACE_PREPLACE
            return Phase.PLACE_HOVER, place_hover

        if self.phase == Phase.PLACE_PREPLACE:
            self.phase = Phase.PLACE_DESCEND
            return Phase.PLACE_PREPLACE, place_preplace

        if self.phase == Phase.PLACE_DESCEND:
            self.phase = Phase.OPEN_GRIPPER
            return Phase.PLACE_DESCEND, place_contact

        if self.phase == Phase.OPEN_GRIPPER:
            self.phase = Phase.POST_PLACE_SETTLE
            return Phase.OPEN_GRIPPER, None

        if self.phase == Phase.POST_PLACE_SETTLE:
            self.phase = Phase.RETREAT_FROM_PLACE
            return Phase.POST_PLACE_SETTLE, None

        if self.phase == Phase.RETREAT_FROM_PLACE:
            self.move_idx += 1
            self.current_move = None
            self.phase = Phase.ACQUIRE_MOVE
            return Phase.RETREAT_FROM_PLACE, place_hover

        return Phase.ACQUIRE_MOVE, None

    def _parse_observation(self, observation: Dict) -> Dict:
        return {
            "cube_pos": np.asarray(
                observation["cube_pos"], dtype=float
            ).reshape(4, 3),
            "cube_quat": np.asarray(
                observation["cube_quat"], dtype=float
            ).reshape(4, 4),
            "cube_half": np.asarray(
                observation["cube_size"], dtype=float
            ).reshape(4),
            "boxes_pos": np.asarray(
                observation["boxes_pos"], dtype=float
            ).reshape(3, 3),
            "boxes_quat": np.asarray(
                observation["boxes_quat"], dtype=float
            ).reshape(3, 4),
        }

    def _hole_center(self, box_pose, box_quat):
        return np.asarray(box_pose, dtype=float)

    def _peg_hole_centers(self, parsed_obs):
        return [
            self._hole_center(
                parsed_obs["boxes_pos"][i],
                parsed_obs["boxes_quat"][i],
            )
            for i in range(3)
        ]

    def _assign_cubes_to_pegs(self, parsed_obs):
        holes = self._peg_hole_centers(parsed_obs)
        out = {0: [], 1: [], 2: [], "unknown": []}

        for i in range(4):
            p = parsed_obs["cube_pos"][i]

            # Missing cubes often come as [0, 0, 0].
            # Do not assign them to any peg.
            if not np.all(np.isfinite(p)) or np.linalg.norm(p[:2]) < 1e-6:
                out["unknown"].append(i)
                continue

            dxy = [np.linalg.norm(p[:2] - h[:2]) for h in holes]
            peg = int(np.argmin(dxy))

            if dxy[peg] < self.peg_assign_xy:
                out[peg].append(i)
            else:
                out["unknown"].append(i)

        for peg in [0, 1, 2]:
            out[peg] = sorted(
                out[peg],
                key=lambda idx: parsed_obs["cube_pos"][idx][2],
            )

        return out, holes

    def _top_disk_on_peg(self, parsed_obs, peg_idx):
        assign, _ = self._assign_cubes_to_pegs(parsed_obs)

        if len(assign[peg_idx]) == 0:
            return None

        return assign[peg_idx][-1]

    # def _compute_place_target(self, parsed_obs, moving_disk, dst_peg):
    #     assign, holes = self._assign_cubes_to_pegs(parsed_obs)
    #     hole = holes[dst_peg]

    #     moving_half = float(parsed_obs["cube_half"][moving_disk]) + self.size_margin
    #     dst_stack = [i for i in assign[dst_peg] if i != moving_disk]

    #     if len(dst_stack) == 0:
    #         support = self.support_z_est
    #         if support is None:
    #             support = 0.0

    #         z = support + moving_half + 0.002
    #         mode = "empty_peg"
    #         support_disk = None
    #     else:
    #         top = dst_stack[-1]
    #         top_half = float(parsed_obs["cube_half"][top]) + self.size_margin

    #         top_pos = parsed_obs["cube_pos"][top].copy()
    #         top_pos[2] = max(
    #             float(top_pos[2]),
    #             float(parsed_obs["cube_half"][top]) + self.min_cube_clearance,
    #         )

    #         # z = top_pos[2] + top_half + moving_half + 0.0015
    #         # mode = "stack_on_cube"
    #         # support_disk = int(top)
    #         stack_place_clearance = 0.006  # start with 6 mm

    #         z = top_pos[2] + top_half + moving_half + stack_place_clearance
    #         mode = "stack_on_cube"
    #         support_disk = int(top)

    #     target = np.array([hole[0], hole[1], z], dtype=float)

    #     self.last_place_debug = {
    #         "mode": mode,
    #         "moving_disk": int(moving_disk),
    #         "dst_peg": int(dst_peg),
    #         "support_disk": support_disk,
    #         "hole": hole.copy(),
    #         "moving_half_with_margin": float(moving_half),
    #         "raw_place_target": target.copy(),
    #         "dst_stack": list(dst_stack),
    #     }

    #     return target
    def _compute_place_target(self, parsed_obs, moving_disk, dst_peg):
        assign, holes = self._assign_cubes_to_pegs(parsed_obs)
        hole = holes[dst_peg]

        moving_half = float(parsed_obs["cube_half"][moving_disk]) + self.size_margin
        dst_stack = [i for i in assign[dst_peg] if i != moving_disk]

        if len(dst_stack) == 0:
            support = self.support_z_est
            if support is None:
                support = float(
                    np.min(parsed_obs["cube_pos"][:, 2] - parsed_obs["cube_half"])
                )

            z = support + moving_half + 0.002
            mode = "empty_peg"
            support_disk = None

        else:
            top = dst_stack[-1]
            z = (
                parsed_obs["cube_pos"][top][2]
                + (float(parsed_obs["cube_half"][top]) + self.size_margin)
                + moving_half
                + 0.0015
            )
            mode = "stack_on_cube"
            support_disk = int(top)

        target = np.array([hole[0], hole[1], z], dtype=float)

        self.last_place_debug = {
            "mode": mode,
            "moving_disk": int(moving_disk),
            "dst_peg": int(dst_peg),
            "support_disk": support_disk,
            "hole": hole.copy(),
            "moving_half_with_margin": float(moving_half),
            "raw_place_target": target.copy(),
            "dst_stack": list(dst_stack),
        }

        return target

    def _update_support_estimate(self, parsed_obs):
        cube_pos = parsed_obs["cube_pos"]
        cube_half = parsed_obs["cube_half"]

        valid_bottoms = []

        for i in range(4):
            p = cube_pos[i]

            if not np.all(np.isfinite(p)):
                continue

            # Ignore missing cubes encoded as zero.
            if np.linalg.norm(p[:2]) < 1e-6:
                continue

            bottom = float(p[2] - cube_half[i])
            valid_bottoms.append(bottom)

        if valid_bottoms:
            # Table/support should not become negative because of bad mask z.
            self.support_z_est = max(0.0, float(np.min(valid_bottoms)))

    def _is_goal_state(self, parsed_obs) -> bool:
        assign, _ = self._assign_cubes_to_pegs(parsed_obs)

        expected = list(range(self.n_disks - 1, -1, -1))
        return assign[2] == expected

    def _generate_hanoi_moves(self, n, src, aux, dst):
        moves = []

        def solve(k, s, a, d):
            if k <= 0:
                return
            solve(k - 1, s, d, a)
            moves.append((s, d))
            solve(k - 1, a, s, d)

        solve(int(n), int(src), int(aux), int(dst))
        return moves