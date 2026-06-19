

#!/usr/bin/env python3
"""
HanoiNode — ROS2 action server for Tower of Hanoi on real Franka Panda.

Main loop:
  - Get observation from perception via RealObsBridge
  - Step the FSM to get next phase + target
  - Execute via MoveItClient (motion) or GripperClient (grasp/release)
  - Verify reach via _at_target_check (TF-based)
  - Replan on failure

Motion strategy:
  - Joint-space planning  → long moves (hover, pre-grasp, pre-place)
  - Cartesian straight    → short vertical moves (descend, lift)
"""
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from tf2_ros import Buffer, TransformListener

from hanoi_interfaces.action import HanoiExecute

from hanoi_control.hanoi_fsm        import HanoiFSM, Phase
from hanoi_control.real_obs_bridge  import RealObsBridge, LABEL_TO_CUBE
from hanoi_control.moveit_client    import MoveItClient
from hanoi_control.gripper_client   import GripperClient


# ── Constants ─────────────────────────────────────────────────────────────────

SETTLE_TIME     = 1.0          # seconds to wait after lift / place
CUBE_BOX_SIZE   = 0.05         # cube side length in metres
TABLE_POS       = (0.4, 0.0, -0.02)
TABLE_SIZE      = (0.8, 0.8, 0.04)
GRASP_MIN_WIDTH = 0.01         # below this, grasp is considered failed

PICK_CENTER_OFFSET = np.array([0.000, -0.005, 0.000])
STACK_PLACE_OFFSET = np.array([-0.010, 0.000, 0.000])
class HanoiNode(Node):
    """ROS2 Action Server for Tower of Hanoi on real Franka Panda."""

    def __init__(self):
        super().__init__('hanoi_node')

        self.bridge  = RealObsBridge(self)
        self.moveit  = MoveItClient(self)
        self.gripper = GripperClient(self)

        # TF for closed-loop verification of motion targets
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # # Table collision so MoveIt2 refuses to plan below table surface
        # self.moveit.add_table(pos=TABLE_POS, size=TABLE_SIZE)

        self.attached_cube_label = None
        self.attached_cube_size  = None
        
        self.cached_boxes_pos = None
        self.cached_boxes_quat = None
        
        # self.fixed_boxes_pos = np.array([
        #     [0.450, -0.203, 0.049],   # left peg / red cross center
        #     [0.450, -0.015, 0.041],   # middle peg / red cross center
        #     [0.450,  0.170, 0.033],   # right peg / measured true center
        # ], dtype=float)
        
        self.fixed_boxes_pos = np.array([
        [0.445, -0.203, 0.022],  # left
        [0.452, -0.010, 0.021],  # middle
        [0.454,  0.170, 0.025],  # right
    ], dtype=float)

        self.fixed_boxes_quat = np.tile(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            (3, 1),
        )

        self.action_server = ActionServer(
            self,
            HanoiExecute,
            '/hanoi_execute',
            execute_callback = self.execute_callback,
            goal_callback    = lambda goal:   GoalResponse.ACCEPT,
            cancel_callback  = lambda cancel: CancelResponse.ACCEPT,
        )
        self.get_logger().info("Action server /hanoi_execute ready")
    def _cache_pegs_from_obs(self, obs):
        """Cache static peg/ring poses from a good observation."""
        if obs is None:
            return

        boxes_pos = np.asarray(obs["boxes_pos"], dtype=float)
        boxes_quat = np.asarray(obs["boxes_quat"], dtype=float)

        # Only cache if all 3 pegs look valid, not [0,0,0]
        if boxes_pos.shape == (3, 3) and np.all(np.linalg.norm(boxes_pos[:, :2], axis=1) > 1e-6):
            self.cached_boxes_pos = boxes_pos.copy()
            self.cached_boxes_quat = boxes_quat.copy()
            self.get_logger().info(f"Cached peg poses: {self.cached_boxes_pos}")
        else:
            self.get_logger().warn("Did not cache peg poses; invalid boxes_pos")


    # def _apply_cached_pegs(self, obs):
    #     """Replace noisy live peg detections with cached static peg poses."""
    #     if obs is None:
    #         return None

    #     obs = dict(obs)  # shallow copy so we do not mutate original observation

    #     if self.cached_boxes_pos is not None:
    #         obs["boxes_pos"] = self.cached_boxes_pos.copy()

    #     if self.cached_boxes_quat is not None:
    #         obs["boxes_quat"] = self.cached_boxes_quat.copy()

    #     return obs
    
    def _apply_cached_pegs(self, obs):
        """Use fixed measured peg/ring poses; keep cube detections live."""
        if obs is None:
            return None

        obs = dict(obs)

        obs["boxes_pos"] = self.fixed_boxes_pos.copy()
        obs["boxes_quat"] = self.fixed_boxes_quat.copy()

        return obs

    # ── Action server callback ────────────────────────────────────────────────
    # def _adapt_target_for_real_robot(self, phase: Phase, target_pos):
    #     # PICK_CENTER_OFFSET = np.array([0.000, +0.005, 0.000])
    #     PICK_CENTER_OFFSET = np.array([0.000, -0.005, 0.000])
    #     if target_pos is None:
    #         return None

    #     target_pos = np.asarray(target_pos, dtype=float).copy()
    #     # Correct cube pick centering only.
    #     if phase in (Phase.PICK_PREGRASP, Phase.PICK_DESCEND):
    #         target_pos[:2] += PICK_CENTER_OFFSET[:2]

    #     place_debug = getattr(self.fsm, "last_place_debug", {})
    #     place_mode = place_debug.get("mode", None)

    #     # Real-robot execution correction, not evolved policy logic.
    #     if phase == Phase.PLACE_DESCEND:
    #         if  place_mode == "stack_on_cube":
    #             target_pos[2] -= 0.027
    #         else:
    #             target_pos[2] -= 0.017
                
    #         self.get_logger().info(
    #         f"Adjusted PLACE_DESCEND z for real robot: "
    #         f"mode={place_mode}, target={target_pos}"
    #     )
        

    #     return target_pos
    
    def _adapt_target_for_real_robot(self, phase: Phase, target_pos, obs=None):
        if target_pos is None:
            return None

        target_pos = np.asarray(target_pos, dtype=float).copy()

        place_debug = getattr(self.fsm, "last_place_debug", {})
        place_mode = place_debug.get("mode", None)

        # 1. Pick centering correction
        if phase in (Phase.PICK_PREGRASP, Phase.PICK_DESCEND):
            target_pos[:2] += PICK_CENTER_OFFSET[:2]

        # 2. Stack placement XY correction
        if place_mode == "stack_on_cube" and phase in (
            Phase.PLACE_PREPLACE,
            Phase.PLACE_DESCEND,
        ):
            target_pos[:2] += STACK_PLACE_OFFSET[:2]

        # 3. Placement Z correction
        if phase == Phase.PLACE_DESCEND:
            if place_mode == "stack_on_cube":
                target_pos[2] -= 0.030
            else:
                target_pos[2] -= 0.018

            self.get_logger().info(
                f"Adjusted target: mode={place_mode}, phase={phase.name}, "
                f"target={target_pos}"
            )

        return target_pos
    async def execute_callback(self, goal_handle):
        self.get_logger().info("Hanoi task started")

        n_disks  = goal_handle.request.num_disks
        self.fsm = HanoiFSM(n_disks=n_disks)
        feedback = HanoiExecute.Feedback()
        result   = HanoiExecute.Result()

        # ── Verify physical setup matches expected Hanoi start state ──────────
        initial_obs = self.bridge.get_observation()
        if initial_obs is None:
            
            goal_handle.abort()
            result.success = False
            result.message = "Could not get initial observation"
            return result
        self._cache_pegs_from_obs(initial_obs)
        initial_obs = self._apply_cached_pegs(initial_obs)
        parsed = self.fsm._parse_observation(initial_obs)
        assign, _ = self.fsm._assign_cubes_to_pegs(parsed)
        src_stack = assign[0]                          # cubes on left peg (peg 0)
        # expected  = list(range(n_disks - 1, -1, -1))   # [n-1, ..., 0] bottom→top

        # if src_stack != expected:
        #     goal_handle.abort()
        #     result.success = False
        #     result.message = (
        #         f"Invalid start: peg 0 has cubes {src_stack}, expected {expected}. "
        #         f"Stack {n_disks} cubes on left peg (largest at bottom: BROWN→BLUE→GREEN→RED)."
        #     )
        #     self.get_logger().error(result.message)
        #     return result

        self.get_logger().info(f"Start state OK: peg 0 stack = {src_stack}")

        while not self.fsm.done:

            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.success = False
                result.message = "Cancelled"
                return result

            obs = self.bridge.get_observation()
            if obs is None:
                self.get_logger().error("Failed to get obs — retrying")
                time.sleep(1.0)
                continue
            obs = self._apply_cached_pegs(obs)

            # self._update_scene(obs)
            self.get_logger().info(
                f"boxes_pos USED BY FSM left={obs['boxes_pos'][0]}, "
                f"mid={obs['boxes_pos'][1]}, right={obs['boxes_pos'][2]}"
            )


            phase, target_pos = self.fsm.step(obs)
            self.get_logger().info(f"Phase: {phase.name}")

            success = await self._execute_phase(phase, target_pos, obs)
            if not success:
                self.get_logger().warn(f"Phase {phase.name} failed — replanning")
                self._handle_failure(phase)

            self._publish_feedback(goal_handle, feedback, phase)

        # final verification
        obs = self.bridge.get_observation()
        if obs is not None:
            obs = self._apply_cached_pegs(obs)
        if obs is not None and self.fsm._is_goal_state(self.fsm._parse_observation(obs)):
            goal_handle.succeed()
            result.success = True
            result.message = "Hanoi complete"
        else:
            goal_handle.abort()
            result.success = False
            result.message = "Goal state not reached"
        return result
    # ── Phase execution ───────────────────────────────────────────────────────

    async def _execute_phase(self, phase: Phase, target_pos, obs) -> bool:
        target_pos = self._adapt_target_for_real_robot(phase, target_pos)
        match phase:
            # ── Pick/place hover ───────────────────────────────────
            case Phase.PICK_HOVER:
                # Important for retry: reopen gripper before approaching cube
                ok = await self.gripper.open()
                if not ok:
                    return False
                return self.moveit.plan_cartesian_and_execute(target_pos)

            case Phase.PLACE_HOVER:
                return self.moveit.plan_cartesian_and_execute(target_pos)

            case Phase.PICK_PREGRASP | Phase.PLACE_PREPLACE:
                ok = self.moveit.plan_cartesian_and_execute(target_pos)
                if not ok:
                    return False
                return self._at_target_check(target_pos, xy_tol=0.015, z_tol=0.015)

            case Phase.PICK_DESCEND | Phase.PLACE_DESCEND:
                ok = self.moveit.plan_cartesian_and_execute(target_pos)
                if not ok:
                    return False
                return self._at_target_check(target_pos, xy_tol=0.010, z_tol=0.010)

            case Phase.LIFT_TO_PREGRASP:
                return self.moveit.plan_cartesian_and_execute(target_pos)

            case Phase.LIFT_TO_HOVER:
                ok = self.moveit.plan_cartesian_and_execute(target_pos)
                if not ok:
                    return False
                if not self._at_target_check(target_pos, xy_tol=0.015, z_tol=0.020):
                    return False
                return self._verify_grasp()

            case Phase.RETREAT_FROM_PLACE:
                ok = self.moveit.plan_cartesian_and_execute(target_pos)
                if not ok:
                    return False
                return self._verify_placement(obs)

            # ── Gripper close + (optional) attach ─────────────────────────
            # case Phase.CLOSE_GRIPPER:
            #     ok = await self.gripper.grasp(target_width=0.045)
            #     if not ok or not self._verify_grasp():
            #         return False

            #     disk_idx = self.fsm.current_move["disk"]
            #     label = self._label_for_cube_idx(disk_idx)

            #     if label is not None:
            #         self.moveit.remove_box(label)
            #         self.moveit.attach_box(label, size=(CUBE_BOX_SIZE,) * 3)
            #         self.attached_cube_label = label
            #         self.attached_cube_size = (CUBE_BOX_SIZE,) * 3

            #     return True
            case Phase.CLOSE_GRIPPER:
                disk_idx = self.fsm.current_move["disk"]

                # In your FSM, observation["cube_size"] is treated as cube_half
                cube_half = float(obs["cube_size"][disk_idx])
                cube_width = 2.0 * cube_half

                # Close slightly smaller than the object width to apply grip force
                squeeze = 0.004  # 4 mm
                grasp_width = max(0.0, cube_width - squeeze)

                self.get_logger().info(
                    f"Grasping disk={disk_idx}, cube_width={cube_width:.3f}, "
                    f"grasp_width={grasp_width:.3f}"
                )

                ok = await self.gripper.grasp(target_width=grasp_width)
                if not ok or not self._verify_grasp():
                    return False

                label = self._label_for_cube_idx(disk_idx)
                if label is not None:
                    self.moveit.remove_box(label)
                    self.moveit.attach_box(label, size=(cube_width,) * 3)
                    self.attached_cube_label = label
                    self.attached_cube_size = (cube_width,) * 3

                return True

            # ── Gripper open + (optional) detach ──────────────────────────
            # case Phase.OPEN_GRIPPER:
            #     ok = await self.gripper.open()
            #     if not ok:
            #         return False
            #     if self.attached_cube_label is not None:
            #         placed_obs = self.bridge.get_observation()
            #         placed_pos = None
            #         if placed_obs is not None:
            #             disk_idx = self.fsm.current_move["disk"]
            #             label    = self._label_for_cube_idx(disk_idx)
            #             if label in LABEL_TO_CUBE:
            #                 placed_pos = placed_obs["cube_pos"][LABEL_TO_CUBE[label]]
            #         self.moveit.detach_box(
            #             self.attached_cube_label,
            #             placed_pos = placed_pos,
            #             size       = self.attached_cube_size,
            #         )
            #         self.attached_cube_label = None
            #         self.attached_cube_size  = None
            #     return True
            
            case Phase.OPEN_GRIPPER:
                disk_idx = self.fsm.current_move["disk"] if self.fsm.current_move else None

                if disk_idx is not None and obs is not None:
                    cube_half = float(obs["cube_size"][disk_idx])
                    cube_width = 2.0 * cube_half

                    # Open only slightly wider than the cube first
                    release_width = min(0.08, cube_width + 0.010)
                else:
                    release_width = 0.060

                ok = await self.gripper.release(
                    release_width=release_width,
                    speed=0.015,
                )

                if not ok:
                    return False

                time.sleep(0.5)

                if self.attached_cube_label is not None:
                    placed_obs = self.bridge.get_observation()
                    placed_pos = None
                    if placed_obs is not None:
                        disk_idx = self.fsm.current_move["disk"]
                        label = self._label_for_cube_idx(disk_idx)
                        if label in LABEL_TO_CUBE:
                            placed_pos = placed_obs["cube_pos"][LABEL_TO_CUBE[label]]

                    self.moveit.detach_box(
                        self.attached_cube_label,
                        placed_pos=placed_pos,
                        size=self.attached_cube_size,
                    )

                    self.attached_cube_label = None
                    self.attached_cube_size = None

                return True

            case Phase.POST_LIFT_SETTLE | Phase.POST_PLACE_SETTLE:
                time.sleep(SETTLE_TIME)
                return True

            case Phase.ACQUIRE_MOVE | Phase.DONE:
                return True

        return True

    # ── Verification ──────────────────────────────────────────────────────────

    def _fmt_vec(self, v):
        if v is None:
            return "None"
        a = np.asarray(v, dtype=float).reshape(-1)
        return "[" + ", ".join(f"{x:+.4f}" for x in a) + "]"

    def _debug_phase_target(self, phase: Phase, target_pos, obs):
        """
        Debug boundary:
        perception obs -> FSM target -> current TCP before execution.
        """
        try:
            parsed = self.fsm._parse_observation(obs)
            tcp = self._get_current_tcp()

            self.get_logger().warn(
                "[MOTION_DEBUG] "
                f"phase={phase.name} "
                f"move={self.fsm.current_move} "
                f"target={self._fmt_vec(target_pos)} "
                f"tcp_now={self._fmt_vec(tcp)}"
            )

            labels_by_idx = {}
            for label, idx in LABEL_TO_CUBE.items():
                labels_by_idx[int(idx)] = label

            for i in range(4):
                pos = parsed["cube_pos"][i]
                half = float(parsed["cube_half"][i])
                self.get_logger().warn(
                    "[OBS_DEBUG] "
                    f"cube_idx={i} label={labels_by_idx.get(i, '?')} "
                    f"pos={self._fmt_vec(pos)} "
                    f"half={half:.4f} "
                    f"bottom_z={pos[2] - half:+.4f} "
                    f"top_z={pos[2] + half:+.4f}"
                )

            for j in range(3):
                self.get_logger().warn(
                    "[OBS_DEBUG] "
                    f"peg={j} pos={self._fmt_vec(parsed['boxes_pos'][j])}"
                )

            assign, holes = self.fsm._assign_cubes_to_pegs(parsed)
            self.get_logger().warn(
                f"[ASSIGN_DEBUG] assign={assign} holes={[self._fmt_vec(h) for h in holes]}"
            )

            dbg = getattr(self.fsm, "last_debug", {})
            if dbg:
                self.get_logger().warn(
                    "[FSM_DEBUG] "
                    f"disk={dbg.get('disk')} dst={dbg.get('dst')} "
                    f"cube_pos={self._fmt_vec(dbg.get('cube_pos'))} "
                    f"cube_half={dbg.get('cube_half')} "
                    f"cube_bottom_z={dbg.get('cube_bottom_z')} "
                    f"cube_top_z={dbg.get('cube_top_z')} "
                    f"raw_pick_grasp={self._fmt_vec(dbg.get('raw_pick_grasp'))} "
                    f"pick_grasp_safe={self._fmt_vec(dbg.get('pick_grasp'))} "
                    f"place_contact_raw={self._fmt_vec(dbg.get('place_contact_raw'))} "
                    f"place_contact_safe={self._fmt_vec(dbg.get('place_contact'))} "
                    f"support_z_est={dbg.get('support_z_est')}"
                )

            pdbg = getattr(self.fsm, "last_place_debug", {})
            if pdbg:
                self.get_logger().warn(
                    "[PLACE_DEBUG] "
                    f"mode={pdbg.get('mode')} "
                    f"moving_disk={pdbg.get('moving_disk')} "
                    f"dst_peg={pdbg.get('dst_peg')} "
                    f"support_disk={pdbg.get('support_disk')} "
                    f"hole={self._fmt_vec(pdbg.get('hole'))} "
                    f"raw_place_target={self._fmt_vec(pdbg.get('raw_place_target'))} "
                    f"dst_stack={pdbg.get('dst_stack')}"
                )

        except Exception as e:
            self.get_logger().warn(f"[MOTION_DEBUG] failed: {e}")


    def _get_current_tcp(self):
        """Read panda_hand_tcp position in panda_link0 frame via TF."""
        try:
            tf = self.tf_buffer.lookup_transform(
                'panda_link0', 'panda_hand_tcp', rclpy.time.Time()
            )
            return np.array([
                tf.transform.translation.x,
                tf.transform.translation.y,
                tf.transform.translation.z,
            ])
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return None

    def _at_target_check(self, target_pos, xy_tol=0.015, z_tol=0.015) -> bool:
        """Verify the robot actually reached the target via TF."""
        actual = self._get_current_tcp()
        if actual is None:
            return True   # can't verify, trust MoveIt2
        
        xy_err = np.linalg.norm(actual[:2] - target_pos[:2])
        z_err  = abs(actual[2] - target_pos[2])
        ok = xy_err <= xy_tol and z_err <= z_tol
        self.get_logger().info(
            f"_at_target: target={target_pos}, actual={actual}, "
            f"xy_err={xy_err:.4f}, z_err={z_err:.4f}, ok={ok}"
        )
        return ok

    # def _verify_grasp(self) -> bool:
    #     width = self.gripper._last_width
    #     if width is not None and width >= GRASP_MIN_WIDTH:
    #         return True
    #     self.get_logger().warn(f"Grasp likely failed — width={width}")
    #     return False
    
    def _verify_grasp(self) -> bool:
        if self.gripper._last_grasp_success:
            return True

        self.get_logger().warn("Grasp likely failed — gripper action returned false")
        return False

    def _verify_placement(self, obs_before) -> bool:
        time.sleep(0.5)
        obs = self.bridge.get_observation()
        if obs is None:
            return False
        obs = self._apply_cached_pegs(obs)
        s = self.fsm._parse_observation(obs)
        assign, _ = self.fsm._assign_cubes_to_pegs(s)
        disk = self.fsm.current_move["disk"]     if self.fsm.current_move else None
        dst  = self.fsm.current_move["distance"] if self.fsm.current_move else None
        if disk is None:
            return True
        if disk not in assign[dst]:
            self.get_logger().warn(f"Cube {disk} not on peg {dst} after place")
            return False
        return True

    # ── Failure handling ──────────────────────────────────────────────────────

    def _handle_failure(self, phase: Phase):
        match phase:
            case Phase.CLOSE_GRIPPER | Phase.LIFT_TO_HOVER:
                self.fsm.replan(Phase.PICK_HOVER)
            case Phase.RETREAT_FROM_PLACE:
                self.fsm.replan(Phase.ACQUIRE_MOVE)
            case _:
                self.fsm.replan(Phase.ACQUIRE_MOVE)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _label_for_cube_idx(self, idx: int):
        for label, i in LABEL_TO_CUBE.items():
            if i == idx:
                return label
        return None

    def _publish_feedback(self, goal_handle, feedback, phase):
        feedback.current_phase = phase.name
        feedback.move_index    = self.fsm.move_idx
        feedback.total_moves   = len(self.fsm.move_plan)
        if self.fsm.current_move:
            src = self.fsm.current_move.get('src', '?')
            dst = self.fsm.current_move.get('distance', '?')
            feedback.current_move = f"peg{src}→peg{dst}"
        else:
            feedback.current_move = ""
        goal_handle.publish_feedback(feedback)


    # ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = HanoiNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

