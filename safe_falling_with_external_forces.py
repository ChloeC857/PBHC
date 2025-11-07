"""
Detect external pushes during the play phase and trigger one of four directional falling recovery policies.

This script implements the following functions:

1. Load the normal walking policy.
2. Load four falling recovery policies for forward, backward, left, and right directions.
3. Monitor the robot base acceleration to detect pushes in all four directions in real time.
4. When a push exceeds the threshold, switch to the corresponding recovery policy.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play with fall recovery policy triggered by external force detection.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument(
    "--fall_forward_policy_path",
    type=str,
    default=None,
    help="Path to the forward fall recovery policy checkpoint directory.",
)
parser.add_argument(
    "--fall_backward_policy_path",
    type=str,
    default=None,
    help="Path to the backward fall recovery policy checkpoint directory.",
)
parser.add_argument(
    "--fall_left_policy_path",
    type=str,
    default=None,
    help="Path to the left fall recovery policy checkpoint directory.",
)
parser.add_argument(
    "--fall_right_policy_path",
    type=str,
    default=None,
    help="Path to the right fall recovery policy checkpoint directory.",
)
parser.add_argument(
    "--normal_policy_path",
    type=str,
    default=None,
    help="Path to the normal walking policy checkpoint directory.",
)
parser.add_argument(
    "--force_threshold",
    type=float,
    default=2.0,
    help="Threshold for force detection in any direction (in m/s^2). Default: 2.0",
)
parser.add_argument(
    "--detection_window",
    type=int,
    default=5,
    help="Number of steps to average acceleration for detection. Default: 5",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import torch

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401


class ExternalForceDetector:
    """
    Module for detecting external pushes.

    Detects pushes in four directions (forward, backward, left, right) by monitoring changes in the robot base linear velocity.

    Coordinates:
        x-axis: forward and backward (forward is positive, backward is negative)
        y-axis: left and right (left is positive, right is negative)
        z-axis: up and down
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        force_threshold: float = 2.0,
        detection_window: int = 5,
    ):
        """
        Initialize the external push detector.

        Args:
            num_envs: Number of environments.
            device: Computation device.
            force_threshold: Threshold for push detection (m/s²).
            detection_window: Window size for smoothing acceleration.
        """
        self.num_envs = num_envs
        self.device = device
        self.force_threshold = force_threshold
        self.detection_window = detection_window

        # Store history velocities for calculating acceleration
        self.prev_lin_vel = torch.zeros(num_envs, 3, device=device)
        self.acceleration_history = torch.zeros(num_envs, detection_window, 3, device=device)
        self.acceleration_index = 0
        self.dt = None

        # Detection state: detected direction for each environment  
        # 0: No push, 1: Forward, 2: Backward, 3: Left, 4: Right
        self.detected_direction = torch.zeros(num_envs, dtype=torch.long, device=device)

    def update(self, robot_lin_vel: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Update the detector and return the detected directions.

        Args:
            robot_lin_vel: Linear velocity of the robot base, shape (num_envs, 3).
            dt: Time step.

        Returns:
            Detected directions, shape (num_envs), values: 0 = none, 1 = forward, 2 = backward, 3 = left, 4 = right.
        """
        self.dt = dt

        # Calculate acceleration
        acceleration = (robot_lin_vel - self.prev_lin_vel) / dt

        # Update acceleration history
        self.acceleration_history[:, self.acceleration_index] = acceleration
        self.acceleration_index = (self.acceleration_index + 1) % self.detection_window

        # Calculate mean acceleration (for smoothing)
        avg_acceleration = self.acceleration_history.mean(dim=1)  # shape: (num_envs, 3)

        # Detect pushes in four directions
        # x-axis：forward (positive) and backward (negative)
        forward_acc = avg_acceleration[:, 0]  # positive x-axis (forward)
        backward_acc = -avg_acceleration[:, 0]  # negative x-axis (backward)
        # y-aixs：left (positive) and right (negative)
        left_acc = avg_acceleration[:, 1]  # positive y-axis (left)
        right_acc = -avg_acceleration[:, 1]  # negative y-axis (right)

        # Find the direction of maximum acceleration in every environment  
        # Create a tensor to store acceleration values for the four directions
        direction_accs = torch.stack(
            [
                torch.zeros(self.num_envs, device=self.device),  # 0: no push
                forward_acc,  # 1: forward
                backward_acc,  # 2: backward
                left_acc,  # 3: left
                right_acc,  # 4: right
            ],
            dim=1,
        )  # shape: (num_envs, 5)

        # Find the direction of maximum acceleration
        max_acc, max_direction = direction_accs.max(dim=1)

        # WHen maximum acceleration exceeds the threshold, push detected
        # Otherwise, the direction is set to 0 (no push)
        self.detected_direction = torch.where(
            max_acc > self.force_threshold, max_direction, torch.zeros_like(max_direction)
        )

        # Update history velocity
        self.prev_lin_vel = robot_lin_vel.clone()

        return self.detected_direction

    def reset(self, env_ids: torch.Tensor | None = None):
        """
        Reset the detector state.

        Args:
            env_ids: IDs of the environments to reset. If None, reset all environments.
        """

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self.prev_lin_vel[env_ids] = 0.0
        self.acceleration_history[env_ids] = 0.0
        self.detected_direction[env_ids] = 0


def main():
    """ Main function: run the play script with four-direction falling recovery."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg = hydra_task_config(
        args_cli.task, "rsl_rl_cfg_entry_point"
    )
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # Load walking policy path
    if args_cli.normal_policy_path:
        normal_policy_path = args_cli.normal_policy_path
    else:
        # Use path from agent_cfg as default
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        normal_policy_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # Load four-direction falling recovery policies paths
    fall_policy_paths = {}
    fall_policy_names = {
        1: ("forward", args_cli.fall_forward_policy_path),
        2: ("backward", args_cli.fall_backward_policy_path),
        3: ("left", args_cli.fall_left_policy_path),
        4: ("right", args_cli.fall_right_policy_path),
    }

    for direction_id, (direction_name, policy_path) in fall_policy_names.items():
        if policy_path is None:
            raise ValueError(
                f"Please provide recovery policy path for direction {direction_name}!"
                f"Use --fall_{direction_name}_policy_path, specify the checkpoint directory for falling policy in the {direction_name} direction."
            )
        fall_policy_paths[direction_id] = get_checkpoint_path(
            policy_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )

    print(f"[INFO]: Load walking policy from: {normal_policy_path}")
    for direction_id, (direction_name, _) in fall_policy_names.items():
        print(f"[INFO]: Load recovery policy in {direction_name} direction from: {fall_policy_paths[direction_id]}")

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Wrap environment
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    # Load walking policy
    normal_ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    normal_ppo_runner.load(normal_policy_path)
    normal_policy = normal_ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # Load four-direction falling recovery policies
    fall_policies = {}
    for direction_id, policy_path in fall_policy_paths.items():
        fall_ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        fall_ppo_runner.load(policy_path)
        fall_policies[direction_id] = fall_ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # Initial external force detector
    force_detector = ExternalForceDetector(
        num_envs=env_cfg.scene.num_envs,
        device=env.unwrapped.device,
        force_threshold=args_cli.force_threshold,
        detection_window=args_cli.detection_window,
    )

    # Track the policy used in every environment
    # 0: walking policy, 1: forward, 2: backward, 3: left, 4: right
    current_policy_direction = torch.zeros(env_cfg.scene.num_envs, dtype=torch.long, device=env.unwrapped.device)

    # Get robot articulation
    robot = env.unwrapped.scene["robot"]

    # Reset environment
    obs, _ = env.get_observations()
    timestep = 0

    print(f"[INFO]: Simulation starts, force_threshold: {args_cli.force_threshold} m/s^2")

    # Simulation loop
    while simulation_app.is_running():
        with torch.inference_mode():
            # Get the linear velocity of the robot base root node is usually the first one)  
            # Use body_lin_vel_w to get velocities of all bodies, then take the first one (index 0) as root node
            robot_lin_vel = robot.data.body_lin_vel_w[:, 0]  # shape: (num_envs, 3)
            dt = env.unwrapped.step_dt

            # Update external force detector
            detected_direction = force_detector.update(robot_lin_vel, dt)

            # If push detected in any specific direction, switch to corresponding recovery policy
            # Switch only when push detected and in walking state

            switch_mask = (detected_direction > 0) & (current_policy_direction == 0)
            current_policy_direction[switch_mask] = detected_direction[switch_mask]

            # If a falling recovery policy is already in use, check whether to switch back to the walking policy  
            # More complex switching logic can be added here if needed  
            # For example, detect whether the robot has regained stability (e.g., reduced velocity variation)  
            # For now, keep using the falling recovery policy until a manual or environment reset occurs  

            # Execute the policy  
            # Select the corresponding policy based on the current direction used in every environment  
            # Calculate actions for all environments (use the walking policy as default)

            actions = normal_policy(obs)

            # Calculate actions for every direction
            for direction_id in [1, 2, 3, 4]:  # 1:forward, 2:backward, 3:left, 4:right
                direction_mask = current_policy_direction == direction_id
                if direction_mask.any():
                    direction_actions = fall_policies[direction_id](obs[direction_mask])
                    actions[direction_mask] = direction_actions

            # env step
            obs, _, terminated, infos = env.step(actions)

            # Reset environment
            if terminated.any():
                reset_env_ids = torch.where(terminated)[0]
                # Reset external force detector
                force_detector.reset(reset_env_ids)
                # Reset policy state
                current_policy_direction[reset_env_ids] = 0
                print(f"[INFO] TImestep {timestep}: Environment {reset_env_ids.tolist()} has been reset.")

            # Print detection info (for the first environment only)
            direction_names = {0: "No push", 1: "Forward", 2: "Backward", 3: "Left", 4: "Right"}
            detected_dir = int(detected_direction[0].item())
            if detected_dir > 0:
                direction_name = direction_names[detected_dir]
                print(f"[INFO] Timestep {timestep}: Detects push in {direction_name} direction! Switch to recovery policy in {direction_name} direction.")

            timestep += 1

            if args_cli.video:
                if timestep >= args_cli.video_length:
                    break

    # Close environment
    env.close()
    print("[INFO]: Simulation ends.")


if __name__ == "__main__":
    # Run main function
    main()
    # Close simulation app
    simulation_app.close()

