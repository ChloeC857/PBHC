"""脚本：在play阶段检测外部推力并触发四个方向的摔倒恢复policy。

该脚本实现了以下功能：
1. 加载正常行走的policy
2. 加载前、后、左、右四个方向的摔倒恢复policy
3. 实时检测四个方向的推力（通过监控机器人基座的加速度）
4. 当检测到某个方向的推力超过阈值时，自动切换到对应方向的摔倒恢复policy
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
    """检测外部推力的模块。

    通过监控机器人基座的线性速度变化来检测四个方向（前、后、左、右）的推力。
    坐标系假设：
    - x轴：前后方向（前为正，后为负）
    - y轴：左右方向（左为正，右为负）
    - z轴：上下方向
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        force_threshold: float = 2.0,
        detection_window: int = 5,
    ):
        """初始化外部力检测器。

        Args:
            num_envs: 环境数量
            device: 计算设备
            force_threshold: 检测推力的阈值（m/s^2）
            detection_window: 用于平滑加速度的窗口大小
        """
        self.num_envs = num_envs
        self.device = device
        self.force_threshold = force_threshold
        self.detection_window = detection_window

        # 存储历史速度用于计算加速度
        self.prev_lin_vel = torch.zeros(num_envs, 3, device=device)
        self.acceleration_history = torch.zeros(num_envs, detection_window, 3, device=device)
        self.acceleration_index = 0
        self.dt = None

        # 检测状态：每个环境检测到的方向
        # 0: 无推力, 1: 前, 2: 后, 3: 左, 4: 右
        self.detected_direction = torch.zeros(num_envs, dtype=torch.long, device=device)

    def update(self, robot_lin_vel: torch.Tensor, dt: float) -> torch.Tensor:
        """更新检测器并返回检测到的方向。

        Args:
            robot_lin_vel: 机器人基座的线性速度，shape (num_envs, 3)
            dt: 时间步长

        Returns:
            检测到的方向，shape (num_envs,)，值为：0=无, 1=前, 2=后, 3=左, 4=右
        """
        self.dt = dt

        # 计算加速度（速度变化率）
        acceleration = (robot_lin_vel - self.prev_lin_vel) / dt

        # 更新历史记录
        self.acceleration_history[:, self.acceleration_index] = acceleration
        self.acceleration_index = (self.acceleration_index + 1) % self.detection_window

        # 计算平均加速度（用于平滑）
        avg_acceleration = self.acceleration_history.mean(dim=1)  # shape: (num_envs, 3)

        # 检测四个方向的推力
        # x方向：前（正）和后（负）
        forward_acc = avg_acceleration[:, 0]  # x正方向（前）
        backward_acc = -avg_acceleration[:, 0]  # x负方向（后）
        # y方向：左（正）和右（负）
        left_acc = avg_acceleration[:, 1]  # y正方向（左）
        right_acc = -avg_acceleration[:, 1]  # y负方向（右）

        # 找到每个环境的最大加速度方向
        # 创建一个张量存储四个方向的加速度值
        direction_accs = torch.stack(
            [
                torch.zeros(self.num_envs, device=self.device),  # 0: 无推力
                forward_acc,  # 1: 前
                backward_acc,  # 2: 后
                left_acc,  # 3: 左
                right_acc,  # 4: 右
            ],
            dim=1,
        )  # shape: (num_envs, 5)

        # 找到最大加速度的方向
        max_acc, max_direction = direction_accs.max(dim=1)

        # 只有当最大加速度超过阈值时才认为检测到推力
        # 否则方向为0（无推力）
        self.detected_direction = torch.where(
            max_acc > self.force_threshold, max_direction, torch.zeros_like(max_direction)
        )

        # 更新历史速度
        self.prev_lin_vel = robot_lin_vel.clone()

        return self.detected_direction

    def reset(self, env_ids: torch.Tensor | None = None):
        """重置检测器状态。

        Args:
            env_ids: 要重置的环境ID，如果为None则重置所有环境
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self.prev_lin_vel[env_ids] = 0.0
        self.acceleration_history[env_ids] = 0.0
        self.detected_direction[env_ids] = 0


def main():
    """主函数：运行带四个方向摔倒恢复的play脚本。"""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg = hydra_task_config(
        args_cli.task, "rsl_rl_cfg_entry_point"
    )
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # 加载正常policy的路径
    if args_cli.normal_policy_path:
        normal_policy_path = args_cli.normal_policy_path
    else:
        # 默认使用agent_cfg中的路径
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        normal_policy_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # 加载四个方向的摔倒恢复policy路径
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
                f"必须提供{direction_name}方向摔倒policy的路径！"
                f"使用 --fall_{direction_name}_policy_path 参数指定{direction_name}方向摔倒policy的checkpoint目录。"
            )
        fall_policy_paths[direction_id] = get_checkpoint_path(
            policy_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )

    print(f"[INFO]: 加载正常行走policy从: {normal_policy_path}")
    for direction_id, (direction_name, _) in fall_policy_names.items():
        print(f"[INFO]: 加载{direction_name}方向摔倒恢复policy从: {fall_policy_paths[direction_id]}")

    # 创建环境
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # 包装环境
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    # 加载正常行走policy
    normal_ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    normal_ppo_runner.load(normal_policy_path)
    normal_policy = normal_ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # 加载四个方向的摔倒恢复policy
    fall_policies = {}
    for direction_id, policy_path in fall_policy_paths.items():
        fall_ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        fall_ppo_runner.load(policy_path)
        fall_policies[direction_id] = fall_ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # 初始化外部力检测器
    force_detector = ExternalForceDetector(
        num_envs=env_cfg.scene.num_envs,
        device=env.unwrapped.device,
        force_threshold=args_cli.force_threshold,
        detection_window=args_cli.detection_window,
    )

    # 跟踪每个环境当前使用的policy方向
    # 0: 正常policy, 1: 前, 2: 后, 3: 左, 4: 右
    current_policy_direction = torch.zeros(env_cfg.scene.num_envs, dtype=torch.long, device=env.unwrapped.device)

    # 获取机器人articulation
    robot = env.unwrapped.scene["robot"]

    # 重置环境
    obs, _ = env.get_observations()
    timestep = 0

    print(f"[INFO]: 开始仿真，推力检测阈值: {args_cli.force_threshold} m/s^2")

    # 仿真循环
    while simulation_app.is_running():
        with torch.inference_mode():
            # 获取机器人基座的线性速度（根节点通常是第一个body）
            # 使用body_lin_vel_w获取所有body的速度，然后取第一个（索引0）作为根节点
            robot_lin_vel = robot.data.body_lin_vel_w[:, 0]  # shape: (num_envs, 3)
            dt = env.unwrapped.step_dt

            # 更新外部力检测器
            detected_direction = force_detector.update(robot_lin_vel, dt)

            # 如果检测到某个方向的推力，切换到对应方向的摔倒恢复policy
            # 只有当检测到方向且当前不是摔倒恢复状态时，才切换
            switch_mask = (detected_direction > 0) & (current_policy_direction == 0)
            current_policy_direction[switch_mask] = detected_direction[switch_mask]

            # 如果已经使用摔倒恢复policy，可以检测是否应该切换回正常policy
            # 这里可以根据需要实现更复杂的切换逻辑
            # 例如：检测机器人是否已经恢复稳定（速度变化率变小）
            # 暂时保持使用摔倒恢复policy，直到手动重置或环境重置

            # 执行policy
            # 需要根据每个环境当前使用的policy方向来选择对应的policy
            # 先计算所有环境的actions（使用normal policy作为默认）
            actions = normal_policy(obs)

            # 为每个方向分别计算actions
            for direction_id in [1, 2, 3, 4]:  # 1:前, 2:后, 3:左, 4:右
                direction_mask = current_policy_direction == direction_id
                if direction_mask.any():
                    direction_actions = fall_policies[direction_id](obs[direction_mask])
                    actions[direction_mask] = direction_actions

            # 环境步进
            obs, _, terminated, infos = env.step(actions)

            # 处理环境重置
            if terminated.any():
                reset_env_ids = torch.where(terminated)[0]
                # 重置检测器状态
                force_detector.reset(reset_env_ids)
                # 重置policy使用状态
                current_policy_direction[reset_env_ids] = 0
                print(f"[INFO] 时间步 {timestep}: 环境 {reset_env_ids.tolist()} 已重置")

            # 打印检测信息（仅第一个环境）
            direction_names = {0: "正常", 1: "前", 2: "后", 3: "左", 4: "右"}
            detected_dir = int(detected_direction[0].item())
            if detected_dir > 0:
                direction_name = direction_names[detected_dir]
                print(f"[INFO] 时间步 {timestep}: 检测到{direction_name}方向推力！切换到{direction_name}方向摔倒恢复policy。")

            timestep += 1

            if args_cli.video:
                if timestep >= args_cli.video_length:
                    break

    # 关闭环境
    env.close()
    print("[INFO]: 仿真结束")


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭仿真应用
    simulation_app.close()

