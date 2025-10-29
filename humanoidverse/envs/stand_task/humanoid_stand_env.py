import os
import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces

class HumanoidStandEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self,
                 xml_path="description/robots/g1/g1_29dof_rev_1_0.xml",
                 sim_dt=1/240,           # 物理步长
                 control_dt=1/30,        # 控制周期（每 N 个物理步执行一次控制）
                 target_height=0.85,
                 tilt_limit_deg=60.0,
                 action_scale=0.3,
                 seed=None,
                 render_mode=None):
        super().__init__()
        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = sim_dt
        self.render_mode = render_mode
        self.viewer = None

        # 基本尺寸
        self.nq = self.model.nq           # 包含 7 个 free-joint + dof
        self.nv = self.model.nv
        self.nu = self.model.nu           # actuator 数
        self.dof = self.nq - 7            # 仅关节自由度数

        # 动作：关节增量目标（期望姿势的微调，位置式控制假设）
        self.action_scale = action_scale
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.dof,), dtype=np.float32)

        # 观测：关节角(qpos[7:])、关节速度(qvel[6:])、root 高度、root 朝向（重力方向投影）
        obs_dim = self.dof + (self.nv - 6) + 1 + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # 参数
        self.control_dt = control_dt
        self.sim_steps_per_ctrl = max(1, int(np.round(control_dt / sim_dt)))
        self.target_height = target_height
        self.tilt_limit_rad = np.deg2rad(tilt_limit_deg)

        self.rw_w_upright = 3.0
        self.rw_w_height  = 2.0
        self.rw_w_vel     = 0.1
        self.rw_w_action  = 0.05
        self.rw_w_feet    = 0.5   # 可选：鼓励脚部稳定（需根据 geom/actuator 调整）

        self.max_steps = int(10.0 / control_dt)  # 10s
        self.step_count = 0

        self.rng = np.random.default_rng(seed)

        # 期望姿态（站立）
        self.qpos_ref = np.zeros(self.dof, dtype=np.float32)

    # ---------- 工具函数 ----------
    def _get_root_height(self):
        return float(self.data.qpos[2])  # z

    def _get_root_up(self):
        # 从 qpos[3:7]（wxyz or xyzw）注意你的 XML；一般 mujoco 是 qw, qx, qy, qz（wxyz）
        # 这里假设是 wxyz：如果你的数据是 xyzw，请根据仓库改位序！
        qw, qx, qy, qz = self.data.qpos[3:7]
        # 计算机体 z 轴在世界坐标的方向，上向量
        # 方向余弦矩阵 R(q)，第三列即 body z 轴，推导略
        R = mujoco.mju_quat2Mat(np.zeros(9), np.array([qw, qx, qy, qz]))
        up = np.array([R[6], R[7], R[8]], dtype=np.float32)  # 第三列
        return up

    def _tilt_angle(self, up):
        # 与世界 z 轴的夹角
        cosang = np.clip(up[2] / (np.linalg.norm(up)+1e-8), -1.0, 1.0)
        return float(np.arccos(cosang))

    def _observe(self):
        q = self.data.qpos[7:].copy()
        v = self.data.qvel[6:].copy()
        h = np.array([self._get_root_height()], dtype=np.float32)
        up = self._get_root_up()  # 3
        return np.concatenate([q, v, h, up]).astype(np.float32)

    def _terminated(self):
        # 终止：高度过低 或 倾斜过大
        if self._get_root_height() < 0.5:
            return True
        if self._tilt_angle(self._get_root_up()) > self.tilt_limit_rad:
            return True
        return False

    # ---------- Gym API ----------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)

        # 初始姿态：站立 + 小噪声
        self.data.qpos[:3] = np.array([0., 0., self.target_height], dtype=np.float64)
        self.data.qpos[3:7] = np.array([1., 0., 0., 0.], dtype=np.float64)  # wxyz: 单位四元数
        noise = self.rng.normal(0, 0.02, size=self.dof)
        self.data.qpos[7:] = self.qpos_ref + noise
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        obs = self._observe()
        info = {}
        return obs, info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        # 这里假设 actuator 是位置或通用控制；若是扭矩式，你可以转成 PD：
        # ctrl = Kp * (qpos_ref - qpos) - Kd * qvel
        # 为最小可运行，这里简单把 action 当成“期望关节增量”
        desired_q = self.data.qpos[7:].copy()
        desired_q += self.action_scale * action

        # 简化的“位置追踪”——将误差直接写入 qpos（研究时建议实现 PD + 扭矩控制）
        # 这里为了演示与可视化连通，采用多物理步的微调
        for _ in range(self.sim_steps_per_ctrl):
            blend = 0.2
            self.data.qpos[7:] = (1 - blend) * self.data.qpos[7:] + blend * desired_q
            mujoco.mj_forward(self.model, self.data)

        # 奖励
        up = self._get_root_up()
        upright = up[2]                      # 与世界 z 对齐，越接近 1 越好
        height = self._get_root_height()
        qvel = self.data.qvel[6:]
        a_cost = float(np.sum(np.square(action)))

        r_upright = upright                 # [-1,1]; 正常在[0,1]
        r_height  = -abs(height - self.target_height)
        r_vel     = -float(np.linalg.norm(qvel))
        r_action  = -a_cost

        reward = (self.rw_w_upright * r_upright
                  + self.rw_w_height  * r_height
                  + self.rw_w_vel     * r_vel
                  + self.rw_w_action  * r_action)

        self.step_count += 1
        terminated = self._terminated()
        truncated = self.step_count >= self.max_steps
        obs = self._observe()
        info = {"upright": r_upright, "height": height}
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.viewer.cam.lookat[:] = np.array([0, 0, 0.7])
                self.viewer.cam.distance = 3.0
                self.viewer.cam.azimuth = 180
                self.viewer.cam.elevation = -30
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            width, height = 800, 600
            renderer = mujoco.Renderer(self.model, width, height)
            renderer.update_scene(self.data)
            return renderer.render()
        else:
            return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
