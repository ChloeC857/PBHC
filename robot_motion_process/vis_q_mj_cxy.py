import os
import sys
import time
import argparse
import pdb
import os.path as osp

sys.path.append(os.getcwd())


# from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree

import numpy as np
# import math
# from copy import deepcopy
from collections import defaultdict
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as sRot
# import joblib
import hydra
from omegaconf import DictConfig, OmegaConf

# from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch

# TODO A: stand config
def _make_stand_motion(T=300, height=0.85):
    """Make a simple standing motion"""
    root_trans = np.tile(np.array([[0., 0., height]], dtype=np.float32), (T,1))
    root_rot   = np.tile(np.array([[0., 0., 0., 1.]], dtype=np.float32), (T,1))  # xyzw
    dof        = np.zeros((T, 29), dtype=np.float32)  # 23 dofs for g1_23dof
    motion = {
        'fps': 30,
        'root_trans_offset': root_trans,
        'root_rot': root_rot,
        'dof': dof
    }
    return motion


def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_makeConnector(scene.geoms[scene.ngeom-1],
                            mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                            point1[0], point1[1], point1[2],
                            point2[0], point2[1], point2[2])

def key_call_back( keycode):
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, speed, paused, rewind, motion_data_keys, contact_mask, curr_time, resave
    if chr(keycode) == "R":
        print("Reset")
        time_step = 0
    elif chr(keycode) == " ":
        print("Paused")
        paused = not paused
    elif keycode == 256 or chr(keycode) == "Q":
        print("Esc")
        os._exit(0)
    elif chr(keycode) == 'L':
        speed = speed * 1.5
        print("Speed: ", speed)
    elif chr(keycode) == 'K':
        speed = speed / 1.5
        print("Speed: ", speed)
    elif chr(keycode) == 'J':
        print("Toggle Rewind: ", not rewind)
        rewind = not rewind
    elif keycode == 262: #(Right)
        time_step+=dt
    elif keycode == 263: #(Left)
        time_step-=dt
    elif chr(keycode) == "Q":
        print('Modify left foot contact!!!')
        contact_mask[curr_time][0] = 1. - contact_mask[curr_time][0]
        resave = True
    elif chr(keycode) == "E":
        print('Modify right foot contact!!!')
        contact_mask[curr_time][1] = 1. - contact_mask[curr_time][1]
        resave = True
    else:
        print("not mapped", chr(keycode), keycode)
    
    
        
@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg : DictConfig) -> None:
    # TODO A: stand config
    stand_still = bool(getattr(cfg, "stand_still", False))
    frames = int(getattr(cfg, "frames", 300))
    height = float(getattr(cfg, "height", 0.85))
    
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, speed, paused, rewind, motion_data_keys, contact_mask, curr_time, resave
    curr_start, num_motions, motion_id, motion_acc, time_step, dt, speed, paused, rewind \
        = 0, 1, 0, set(), 0, 1/30, 1.0, False, False
    # if 'dt' in cfg:
    #     dt = cfg.dt
    # TODO A: add stand still motion
    if stand_still:
        #  make a stand still motion
        motion_data = {"stand": _make_stand_motion(T=frames, height=height)}
        motion_data_keys = list(motion_data.keys())
        curr_motion_key = motion_data_keys[0]
        curr_motion = motion_data[curr_motion_key]
        dt = 1.0 / curr_motion['fps']
        hang = False
        speed = 0.0
        contact_mask = None
        curr_time = 0
        resave = False
        vis_contact = False  # 避免断言错误
        print("[INFO] Stand-still visualization (no motion file).")

        # 初始化站立姿态
        humanoid_xml = "./description/robots/g1/g1_29dof_rev_1_0.xml"
        mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)
        mj_data = mujoco.MjData(mj_model)
        
        # 使用更小的时间步长以提高稳定性
        mj_model.opt.timestep = 0.001  # 1ms，比默认更小
        
        # 恢复重力，实现真实站立
        mj_model.opt.gravity[:] = np.array([0, 0, -9.81])
        
        # 设置初始高度，让脚刚好接触地面（G1机器人腿长约0.8m）
        # 根据G1的URDF，站立高度约0.75-0.8m
        initial_height = 0.77  # 精确调整
        mj_data.qpos[:3] = np.array([0, 0, initial_height])  # 根位置
        mj_data.qpos[3:7] = np.array([1, 0, 0, 0])   # 根旋转（wxyz）
        
        # 设置对称的站立姿态
        qpos_init = np.zeros(29)
        # 轻微弯曲膝盖以增加稳定性
        qpos_init[3] = 0.08     # left_knee（轻微弯曲）
        qpos_init[9] = 0.08     # right_knee
        # 踝关节轻微补偿
        qpos_init[4] = -0.04    # left_ankle_pitch
        qpos_init[10] = -0.04   # right_ankle_pitch
        
        mj_data.qpos[7:] = qpos_init
        mj_data.qvel[:] = 0  # 初始速度为0
        
        # 前向运动学
        mujoco.mj_forward(mj_model, mj_data)
        
        # 让机器人先静态settle（多次调用mj_forward直到力平衡）
        print("[INFO] Pre-settling the robot...")
        for _ in range(50):
            mujoco.mj_forward(mj_model, mj_data)
        
        # 记录settle后的姿态作为参考
        qpos_ref = mj_data.qpos.copy()
        
        # PD控制参数（腿部需要更强的控制）
        kp_leg = 100.0    # 腿部比例增益
        kd_leg = 8.0      # 腿部微分增益
        kp_waist = 80.0   # 腰部需要强控制以保持躯干直立
        kd_waist = 6.0    # 腰部微分增益
        kp_arm = 20.0     # 手臂比例增益
        kd_arm = 2.0      # 手臂微分增益
        
        step_count = 0

        with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_call_back) as viewer:
            viewer.cam.lookat[:] = np.array([0,0,0.7])
            viewer.cam.distance = 3.0
            viewer.cam.azimuth = 180
            viewer.cam.elevation = -30

            while viewer.is_running():
                step_count += 1
                
                # 计算关节误差和速度
                qpos = mj_data.qpos.copy()
                qvel = mj_data.qvel.copy()
                # 只对关节部分做PD（不含根）
                # qpos[7:] 是29个关节角度，qvel[6:] 是29个关节角速度（根占qvel前6个：3平移+3角速度）
                err = qpos[7:] - qpos_ref[7:]
                derr = qvel[6:]
                
                # 为不同关节设置不同的PD增益
                # 前12个是腿部关节（左腿6个+右腿6个），后面是腰部3个和手臂14个
                kp_gains = np.concatenate([
                    np.full(12, kp_leg),   # 腿部（左右各6个）
                    np.full(3, kp_waist),  # 腰部（yaw, roll, pitch）
                    np.full(14, kp_arm)    # 手臂（左右各7个）
                ])
                kd_gains = np.concatenate([
                    np.full(12, kd_leg),
                    np.full(3, kd_waist),
                    np.full(14, kd_arm)
                ])
                
                # PD控制输出
                ctrl = -kp_gains * err - kd_gains * derr
                
                # 添加躯干姿态稳定控制（防止前倾）
                # 获取根部的姿态（四元数）
                root_quat = qpos[3:7]  # wxyz
                # 转换为欧拉角以检测俯仰角
                root_rot = sRot.from_quat([root_quat[1], root_quat[2], root_quat[3], root_quat[0]])  # xyzw
                euler = root_rot.as_euler('xyz', degrees=False)
                pitch = euler[1]  # 俯仰角（正值=前倾，负值=后倾）
                
                # 如果躯干前倾，增加腰部pitch关节的反向力矩
                if abs(pitch) > 0.05:  # 超过约3度
                    waist_pitch_idx = 14  # waist_pitch_joint在控制数组中的索引（12个腿+2个腰部yaw/roll）
                    # 添加额外的稳定力矩（与倾斜方向相反）
                    stabilization_torque = -50.0 * pitch  # 比例控制
                    ctrl[waist_pitch_idx] += stabilization_torque
                    ctrl[waist_pitch_idx] = np.clip(ctrl[waist_pitch_idx], -30, 30)
                
                # 限制控制力矩在合理范围（使用更保守的限制）
                ctrl_limits = np.array([
                    60, 100, 60, 100, 40, 40,  # 左腿（增大腿部限制）
                    60, 100, 60, 100, 40, 40,  # 右腿
                    50, 40, 40,                # 腰部（增大腰部限制）
                    15, 15, 15, 15, 15, 15, 15, # 左臂
                    15, 15, 15, 15, 15, 15, 15  # 右臂
                ])
                ctrl = np.clip(ctrl, -ctrl_limits, ctrl_limits)
                
                # 写入控制器
                mj_data.ctrl[:] = ctrl

                mujoco.mj_step(mj_model, mj_data)

                viewer.sync()
                
                # 实时打印调试信息（每30帧打印一次）
                if step_count % 30 == 0:
                    print(f"Step: {step_count:5d} | Root Z: {mj_data.qpos[2]:.3f} | "
                          f"Max ctrl: {np.max(np.abs(ctrl)):.1f} | "
                          f"Max err: {np.max(np.abs(err)):.3f}", end='\r')

        return
    else: 
        motion_file = cfg.motion_file
        # motion_data = joblib.load(motion_file)
        motion_data = None
        motion_data_keys = list(motion_data.keys())
        curr_motion_key = motion_data_keys[motion_id]
        curr_motion = motion_data[curr_motion_key]
        print(motion_file)
        
        speed = 1.0 if 'speed' not in cfg else cfg.speed
        hang = False if 'hang' not in cfg else cfg.hang
        if 'fps' in curr_motion:
            dt = 1.0 / curr_motion['fps']
        elif 'dt' in cfg:
            dt = cfg.dt
        
        print("Motion file: ", motion_file)
        print("Motion length: ", motion_data[motion_data_keys[0]]['dof'].shape[0], 'frames')
        print("Speed: ", speed)
        print()

        if 'contact_mask' in curr_motion.keys():
            contact_mask = curr_motion['contact_mask']
        else:
            contact_mask = None
        curr_time = 0
        resave = False
    
        return


if __name__ == "__main__":
    main()
