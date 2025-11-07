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
    dof        = np.zeros((T, 23), dtype=np.float32)  # 23 dofs for g1_23dof
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


def get_terrain_init_params(terrain_type):
    """
    Returns terrain-specific initialization parameters for robot placement.
    
    Args:
        terrain_type: 'flat', 'ramp', or 'stair'
    
    Returns:
        dict: Contains init_x, init_y, init_z_offset, pitch, camera_lookat, camera_distance
    """
    pelvis_to_foot = 0.793  # Vertical distance from pelvis to foot
    
    if terrain_type == 'flat':
        return {
            'init_x': 0.0,
            'init_y': 0.0,
            'init_z_offset': 0.0,
            'pitch': 0.0,
            'camera_lookat': [0.0, 0.0, 0.9],
            'camera_distance': 3.0,
            'camera_azimuth': 180,
            'camera_elevation': -30,
            'description': 'Flat ground',
            # PD control parameters for flat terrain (moderate gains)
            'kp_leg': 60.0,
            'kd_leg': 8.0,
            'kp_waist': 40.0,
            'kd_waist': 4.0,
            'kp_arm': 15.0,
            'kd_arm': 1.5
        }
    
    elif terrain_type == 'ramp':
        # Ramp parameters from XML: pos="2.0 0 0.05" quat="0.9962 0 0.0872 0"
        ramp_quat = np.array([0.9962, 0, 0.0872, 0])  # wxyz
        ramp_rot = sRot.from_quat([ramp_quat[1], ramp_quat[2], ramp_quat[3], ramp_quat[0]])  # xyzw
        ramp_euler = ramp_rot.as_euler('xyz', degrees=False)
        ramp_pitch = ramp_euler[1]  # ~5 degrees
        
        ramp_x_pos = 2.0
        ramp_base_height = 0.05
        ramp_thickness = 0.05
        foot_offset_x = 0.05
        
        ramp_top_center_z = ramp_base_height + ramp_thickness
        foot_z_on_ramp = ramp_top_center_z + np.tan(ramp_pitch) * foot_offset_x
        pelvis_z = foot_z_on_ramp + pelvis_to_foot * np.cos(ramp_pitch)
        
        return {
            'init_x': ramp_x_pos,
            'init_y': 0.0,
            'init_z_offset': pelvis_z - pelvis_to_foot,  # Offset from default height
            'pitch': ramp_pitch,
            'camera_lookat': [ramp_x_pos, 0.0, 0.9],
            'camera_distance': 3.5,
            'camera_azimuth': 180,
            'camera_elevation': -30,
            'description': f'5-degree ramp at x={ramp_x_pos}',
            # PD control parameters for ramp (higher gains to resist sliding)
            'kp_leg': 100.0,
            'kd_leg': 12.0,
            'kp_waist': 80.0,
            'kd_waist': 8.0,
            'kp_arm': 20.0,
            'kd_arm': 2.0
        }
    
    elif terrain_type == 'stair':
        # Stairs start at x=1.8, first step at x=1.55, height=0.05
        stair_x_pos = 1.55  # Center of first step
        stair_height = 0.05  # First step height
        stair_thickness = 0.05
        
        return {
            'init_x': stair_x_pos,
            'init_y': 0.0,
            'init_z_offset': stair_height + stair_thickness,
            'pitch': 0.0,
            'camera_lookat': [stair_x_pos, 0.0, 0.9],
            'camera_distance': 4.0,
            'camera_azimuth': 180,
            'camera_elevation': -25,
            'description': f'Stairs starting at x=1.8, robot on first step',
            # PD control parameters for stairs (high gains for stability on steps)
            'kp_leg': 60.0,
            'kd_leg': 8.0,
            'kp_waist': 40.0,
            'kd_waist': 4.0,
            'kp_arm': 15.0,
            'kd_arm': 1.5
        }
    
    else:
        # Default to flat
        return get_terrain_init_params('flat')

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


def get_xml_path_for_terrain(terrain:str):
    if terrain == 'flat':
        humanoid_xml = "description/robots/g1/g1_23dof_lock_wrist_flat.xml"
    elif terrain == 'ramp':
        humanoid_xml = "description/robots/g1/g1_23dof_lock_wrist_ramp.xml"
    elif terrain == 'stair':
        humanoid_xml = "description/robots/g1/g1_23dof_lock_wrist_stair.xml"
    else:
        humanoid_xml = "description/robots/g1/g1_23dof_lock_wrist_flat.xml"
    return humanoid_xml
         
@hydra.main(version_base=None)
def main(cfg : DictConfig) -> None:
    # TODO A: stand config
    stand_still = bool(getattr(cfg, "stand_still", False))
    frames = int(getattr(cfg, "frames", 300))
    height = float(getattr(cfg, "height", 0.9))
    terrain = str(getattr(cfg, "terrain", "flat"))
    
    print("Configuration:\n", OmegaConf.to_yaml(cfg))
    
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

        # Get terrain-specific XML and initialization parameters
        humanoid_xml = get_xml_path_for_terrain(terrain)
        terrain_params = get_terrain_init_params(terrain)
        print(f"[INFO] Terrain: {terrain_params['description']}")
        
        # Initialize robot model
        mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)        
        mj_data = mujoco.MjData(mj_model)
        
        # 使用更小的时间步长以提高稳定性
        mj_model.opt.timestep = 0.001  # 1ms，比默认更小
        
        # 恢复重力，实现真实站立
        mj_model.opt.gravity[:] = np.array([0, 0, -9.81])
        
        # Extract terrain-specific parameters
        init_x = terrain_params['init_x']
        init_y = terrain_params['init_y']
        init_z_offset = terrain_params['init_z_offset']
        pitch = terrain_params['pitch']

        # Set robot initial position based on terrain
        mj_data.qpos[0] = init_x
        mj_data.qpos[1] = init_y
        mj_data.qpos[2] = height + init_z_offset
        
        # Set root orientation based on terrain pitch
        robot_rot = sRot.from_euler('xyz', [0, pitch, 0])
        robot_quat_xyzw = robot_rot.as_quat()
        mj_data.qpos[3] = robot_quat_xyzw[3]  # w
        mj_data.qpos[4] = robot_quat_xyzw[0]  # x
        mj_data.qpos[5] = robot_quat_xyzw[1]  # y
        mj_data.qpos[6] = robot_quat_xyzw[2]  # z
        
        print(f"[INFO] Robot initialized at position ({init_x:.2f}, {init_y:.2f}, {mj_data.qpos[2]:.2f}), pitch={np.degrees(pitch):.2f}°")        # 让机器人先静态settle（多次调用mj_forward直到力平衡）
        print("[INFO] Pre-settling the robot...")
        for _ in range(1000):
            mj_data.ctrl[:] = 0
            mujoco.mj_forward(mj_model, mj_data)
        
        
        # 记录settle后的姿态作为参考
        qpos_ref = mj_data.qpos.copy()
        
        mj_model.qpos0[:] = qpos_ref
        print("[INFO] Updated mj_model.qpos0 with settled pose (for Reset consistency)")
        
        # PD控制参数（根据地形类型自动设置）
        kp_leg = terrain_params['kp_leg']
        kd_leg = terrain_params['kd_leg']
        kp_waist = terrain_params['kp_waist']
        kd_waist = terrain_params['kd_waist']
        kp_arm = terrain_params['kp_arm']
        kd_arm = terrain_params['kd_arm']
        
        print(f"[INFO] PD Gains - Leg: kp={kp_leg}, kd={kd_leg} | Waist: kp={kp_waist}, kd={kd_waist} | Arm: kp={kp_arm}, kd={kd_arm}")
        
        step_count = 0
        
        # Store reference pitch for stabilization
        ref_pitch = pitch

        with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_call_back) as viewer:
            # Set camera based on terrain parameters
            viewer.cam.lookat[:] = np.array(terrain_params['camera_lookat'])
            viewer.cam.distance = terrain_params['camera_distance']
            viewer.cam.azimuth = terrain_params['camera_azimuth']
            viewer.cam.elevation = terrain_params['camera_elevation']

            while viewer.is_running():
                step_count += 1
                # detect large unexpected deviation (e.g. GUI Reset) and reapply settled pose
                if step_count < 200:
                    # compute pose difference (only root pos + joint angles)
                    pose_diff = np.linalg.norm(mj_data.qpos[:7] - qpos_ref[:7]) + np.linalg.norm(mj_data.qpos[7:] - qpos_ref[7:])
                    if pose_diff > 0.1:
                        print(f"[INFO] Detected large pose deviation ({pose_diff:.3f}) at step {step_count}, reapplying settled pose")
                        # Reapply settled pose and zero velocities to avoid large impulses
                        mj_data.qpos[:] = qpos_ref[:]
                        mj_data.qvel[:] = 0
                        # forward to update derived quantities
                        mujoco.mj_forward(mj_model, mj_data)
                        # skip controller this iteration so contacts stabilize
                        viewer.sync()
                        continue

                # 计算关节误差和速度
                qpos = mj_data.qpos.copy()
                qvel = mj_data.qvel.copy()
                # 只对关节部分做PD（不含根）
                # qpos[7:] 是23个关节角度，qvel[6:] 是23个关节角速度（根占qvel前6个：3平移+3角速度）
                err = qpos[7:] - qpos_ref[7:]
                derr = qvel[6:]
                
                # 为不同关节设置不同的PD增益
                # 23自由度：腿部12个 + 腰部3个 + 手臂8个（左右各4个：shoulder_pitch/roll/yaw, elbow）
                kp_gains = np.concatenate([
                    np.full(12, kp_leg),   # 腿部（左右各6个）
                    np.full(3, kp_waist),  # 腰部（yaw, roll, pitch）
                    np.full(8, kp_arm)     # 手臂（左右各4个，无wrist）
                ])
                kd_gains = np.concatenate([
                    np.full(12, kd_leg),
                    np.full(3, kd_waist),
                    np.full(8, kd_arm)
                ])
                
                # PD控制输出
                ctrl = -kp_gains * err - kd_gains * derr
                
                # 添加躯干姿态稳定控制（相对于地形保持正确姿态）
                # 获取根部的姿态（四元数）
                root_quat = qpos[3:7]  # wxyz
                # 转换为欧拉角以检测俯仰角
                root_rot = sRot.from_quat([root_quat[1], root_quat[2], root_quat[3], root_quat[0]])  # xyzw
                euler = root_rot.as_euler('xyz', degrees=False)
                curr_pitch = euler[1]  # 俯仰角（正值=前倾，负值=后倾）
                
                # 计算相对于参考姿态的俯仰角偏差（机器人应该与地形平行）
                pitch_error = curr_pitch - ref_pitch
                
                # 如果躯干相对斜坡倾斜，增加腰部pitch关节的反向力矩
                if abs(pitch_error) > 0.05:  # 超过约3度
                    waist_pitch_idx = 14  # waist_pitch_joint在控制数组中的索引（12个腿+2个腰部yaw/roll）
                    # 添加额外的稳定力矩（与相对倾斜方向相反）
                    stabilization_torque = -50.0 * pitch_error  # 比例控制
                    ctrl[waist_pitch_idx] += stabilization_torque
                    ctrl[waist_pitch_idx] = np.clip(ctrl[waist_pitch_idx], -30, 30)
                
                # 限制控制力矩在合理范围（23自由度）
                ctrl_limits = np.array([
                    60, 100, 60, 100, 40, 40,  # 左腿6个
                    60, 100, 60, 100, 40, 40,  # 右腿6个
                    50, 40, 40,                # 腰部3个
                    15, 15, 15, 15,            # 左臂4个（无wrist）
                    15, 15, 15, 15             # 右臂4个（无wrist）
                ])
                ctrl = np.clip(ctrl, -ctrl_limits, ctrl_limits)
                
                # 写入控制器
                mj_data.ctrl[:] = ctrl

                mujoco.mj_step(mj_model, mj_data)

                viewer.sync()
                
                # 实时打印调试信息（每30帧打印一次）
                if step_count % 30 == 0:
                    print(f"Step: {step_count:5d} | Root X: {mj_data.qpos[0]:.3f} Z: {mj_data.qpos[2]:.3f} | "
                          f"Pitch: {np.degrees(curr_pitch):.2f}° (ref: {np.degrees(ref_pitch):.2f}° err: {np.degrees(pitch_error):.2f}°) | "
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