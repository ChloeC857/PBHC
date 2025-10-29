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

def _make_stand_motion(T, height):
    """
    Generate a simple standing motion.
    
    Args:
        T: Number of frames
        height: Robot pelvis height in meters
    
    Returns:
        dict: Motion data with root position, rotation, and joint angles
    """
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
    """
    Add a capsule visualization to MuJoCo scene.
    
    Args:
        scene: MuJoCo scene object
        point1, point2: Capsule endpoints (3D coordinates)
        radius: Capsule radius
        rgba: Color in RGBA format
    """
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
            'kp_leg': 60.0,
            'kd_leg': 8.0,
            'kp_waist': 40.0,
            'kd_waist': 4.0,
            'kp_arm': 15.0,
            'kd_arm': 1.5,
            'kp_trunk': 50.0,
            'kd_trunk': 5.0
        }
    
    elif terrain_type == 'ramp':
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
            'description': 'Ramp',
            'kp_leg': 100.0,
            'kd_leg': 12.0,
            'kp_waist': 80.0,
            'kd_waist': 8.0,
            'kp_arm': 20.0,
            'kd_arm': 2.0,
            'kp_trunk': 50.0,
            'kd_trunk': 5.0
        }
    
    elif terrain_type == 'stair':
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
            'description': 'Stairs',
            'kp_leg': 60.0,
            'kd_leg': 8.0,
            'kp_waist': 40.0,
            'kd_waist': 4.0,
            'kp_arm': 15.0,
            'kd_arm': 1.5,
            'kp_trunk': 50.0,
            'kd_trunk': 5.0
        }
    
    else:
        # Default to flat
        return get_terrain_init_params('flat')

def key_call_back(keycode):
    """
    Keyboard callback for interactive simulation control.
    
    Controls:
        R: Reset simulation
        Space: Pause/unpause
        L/K: Increase/decrease playback speed
        J: Toggle rewind
        Q/E: Modify foot contact masks (for motion editing)
        Esc: Exit
    """
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
    """
    Return terrain-specific XML file path.
    
    Args:
        terrain: 'flat', 'ramp', or 'stair'
    
    Returns:
        str: Path to corresponding XML file
    """
    if terrain == 'flat':
        humanoid_xml = "description/robots/g1/g1_29dof_rev_1_0.xml"
    elif terrain == 'ramp':
        humanoid_xml = "description/robots/g1/g1_23dof_lock_wrist_ramp.xml"
    elif terrain == 'stair':
        humanoid_xml = "description/robots/g1/g1_23dof_lock_wrist_stair.xml"
    else:
        humanoid_xml = "description/robots/g1/g1_29dof_rev_1_0.xml"
    return humanoid_xml
         
@hydra.main(version_base=None)
def main(cfg : DictConfig) -> None:
    """
    Main visualization function.
    
    Two modes:
        1. Stand-still mode: PD-controlled standing on different terrains
        2. Motion playback mode: Replay recorded motion sequences (not used)
    
    Configuration:
        stand_still: Enable standing mode, default True
        terrain: Terrain type - 'flat', 'ramp', or 'stair', default: 'flat'
        height: Robot pelvis height in meters, default: 0.9
        frames: Simulation duration in frames, default: 600
    """
    stand_still = bool(getattr(cfg, "stand_still", True))
    frames = int(getattr(cfg, "frames", 600))
    height = float(getattr(cfg, "height", 0.9))
    terrain = str(getattr(cfg, "terrain", "flat"))
    
    # Initialize global variables for motion playback control
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, speed, paused, rewind, motion_data_keys, contact_mask, curr_time, resave
    curr_start, num_motions, motion_id, motion_acc, time_step, dt, speed, paused, rewind \
        = 0, 1, 0, set(), 0, 1/30, 1.0, False, False
    # if 'dt' in cfg:
    #     dt = cfg.dt
    
    # ============================================================================
    # Stand-Still Mode: PD-Controlled Standing on Different Terrains
    # ============================================================================
    if stand_still:
        motion_data = {"stand": _make_stand_motion(T=frames, height=height)}
        motion_data_keys = list(motion_data.keys())
        curr_motion_key = motion_data_keys[0]
        curr_motion = motion_data[curr_motion_key]
        dt = 1.0 / curr_motion['fps']
        speed = 0.0
        contact_mask = None
        curr_time = 0
        resave = False
        print("[INFO] Stand-still visualization.")

        humanoid_xml = get_xml_path_for_terrain(terrain)
        print(f"[INFO] Loading MuJoCo model from: {humanoid_xml}")
        terrain_params = get_terrain_init_params(terrain)
        print(f"[INFO] Terrain: {terrain_params['description']}")
        
        # Initialize MuJoCo model and data
        mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)        
        mj_data = mujoco.MjData(mj_model)
        
        # Configure physics parameters
        mj_model.opt.timestep = 0.001  # 1ms timestep for stability
        mj_model.opt.gravity[:] = np.array([0, 0, -9.81])  # Standard gravity
        
        init_x = terrain_params['init_x']
        init_y = terrain_params['init_y']
        init_z_offset = terrain_params['init_z_offset']
        pitch = terrain_params['pitch']

        # Set robot initial pose (position + orientation)
        mj_data.qpos[0] = init_x
        mj_data.qpos[1] = init_y
        mj_data.qpos[2] = height + init_z_offset
        
        # Set root orientation (quaternion) based on terrain pitch
        robot_rot = sRot.from_euler('xyz', [0, pitch, 0])
        robot_quat_xyzw = robot_rot.as_quat()
        mj_data.qpos[3] = robot_quat_xyzw[3]  # w
        mj_data.qpos[4] = robot_quat_xyzw[0]  # x
        mj_data.qpos[5] = robot_quat_xyzw[1]  # y
        mj_data.qpos[6] = robot_quat_xyzw[2]  # z
        
        print(f"[INFO] Robot initialized at position ({init_x:.2f}, {init_y:.2f}, {mj_data.qpos[2]:.2f}), pitch={np.degrees(pitch):.2f}°")
        
        # Pre-settle: Let physics stabilize before control starts
        print("[INFO] Pre-settling the robot...")
        for _ in range(1000):
            mj_data.ctrl[:] = 0
            mujoco.mj_forward(mj_model, mj_data)
        
        # Save settled pose as reference for PD control
        qpos_ref = mj_data.qpos.copy()
        mj_model.qpos0[:] = qpos_ref
       
        # Extract terrain-specific PD control gains
        kp_leg = terrain_params['kp_leg']
        kd_leg = terrain_params['kd_leg']
        kp_waist = terrain_params['kp_waist']
        kd_waist = terrain_params['kd_waist']
        kp_arm = terrain_params['kp_arm']
        kd_arm = terrain_params['kd_arm']
        kp_trunk = terrain_params['kp_trunk']
        kd_trunk = terrain_params['kd_trunk']
        
        print(f"[INFO] PD Gains - Leg: kp={kp_leg}, kd={kd_leg} | Waist: kp={kp_waist}, kd={kd_waist} | Arm: kp={kp_arm}, kd={kd_arm}")
        
        step_count = 0
        ref_pitch = pitch  # Store reference pitch for stabilization

        # ========================================================================
        # Main Simulation Loop with PD Control
        # ========================================================================
        with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_call_back) as viewer:
            # Set camera parameters
            viewer.cam.lookat[:] = np.array(terrain_params['camera_lookat'])
            viewer.cam.distance = terrain_params['camera_distance']
            viewer.cam.azimuth = terrain_params['camera_azimuth']
            viewer.cam.elevation = terrain_params['camera_elevation']

            while viewer.is_running():
                step_count += 1
                
                if step_count < 100:
                    # Compute pose difference (root + joints)
                    pose_diff = np.linalg.norm(mj_data.qpos[:7] - qpos_ref[:7]) + np.linalg.norm(mj_data.qpos[7:] - qpos_ref[7:])
                    if pose_diff > 0.1:
                        print(f"[INFO] Detected large pose deviation ({pose_diff:.3f}) at step {step_count}, reapplying settled pose")
                        # Reapply settled pose and zero velocities
                        mj_data.qpos[:] = qpos_ref[:]
                        mj_data.qvel[:] = 0
                        # Update physics state
                        mujoco.mj_forward(mj_model, mj_data)
                        # Skip controller
                        viewer.sync()
                        continue

                # Compute joint errors for PD control (29-DoF humanoid, 87 hinge joints)
                qpos = mj_data.qpos.copy()
                qvel = mj_data.qvel.copy()

                # Skip 7 root DoFs (freejoint: 3 pos + 4 quat)
                joint_pos = qpos[7:]
                joint_vel = qvel[6:]

                err = joint_pos - qpos_ref[7:]
                derr = joint_vel

                # === Group DOFs by region (count in multiples of 3) ===
                # Group counts from XML
                n_leg   = 12   # 6 per leg
                n_waist = 3
                n_arm   = 14   # 7 per arm
                assert len(err) == n_leg + n_waist + n_arm
                total_dof = n_leg + n_waist + n_arm   # 29

                assert err.shape[0] == total_dof, f"Mismatch: expected {total_dof}, got {err.shape[0]}"

                # === PD gains ===
                kp_gains = np.concatenate([
                    np.full(n_leg,   kp_leg),
                    np.full(n_waist, kp_waist),
                    np.full(n_arm,   kp_arm)
                ])
                kd_gains = np.concatenate([
                    np.full(n_leg,   kd_leg),
                    np.full(n_waist, kd_waist),
                    np.full(n_arm,   kd_arm)
                ])

                # === PD torque ===
                ctrl = -kp_gains * err - kd_gains * derr

                # === Optional pitch stabilization ===
                root_quat = qpos[3:7]  # [x y z w] in MuJoCo
                root_rot = sRot.from_quat([root_quat[1], root_quat[2], root_quat[3], root_quat[0]])
                curr_pitch = root_rot.as_euler('xyz')[1]
                pitch_error = curr_pitch - ref_pitch

                # waist_pitch_joint is 15th joint in actuator order
                if abs(pitch_error) > 0.05:
                    waist_pitch_idx = 12 + 2  # waist_yaw(12), waist_roll(13), waist_pitch(14) → index 14-7=7?? let's count properly
                    waist_pitch_idx = n_leg + 2  # inside waist group (3rd in that group)
                    stabilization_torque = -60.0 * pitch_error
                    ctrl[waist_pitch_idx] += np.clip(stabilization_torque, -40, 40)

                # === Torque limits by region ===
                ctrl_limits = np.concatenate([
                    np.full(n_leg,   100.0),
                    np.full(n_waist, 80.0),
                    np.full(n_arm,   40.0)
                ])
                ctrl = np.clip(ctrl, -ctrl_limits, ctrl_limits)

                # Send torques
                mj_data.ctrl[:] = ctrl

                
                # Apply control torques and step simulation
                mj_data.ctrl[:] = ctrl
                mujoco.mj_step(mj_model, mj_data)
                viewer.sync()
                
                # Debug output
                if step_count % 30 == 0:
                    print(f"Step: {step_count:5d} | Root X: {mj_data.qpos[0]:.3f} Z: {mj_data.qpos[2]:.3f} | "
                          f"Pitch: {np.degrees(curr_pitch):.2f}° (ref: {np.degrees(ref_pitch):.2f}° err: {np.degrees(pitch_error):.2f}°) | "
                          f"Max ctrl: {np.max(np.abs(ctrl)):.1f} | "
                          f"Max err: {np.max(np.abs(err)):.3f}", end='\r')

        return
    
    # ========================================================================
    # Motion Playback Mode (Not Used)
    # ========================================================================
    
    else: 
        motion_file = cfg.motion_file
        # motion_data = joblib.load(motion_file)  # Uncomment when implementing
        motion_data = None  # Placeholder
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
