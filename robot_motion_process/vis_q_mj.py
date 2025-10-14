import os
import sys
import time
import argparse
import pdb
import os.path as osp

sys.path.append(os.getcwd())


# from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
import torch

import numpy as np
import math
from copy import deepcopy
from collections import defaultdict
import mujoco
import mujoco.viewer
import glfw
from scipy.spatial.transform import Rotation as sRot
import joblib
import hydra
from omegaconf import DictConfig, OmegaConf

from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch

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
        print("[INFO] Stand-still visualization (no motion file).")
    else: 
        motion_file = cfg.motion_file
        motion_data = joblib.load(motion_file)
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


    humanoid_xml = "./description/robots/g1/g1_29dof_rev_1_0.xml"
    print(humanoid_xml)
    
    vis_smpl = False if 'vis_smpl' not in cfg else cfg.vis_smpl
    vis_tau_key = 'tau' if 'vis_tau_key' not in cfg else cfg.vis_tau_key
    vis_tau = vis_tau_key in curr_motion if 'vis_tau' not in cfg else cfg.vis_tau
    vis_contact = 'contact_mask' in curr_motion if 'vis_contact' not in cfg else cfg.vis_contact
    
    if vis_smpl: assert 'smpl_joints' in curr_motion
    if vis_tau: assert vis_tau_key in curr_motion and not vis_contact
    if vis_contact: assert 'contact_mask' in curr_motion and not vis_tau

    if not vis_smpl:
        cfg_robot = OmegaConf.load("description/robots/g1/phc_g1_23dof.yaml")
        humanoid_fk = Humanoid_Batch(cfg_robot)  # load forward kinematics model

        # ---------------- 转 quaternion ----------------
        def motion_to_quat(curr_motion):
            # root_trans (T,3) -> tensor (1,T,3)
            root_trans = torch.from_numpy(curr_motion['root_trans_offset']).float().unsqueeze(0)
            
            if 'pose_aa' in curr_motion:
                axis_angle = curr_motion['pose_aa']  # (T, J, 3)
                T, J, _ = axis_angle.shape
                quat_list = []
                for t in range(T):
                    # 转所有关节的旋转向量为 quaternion
                    q_j = sRot.from_rotvec(axis_angle[t]).as_quat()  # xyzw
                    q_j = q_j[:, [3,0,1,2]]  # 转 wxyz
                    quat_list.append(q_j)
                pose_quat = np.stack(quat_list, axis=0)  # (T,J,4)
            elif 'dof' in curr_motion:
                dof = curr_motion['dof']  # (T, J)
                T, J = dof.shape
                quat_list = []
                for t in range(T):
                    q_joints = []
                    for j in range(J):
                        r = sRot.from_euler('z', dof[t,j])  # 假设绕 z 轴旋转
                        q = r.as_quat()                       # xyzw
                        q = np.array([q[3], q[0], q[1], q[2]])  # 转 wxyz
                        q_joints.append(q)
                    q_joints = np.stack(q_joints, axis=0)  # (J,4)
                    quat_list.append(q_joints)
                pose_quat = np.stack(quat_list, axis=0)   # (T,J,4)
            else:
                raise KeyError("curr_motion 中没有 pose_aa 或 dof")

            # 转 tensor 并加 batch 维度
            pose_quat = torch.from_numpy(pose_quat).float().unsqueeze(0)  # (1,T,J,4)
            return pose_quat, root_trans  # (1,T,J,4), (1,T,3)


        pose_quat, root_trans = motion_to_quat(curr_motion)

        print("[DEBUG] pose_quat.shape:", pose_quat.shape)
        print("[DEBUG] pose_quat example:", pose_quat.view(-1, pose_quat.shape[-1])[0])
        print("[DEBUG] last_dim unique:", torch.unique(torch.tensor([p.shape[-1] for p in pose_quat.view(-1, 1, 1, pose_quat.shape[-1])])).tolist())

        fk_return = humanoid_fk.fk_batch(pose_quat, root_trans)
        joint_gt = fk_return.global_translation_extend[0]
    

    mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = dt
    
    print("Init Pose: ",(np.array(np.concatenate(
        [curr_motion['root_trans_offset'][0],curr_motion['root_rot'][0][[3, 0, 1, 2]], curr_motion['dof'][0]]
        ),dtype=np.float32)).__repr__())
    
    # breakpoint()
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_call_back) as viewer:
        
        viewer.cam.lookat[:] = np.array([0,0,0.7])
        viewer.cam.distance = 3.0        
        viewer.cam.azimuth = 180         
        viewer.cam.elevation = -30                      # 负值表示从上往下看viewer
        
        for _ in range(50):
                # not display the ball
            # add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.05, np.array([1, 0, 0, 0]))
            add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.05, np.array([1, 0, 0, 1]))
        
        # breakpoint()
        while viewer.is_running():
            step_start = time.time()
            if time_step >= curr_motion['dof'].shape[0]*dt:
                time_step -= curr_motion['dof'].shape[0]*dt
            curr_time = round(time_step/dt) % curr_motion['dof'].shape[0]
            
            if hang:
                mj_data.qpos[:3] = np.array([0,0,0.8])
            else:
                mj_data.qpos[:3] = curr_motion['root_trans_offset'][curr_time]
            mj_data.qpos[3:7] = curr_motion['root_rot'][curr_time][[3, 0, 1, 2]] #xyzw 2 wxyz
            mj_data.qpos[7:] = curr_motion['dof'][curr_time]
            
            
            mujoco.mj_forward(mj_model, mj_data)
            if not paused:
                time_step += dt * (1 if not rewind else -1) * speed
            
                
            if vis_smpl:
                joint_gt = motion_data[curr_motion_key]['smpl_joints']
                if not np.all(joint_gt[curr_time] == 0):
                    for i in range(joint_gt.shape[1]):
                        viewer.user_scn.geoms[i].pos = joint_gt[curr_time, i]
            else:
                for i in range(23):
                    viewer.user_scn.geoms[i+1].pos = joint_gt[curr_time, i+1]
            
            if vis_contact: 
                viewer.user_scn.geoms[6].rgba = np.array([0, 1-curr_motion['contact_mask'][curr_time, 0], 0, 1])
                viewer.user_scn.geoms[12].rgba = np.array([0, 1-curr_motion['contact_mask'][curr_time, 1], 0, 1])
                
            if vis_tau:
                scale_factor = 0.1
                for i in range(23):
                    tau = curr_motion[vis_tau_key][curr_time, i]
                    color_gradient = abs(tau) * scale_factor
                    if tau > 0:
                        viewer.user_scn.geoms[i+1].rgba = np.array([0.8,0.1,0.1,0.1+color_gradient])
                        # viewer.user_scn.geoms[i+1].rgba = np.array([0.1+color_gradient,0.,0.,1.])
                    elif tau < 0:
                        viewer.user_scn.geoms[i+1].rgba = np.array([0.1,0.8,0.1,0.1+color_gradient])
                        # viewer.user_scn.geoms[i+1].rgba = np.array([0,0.1+color_gradient,0.,1.])
                        
            
                

            viewer.sync()
            time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
                
            print("Frame ID: ",curr_time,'\t | Times ',f"{time_step:4f}",end='\r\b')

            # ----------------- mouse joint interaction (simple) -----------------
            # Register callbacks once (attach to viewer.window). We set them here so
            # they run in the same context as the viewer. Left-click to select the
            # nearest joint marker; while holding left button, horizontal mouse
            # movement will change the corresponding joint DOF (qpos index offset by 7).
            try:
                if not hasattr(viewer, '_mouse_callbacks_registered'):
                    selected = {'dof': None}
                    last_cursor = {'x': 0.0, 'y': 0.0}
                    sensitivity = 0.01  # angle per pixel
                    pick_threshold = 0.35  # meters (fallback spatial threshold)
                    original_colors = {}
                    joint_limits_min = np.full(23, -np.pi)
                    joint_limits_max = np.full(23, np.pi)

                    def _get_camera_frame():
                        cam = viewer.cam
                        lookat = np.array(cam.lookat)
                        az = np.deg2rad(cam.azimuth)
                        el = np.deg2rad(cam.elevation)
                        dist_cam = cam.distance
                        cam_pos = lookat + dist_cam * np.array([
                            np.cos(el) * np.sin(az),
                            -np.cos(el) * np.cos(az),
                            np.sin(el)
                        ])
                        forward = lookat - cam_pos
                        forward /= (np.linalg.norm(forward) + 1e-9)
                        world_up = np.array([0., 0., 1.])
                        right = np.cross(forward, world_up)
                        if np.linalg.norm(right) < 1e-6:
                            right = np.array([1., 0., 0.])
                        else:
                            right /= np.linalg.norm(right)
                        up = np.cross(right, forward)
                        up /= (np.linalg.norm(up) + 1e-9)
                        return cam_pos, lookat, forward, right, up

                    def _project_to_screen(pos):
                        try:
                            vp = viewer.viewport
                            w, h = vp.width, vp.height
                            cam_pos, lookat, forward, right, up = _get_camera_frame()
                            vec = pos - cam_pos
                            zc = np.dot(vec, forward)
                            if zc <= 1e-6:
                                return None
                            xc = np.dot(vec, right)
                            yc = np.dot(vec, up)
                            fov = np.deg2rad(60.0)
                            aspect = w / (h + 1e-9)
                            sx = 0.5 + (xc / zc) / (np.tan(fov/2) * aspect) * 0.5
                            sy = 0.5 - (yc / zc) / (np.tan(fov/2)) * 0.5
                            return np.array([sx * w, sy * h])
                        except Exception:
                            return None

                    def _pick_nearest_joint(x, y):
                        # Try precise screen-space projection first, fallback to spatial score
                        best_idx = None
                        best_dist = float('inf')
                        for i in range(23):
                            pos = joint_gt[curr_time, i+1]
                            screen = _project_to_screen(pos)
                            if screen is not None:
                                d2 = np.linalg.norm(screen - np.array([x, y]))
                                if d2 < best_dist:
                                    best_dist = d2
                                    best_idx = i

                        if best_idx is not None and best_dist < 60.0:  # pixels threshold
                            return best_idx

                        # fallback to previous approximate spatial metric
                        best_idx = None
                        best_score = float('inf')
                        cam_pos, lookat, forward, right, up = _get_camera_frame()
                        for i in range(23):
                            pos = joint_gt[curr_time, i+1]
                            vec = pos - cam_pos
                            d = np.linalg.norm(vec)
                            cosang = np.dot(vec, forward) / (np.linalg.norm(vec) * np.linalg.norm(forward) + 1e-9)
                            cosang = np.clip(cosang, -1.0, 1.0)
                            ang = np.arccos(cosang)
                            score = ang * d
                            if score < best_score:
                                best_score = score
                                best_idx = i

                        if best_idx is not None:
                            pos_best = joint_gt[curr_time, best_idx+1]
                            if np.linalg.norm(pos_best - (lookat)) > 3.0 and best_score > pick_threshold:
                                return None
                        return best_idx

                    def _highlight_geom(idx, highlight=True):
                        geom_idx = idx + 1
                        try:
                            if highlight:
                                # save original
                                if geom_idx not in original_colors:
                                    original_colors[geom_idx] = viewer.user_scn.geoms[geom_idx].rgba.copy()
                                viewer.user_scn.geoms[geom_idx].rgba = np.array([1.0, 1.0, 0.0, 1.0])
                            else:
                                if geom_idx in original_colors:
                                    viewer.user_scn.geoms[geom_idx].rgba = original_colors[geom_idx]
                                    del original_colors[geom_idx]
                        except Exception:
                            ...

                    def mouse_button_callback(window, button, action, mods):
                        if button == glfw.MOUSE_BUTTON_LEFT:
                            x, y = glfw.get_cursor_pos(window)
                            if action == glfw.PRESS:
                                last_cursor['x'], last_cursor['y'] = x, y
                                sel = _pick_nearest_joint(x, y)
                                selected['dof'] = sel
                                if sel is not None:
                                    print(f"Selected joint dof: {sel}")
                                    _highlight_geom(sel, True)
                            elif action == glfw.RELEASE:
                                if selected['dof'] is not None:
                                    _highlight_geom(selected['dof'], False)
                                selected['dof'] = None

                    def cursor_pos_callback(window, xpos, ypos):
                        if selected['dof'] is None:
                            return
                        dx = xpos - last_cursor['x']
                        # update last cursor immediately so movement is incremental
                        last_cursor['x'], last_cursor['y'] = xpos, ypos
                        dof_idx = selected['dof']
                        if dof_idx is None:
                            return
                        qpos_idx = 7 + dof_idx
                        # apply horizontal motion to joint angle with clamping
                        val = mj_data.qpos[qpos_idx] + dx * sensitivity
                        val = float(np.clip(val, joint_limits_min[dof_idx], joint_limits_max[dof_idx]))
                        mj_data.qpos[qpos_idx] = val
                        # forward kinematics so the model updates immediately
                        mujoco.mj_forward(mj_model, mj_data)

                    glfw.set_mouse_button_callback(viewer.window, mouse_button_callback)
                    glfw.set_cursor_pos_callback(viewer.window, cursor_pos_callback)
                    viewer._mouse_callbacks_registered = True
            except Exception:
                # safe fallback: if viewer/window or glfw not available, ignore
                ...

    if resave:
        motion_data[curr_motion_key]['contact_mask'] = contact_mask
        motion_file = motion_file.split('.')[0]+'_edit_cont.pkl'
        print(motion_file)
        joblib.dump(motion_data, motion_file)

if __name__ == "__main__":
    main()
