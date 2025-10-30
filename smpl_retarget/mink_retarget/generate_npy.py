import xml.etree.ElementTree as ET
import yaml
import numpy as np

xml_path = "description/robots/g1/smpl_humanoid.xml"
yaml_path = "humanoidverse/config/robot/g1/g1_29dof_rev_1_0.yaml"
output_path = "description/robots/g1/dof_axis_29dof.npy"

# 读取 YAML
with open(yaml_path, 'r') as f:
    robot_yaml = yaml.safe_load(f)
target_names = robot_yaml['robot']['dof_names']

# 名称映射表（YAML → XML）
name_map = {
    # --- left leg
    "left_hip_yaw_joint": "L_Hip_x",
    "left_hip_roll_joint": "L_Hip_y",
    "left_hip_pitch_joint": "L_Hip_z",
    "left_knee_joint": "L_Knee_x",
    "left_ankle_pitch_joint": "L_Ankle_y",
    "left_ankle_roll_joint": "L_Ankle_x",

    # --- right leg
    "right_hip_yaw_joint": "R_Hip_x",
    "right_hip_roll_joint": "R_Hip_y",
    "right_hip_pitch_joint": "R_Hip_z",
    "right_knee_joint": "R_Knee_x",
    "right_ankle_pitch_joint": "R_Ankle_y",
    "right_ankle_roll_joint": "R_Ankle_x",

    # --- waist
    "waist_yaw_joint": "Torso_x",
    "waist_roll_joint": "Torso_y",
    "waist_pitch_joint": "Torso_z",

    # --- left arm
    "left_shoulder_yaw_joint": "L_Shoulder_x",
    "left_shoulder_roll_joint": "L_Shoulder_y",
    "left_shoulder_pitch_joint": "L_Shoulder_z",
    "left_elbow_joint": "L_Elbow_x",
    "left_wrist_roll_joint": "L_Wrist_x",
    "left_wrist_pitch_joint": "L_Wrist_y",
    "left_wrist_yaw_joint": "L_Wrist_z",

    # --- right arm
    "right_shoulder_yaw_joint": "R_Shoulder_x",
    "right_shoulder_roll_joint": "R_Shoulder_y",
    "right_shoulder_pitch_joint": "R_Shoulder_z",
    "right_elbow_joint": "R_Elbow_x",
    "right_wrist_roll_joint": "R_Wrist_x",
    "right_wrist_pitch_joint": "R_Wrist_y",
    "right_wrist_yaw_joint": "R_Wrist_z",
}

# 解析 XML
tree = ET.parse(xml_path)
root = tree.getroot()
xml_joint_map = {
    joint.attrib["name"]: joint.attrib.get("axis", "1 0 0")
    for joint in root.iter("joint")
    if "name" in joint.attrib
}

# 生成对齐后的 axis 数组
axes = []
unmatched = []

for name in target_names:
    xml_name = name_map.get(name)
    if xml_name in xml_joint_map:
        axis = np.fromstring(xml_joint_map[xml_name], sep=' ')
        axes.append(axis)
    else:
        unmatched.append(name)

axes = np.array(axes, dtype=np.float32)
np.save(output_path, axes)

print(f"✅ Saved {axes.shape[0]} joint axes to {output_path}")
if unmatched:
    print("⚠️ 未匹配到的 joints:", unmatched)
