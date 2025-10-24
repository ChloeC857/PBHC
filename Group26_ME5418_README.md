# Safe Falling for Humanoid Robots Via Motion-Mimic RL

This repository simulates and trains the G1 humanoid robot for walking behavior tests using MuJoCo and PPO as a preparation for safe falling research.

## Environment Setup

Before getting started, ensure that IssacGym is correctly installed and tested. After that, run the following command to set up the conda environment.

```bash
conda env create -f environment.yml
conda activate humanoid-safe-fall

## install and test isaacgym
wget https://developer.nvidia.com/isaac-gym-preview-4
tar -xvzf isaac-gym-preview-4
pip install -e isaacgym/python
```

## Project Structure
- `description`: providing description files for the G1 humanoid robot. Detailed information can be found in `description\robots\g1\g1_23dof_lock_wrist.xml`. Three terrain configurations (flat, ramp, stair) are defined in  `description\robots\g1\g1_23dof_lock_wrist_<TERRAIN>.xml`.
- `motion_source`: tools for extracting SMPL format data from videos.
- `smpl_retarget`: tools for retargeting SMPL format motinos to the G1 humanoid robot.
- `robot_motion_process`: visualization tools for robot format motion.
- `humanoidverse`: training RL policy.
- `video_files`: Storing the walking video, processed motion files, training outputs, and terrain visualizations for setting up the safe falling environment.
- `training_outputs`: Storing the walking policy and corresponding model files after 35,000 training iterations.

Note: Other folders are not used in the current stage.

## Policy Training

Replace `<MOTION_FILE_PATH>` in the command with the path to the target motion file.
We set 35,000 iterations for walking policy training.
```bash
python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=128 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof_lock_wrist \
+domain_rand=main \
+rewards=motion_tracking/main \
experiment_name=debug \
robot.motion.motion_file=<MOTION_FILE_PATH> \
seed=1 \
+device=cuda:0
```

## Demo Running

Replace `<PT_FILE_PATH>` in the command to convert a `.pt` file into an `.onnx` file. This command will also run the trained policy with the visualization in IssacGym. Our trained model is saved in `training_outputs/model_30000.pt`
```bash
python humanoidverse/eval_agent.py \
+device=cuda:0 \
+env.config.enforce_randomize_motion_start_eval=False \
+checkpoint=<PT_FILE_PATH>
```

## Results

### Raw walking video
![teaser](video_files/gifs/1_Youtube.gif)

### Extracted SMPL motions
![teaser](video_files/gifs/2_SMPL.gif)

### Retargeted G1 humanoid robot performance
![teaser](video_files/gifs/3_pkl.gif)

### Train Result after 35,000 iterations
Result A: the robot successfully walks 3 steps.

![teaser](video_files/gifs/4_Train_Result_for_Walking_A.gif)

Result B: the robot manages to walk only 1 step before losing balance.

![teaser](video_files/gifs/5_Train_Result_for_Walking_B.gif)

## Environment Visualization for Safe Falling

We define three types of terrain for safe falling environment. The following command can be run to visualize the real-time simulation of the robot's falling initialization. 

You can adjust the robot's initial height with defining parameter `+height=...` in the command.

```bash
# a flat plane
python .\robot_motion_process\vis_q_mj.py +terrain=flat

# a ramp
python .\robot_motion_process\vis_q_mj.py +terrain=ramp

# a staircase
python .\robot_motion_process\vis_q_mj.py +terrain=stair
```

A quick preview is also provided in the `.gif` files below:

### Humanoid robot on a flat plane
![teaser](video_files/gifs/6_g1_flat.gif)

### Humanoid robot on a ramp
![teaser](video_files/gifs/7_g1_ramp.gif)

### Humanoid robot on a staircase
![teaser](video_files/gifs/8_g1_stair.gif)
