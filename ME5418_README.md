# Safe Falling for Humanoid Robots Via Motion-Mimic RL

This repository simulates and trains the g1 humanoid robot for walking test behaviors using MuJoCo and PPO before safe falling research.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate humanoid-safe-fall
```

## Project Structure
- `description`: provide description file for G1 robot.
- `motion_source`: docs for getting SMPL format data.
- `smpl_retarget`: tools for SMPL to G1 robot retargeting.
- `smpl_vis`: tools for visualizing SMPL format data.
- `robot_motion_process`: tools for processing robot format motion. Including visualization, interpolation, and trajectory analysis.
- `humanoidverse`: training RL policy
- `example`: Demo motion file

## Demo Run

Run the following command to deploy the walking policy in MuJoCo.
```bash
python humanoidverse/urci.py +opt=record +simulator=mujoco +checkpoint=example/walk/exported/model_50000.onnx
```

## Result

Raw walking video
![teaser](video_files/Youtube.gif)

Extracted SMPL motions
![teaser](video_files/SMPL.gif)

pkl


pt

Humanoid robot in flat plane
![teaser](video_files/g1_flat.gif)

Humanoid robot in ramp
![teaser](video_files/g1_ramp.gif)

Humanoid robot in stairs
![teaser](video_files/g1_stair.gif)
