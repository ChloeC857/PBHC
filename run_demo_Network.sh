conda activate pbhc

echo "Activated conda env: $CONDA_DEFAULT_ENV"

python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=demo_exp +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=128 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof_lock_wrist \
+domain_rand=main \
+rewards=motion_tracking/main \
experiment_name=debug \
robot.motion.motion_file=example/motion_data/fall1.pkl \
seed=1 \
+device=cuda:0