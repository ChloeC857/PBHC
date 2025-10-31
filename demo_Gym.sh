conda activate humanoid-safe-fall

echo "Activated conda env: $CONDA_DEFAULT_ENV"

python humanoidverse/eval_agent.py \
+device=cuda:0 \
+env.config.enforce_randomize_motion_start_eval=False \
+checkpoint=training_outputs/model_30000.pt