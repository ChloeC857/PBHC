from humanoidverse.utils.inference_helpers import export_policy_as_onnx
from humanoidverse.agents.base_algo.base_algo import BaseAlgo
from omegaconf import OmegaConf

# 1. 加载配置和算法
config = OmegaConf.load("example/walk/config.yaml")
algo = BaseAlgo(env=None, device="cuda:0")
algo.load("example/walk/model_30000.pt")

# 2. 准备样例输入
example_obs = algo.get_example_obs()

# 3. 导出 ONNX
export_policy_as_onnx(algo.inference_model, "example/walk/exported", "model_30000.onnx", example_obs)
