import torch
import hydra
from pathlib import Path
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from hydra.utils import instantiate
from humanoidverse.agents.mh_ppo import MHPPO
from humanoidverse.utils.helpers import pre_process_config
from humanoidverse.envs.base_task.base_task import BaseTask  # noqa: E402
from humanoidverse.agents.base_algo.base_algo import BaseAlgo  # noqa: E402
from loguru import logger

@hydra.main(config_path="config", config_name="base", version_base="1.1")
def main(config: OmegaConf):
    if hasattr(config, 'device'):
        if config.device is not None:
            device = config.device
        else:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    unresolved_conf = OmegaConf.to_container(config, resolve=False)
    pre_process_config(config)

    # ============================================================
    # 2. 初始化环境和算法（与 train_agent.py 一致）
    # ============================================================
    env: BaseTask = instantiate(config=config.env, device=device)
    experiment_save_dir = Path(config.experiment_dir)
    experiment_save_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Saving config file to {experiment_save_dir}")
    with open(experiment_save_dir / "config.yaml", "w") as file:
        OmegaConf.save(unresolved_conf, file)
    algo: BaseAlgo = instantiate(device=device, env=env, config=config.algo, log_dir=experiment_save_dir)
    algo.setup()

    # ============================================================
    # 3. 获取真实输入观测
    # ============================================================
    with torch.no_grad():
        obs_dict = env.reset_all()  # 环境初始化
        for k in obs_dict:
            obs_dict[k] = obs_dict[k].to(device)

    actor_obs = obs_dict["actor_obs"]
    critic_obs = obs_dict["critic_obs"]

    print("actor_obs shape:", actor_obs.shape)
    print("critic_obs shape:", critic_obs.shape)

    # ============================================================
    # 4. Forward pass（前向传播）
    # ============================================================
    algo.actor.update_distribution(actor_obs)
    actions = algo.actor.act(actor_obs)
    values = algo.critic.evaluate(critic_obs)

    print("Actor mean:", algo.actor.action_mean[0, :5])
    print("Actions:", actions[0, :5])
    print("Critic values:", values[0])

    # ============================================================
    # 5. Backward pass（反向传播）
    # ============================================================
    # 构造一个简单 loss：让动作接近 0，value 接近 1
    loss = (algo.actor.action_mean ** 2).mean() + ((values - 1) ** 2).mean()
    print("Loss value:", loss.item())

    loss.backward()

    # 查看梯度是否有效
    for name, param in algo.actor.named_parameters():
        if param.grad is not None:
            print(f"{name}: grad norm = {param.grad.norm():.4f}")
            break

    print("✅ Forward/Backward demo complete.")

if __name__ == "__main__":
    main()