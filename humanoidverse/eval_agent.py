import os
import sys
from pathlib import Path
import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from humanoidverse.utils.logging import HydraLoggerBridge
import logging
from loguru import logger
from humanoidverse.utils.config_utils import *  # noqa: E402, F403
from humanoidverse.utils.devtool import pdb_decorator
import threading

# Add root directory to Python path for relative imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def on_press(key, env):
    """
    Keyboard callback function for interactive environment control during evaluation.
    
    This function handles real-time keyboard inputs to control the evaluation environment:
    - 'n': Switch to the next task in multi-task evaluation scenarios
    - '1'/'2': Increase/decrease force applied to the left hand (z-axis)
    - '3'/'4': Increase/decrease force applied to the right hand (z-axis)
    
    Args:
        key: Keyboard key object from pynput listener
        env: The environment instance to control
    """
    try:
        if key.char == 'n':
            env.next_task()
            logger.info("Moved to the next task.")
        # Force Control
        if hasattr(key, 'char'):
            if key.char == '1':
                env.apply_force_tensor[:, env.left_hand_link_index, 2] += 1.0
                logger.info(f"Left hand force: {env.apply_force_tensor[:, env.left_hand_link_index, :]}")
            elif key.char == '2':
                env.apply_force_tensor[:, env.left_hand_link_index, 2] -= 1.0
                logger.info(f"Left hand force: {env.apply_force_tensor[:, env.left_hand_link_index, :]}")
            elif key.char == '3':
                env.apply_force_tensor[:, env.right_hand_link_index, 2] += 1.0
                logger.info(f"Right hand force: {env.apply_force_tensor[:, env.right_hand_link_index, :]}")
            elif key.char == '4':
                env.apply_force_tensor[:, env.right_hand_link_index, 2] -= 1.0
                logger.info(f"Right hand force: {env.apply_force_tensor[:, env.right_hand_link_index, :]}")
    except AttributeError:
        pass

def listen_for_keypress(env):
    """
    Background thread function to listen for keyboard inputs during evaluation.
    
    Creates a keyboard listener that captures key presses and forwards them to
    the on_press callback function. This allows real-time interaction with the
    running simulation without blocking the main evaluation loop.
    
    Args:
        env: The environment instance to pass to the keyboard callback
    """
    return
    with keyboard.Listener(on_press=lambda key: on_press(key, env)) as listener:
        listener.join()



# from humanoidverse.envs.base_task.base_task import BaseTask
# from humanoidverse.envs.base_task.omnih2o_cfg import OmniH2OCfg
@hydra.main(config_path="config", config_name="base_eval")
# @pdb_decorator
def main(override_config: OmegaConf):
    """
    Main evaluation function for trained RL agents in HumanoidVerse framework.
    
    This function orchestrates the complete evaluation pipeline:
    1. Configuration loading and merging (training config + eval overrides)
    2. Logging setup (file + console output)
    3. Simulator initialization (IsaacSim or IsaacGym)
    4. Environment and algorithm instantiation
    5. Checkpoint loading
    6. Policy export (JIT/ONNX formats)
    7. Policy evaluation
    
    Args:
        override_config: Hydra configuration with evaluation parameters
        
    Key Configuration Options:
        - checkpoint: Path to trained model checkpoint (.pt file)
        - eval_overrides: Parameters to override from training config
        - num_envs: Number of parallel environments for evaluation
        - device: Computing device (cuda/cpu)
        - headless: Whether to run without GUI rendering
        
    Workflow:
        1. If checkpoint provided: Load training config and merge with eval config
        2. Setup logging (Hydra log file + console with configurable level)
        3. Detect simulator type and initialize appropriately
        4. Create environment with single instance (num_envs=1)
        5. Load trained policy from checkpoint
        6. Export policy to JIT/ONNX for deployment
        7. Run policy evaluation with metrics logging
    """
    # logging to hydra log file
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "eval.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")

    # Get log level from LOGURU_LEVEL environment variable or use INFO as default
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())

    os.chdir(hydra.utils.get_original_cwd())

    # ============================================================================
    # Configuration Loading and Merging Strategy
    # ============================================================================
    
    # If checkpoint path is provided in the configuration
    if override_config.checkpoint is not None:
        has_config = True
        checkpoint = Path(override_config.checkpoint)
        config_path = checkpoint.parent / "config.yaml"
        
        # Search for config.yaml in checkpoint directory hierarchy
        if not config_path.exists():
            config_path = checkpoint.parent.parent / "config.yaml"
            if not config_path.exists():
                has_config = False
                logger.error(f"Could not find config path: {config_path}")
        
        # If training config.yaml exists, load and merge with evaluation parameters
        if has_config:
            logger.info(f"Loading training config file from {config_path}")
            with open(config_path) as file:
                train_config = OmegaConf.load(file)
            
            # Apply eval_overrides from training config
            if train_config.eval_overrides is not None:
                train_config = OmegaConf.merge(
                    train_config, train_config.eval_overrides
                )

            config = OmegaConf.merge(train_config, override_config)
        else:
            config = override_config
    
    # If no checkpoint path is provided
    else:
        if override_config.eval_overrides is not None:
            config = override_config.copy()
            eval_overrides = OmegaConf.to_container(config.eval_overrides, resolve=True)
            
            # Remove duplicate CLI arguments already defined in eval_overrides
            for arg in sys.argv[1:]:
                if not arg.startswith("+"):
                    key = arg.split("=")[0]
                    if key in eval_overrides:
                        del eval_overrides[key]
            
            config.eval_overrides = OmegaConf.create(eval_overrides)
            config = OmegaConf.merge(config, eval_overrides)
        else:
            config = override_config

    # ============================================================================
    # Simulator Detection and Initialization
    # ============================================================================
    
    simulator_type = config.simulator['_target_'].split('.')[-1]
    
    # Initialize IsaacSim
    if simulator_type == 'IsaacSim':
        from omni.isaac.lab.app import AppLauncher
        import argparse
        parser = argparse.ArgumentParser(description="Evaluate an RL agent with RSL-RL.")
        AppLauncher.add_app_launcher_args(parser)
        
        # Parse command line arguments, separating AppLauncher args from Hydra args
        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args
        
        # Configure IsaacSim launch parameters from config
        args_cli.num_envs = config.num_envs
        args_cli.seed = config.seed
        args_cli.env_spacing = config.env.config.env_spacing
        args_cli.output_dir = config.output_dir
        args_cli.headless = config.headless  # Run without GUI if True

        # Launch IsaacSim application
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app
    
    # Initialize IsaacGym
    if simulator_type == 'IsaacGym':
        import isaacgym
        
    from humanoidverse.agents.base_algo.base_algo import BaseAlgo  # noqa: E402
    from humanoidverse.utils.helpers import pre_process_config
    import torch
    from humanoidverse.utils.inference_helpers import export_policy_as_jit, export_policy_as_onnx, export_policy_and_estimator_as_onnx

    # Pre-process configuration (resolve paths, validate settings, etc.)
    pre_process_config(config)
    
    if config.get("device", None):
        device = config.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Create evaluation log directory and save configuration
    eval_log_dir = Path(config.eval_log_dir)
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving eval logs to {eval_log_dir}")
    with open(eval_log_dir / "config.yaml", "w") as file:
        OmegaConf.save(config, file)

    # ============================================================================
    # Environment and Algorithm Instantiation
    # ============================================================================
    
    # Extract checkpoint number from filename for organizing outputs
    # Example: "model_50000.pt" -> "50000"
    ckpt_num = config.checkpoint.split('/')[-1].split('_')[-1].split('.')[0]
    
    # Override to use single environment for evaluation (more stable/interpretable)
    config.num_envs = 1
    
    # Setup output directories for rendered videos/images
    config.env.config.save_rendering_dir = str(checkpoint.parent / "renderings" / f"ckpt_{ckpt_num}")
    config.env.config.ckpt_dir = str(checkpoint.parent)  # For saving motion data if needed
    
    # Instantiate the environment (robot, task, sensors, etc.)
    env = instantiate(config.env, device=device)

    # Start background thread for keyboard interaction (currently disabled in listen_for_keypress)
    # Allows real-time control: task switching, force application, etc.
    key_listener_thread = threading.Thread(target=listen_for_keypress, args=(env,))
    key_listener_thread.daemon = True  # Thread exits when main program exits
    key_listener_thread.start()

    # Instantiate the RL algorithm with loaded policy network
    algo: BaseAlgo = instantiate(config.algo, env=env, device=device, log_dir=None)
    algo.setup()  # Initialize actor-critic networks, optimizers, etc.
    algo.load(config.checkpoint)  # Load trained weights from checkpoint file

    # ============================================================================
    # Export trained policy to optimized formats for deployment on real robots (export to onnx format by default)
    # ============================================================================

    EXPORT_POLICY = False
    EXPORT_ONNX = True

    checkpoint_path = str(checkpoint)
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Determine project root directory for consistent path resolution
    ROBOVERSE_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Create export directory structure: checkpoint_dir/exported/
    exported_policy_path = os.path.join(ROBOVERSE_ROOT_DIR, checkpoint_dir, 'exported')
    os.makedirs(exported_policy_path, exist_ok=True)
    
    # Generate export filenames based on checkpoint name
    exported_policy_name = checkpoint_path.split('/')[-1]  # e.g., "model_50000.pt"
    exported_onnx_name = exported_policy_name.replace('.pt', '.onnx')  # e.g., "model_50000.onnx"

    # Export to TorchScript JIT (optional)
    if EXPORT_POLICY:
        export_policy_as_jit(algo.alg.actor_critic, exported_policy_path, exported_policy_name)
        logger.info('Exported policy as jit script to: ', os.path.join(exported_policy_path, exported_policy_name))
    
    # Export to ONNX
    if EXPORT_ONNX:
        example_obs_dict = algo.get_example_obs()  # Get sample observation for tracing
        export_policy_as_onnx(algo.inference_model, exported_policy_path, exported_onnx_name, example_obs_dict)
        logger.info(f'Exported policy as onnx to: {os.path.join(exported_policy_path, exported_onnx_name)}')

    # ============================================================================
    # Execute the trained policy in the environment and collect performance metrics
    # ============================================================================

    algo.evaluate_policy()


if __name__ == "__main__":
    main()
