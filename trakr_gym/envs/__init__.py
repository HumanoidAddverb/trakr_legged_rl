from trakr_gym import TRAKR_GYM_ROOT_DIR, TRAKR_GYM_ENVS_DIR
from trakr_gym.utils.task_registry import task_registry

from .base.legged_robot import LeggedRobot
from .base.legged_robot_config import RobotCfg, RobotCfgPPO

from .vanilla_rl.trakr_config import VanillaRLConfig, VanillaRLConfigPPO


task_registry.register("vanilla_rl", LeggedRobot, VanillaRLConfig(), VanillaRLConfigPPO() )
