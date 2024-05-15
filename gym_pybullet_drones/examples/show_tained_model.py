from stable_baselines3 import PPO
from gym_pybullet_drones.envs import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync, str2bool
import time
import numpy as np
from gym_pybullet_drones.utils.Logger import Logger
from stable_baselines3.common.evaluation import evaluate_policy


path = 'results/save-05.14.2024_12.01.53/best_model.zip'
model = PPO.load(path)
test_env = HoverAviary(gui=True,
                       obs=ObservationType('kin'),
                       act=ActionType('vel'),
                       record=False)
test_env_nogui = HoverAviary(obs=ObservationType('kin'), act=ActionType('vel'))
logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                num_drones=1,
                output_folder='results',
                colab=False
                )

mean_reward, std_reward = evaluate_policy(model,
                                          test_env_nogui,
                                          n_eval_episodes=10
                                          )
print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

obs, info = test_env.reset(seed=42, options={})
start = time.time()
for i in range((test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ):
    action, _states = model.predict(obs,
                                    deterministic=True
                                    )
    obs, reward, terminated, truncated, info = test_env.step(action)
    obs2 = obs.squeeze()
    act2 = action.squeeze()
    print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
    logger.log(drone=0,
               timestamp=i / test_env.CTRL_FREQ,
               state=np.hstack([obs2[0:3],
                                np.zeros(4),
                                obs2[3:15],
                                act2
                                ]),
               control=np.zeros(12)
               )

    # test_env.render()
    print(terminated)
    sync(i, start, test_env.CTRL_TIMESTEP)
    if terminated:
        obs = test_env.reset(seed=42, options={})
# test_env.close()

logger.plot()