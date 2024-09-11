import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class HoverAviary(BaseRLAviary):
    """Single agent RL problem: hover at position."""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 # initial_xyzs=np.array([1,1,1]),
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.VEL
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        # self.TARGET_POS = np.array([0,3,4])
        self._reset_target_pos()
        self.EPISODE_LEN_SEC = 30
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act,
                         )

    ### build widely used trained model ######
    def _reset_target_pos(self):
        """Reset target position randomly within a specified range."""
        self.TARGET_POS = np.random.uniform(low=[-3, -3, 0.5], high=[3, 3, 3])

    # def reset(self):
    #     """Override reset to randomize target position each episode."""
    #     self._reset_target_pos()
    #     return super().reset()
    ##########################################

    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)
        # ret = max(0, 2 - np.linalg.norm(self.TARGET_POS-state[0:3])**4)
        # return ret


        distance_to_target = np.linalg.norm(self.TARGET_POS - state[0:3])
        # 基本獎勵，使朝着目標移动
        # base_reward = 1.0 - distance_to_target

        # 调整基础奖励，使其始终为正值
        # exp(-distance_to_target) 来平滑地减少奖励
        base_reward = np.exp(-distance_to_target)

        # 逞罰過大姿態偏移或移動過大
        attitude_penalty = np.abs(state[7]) + np.abs(state[8])  # Roll and pitch angles
        attitude_reward = np.exp(-attitude_penalty)

        # 移动惩罚，使其随速度误差收敛
        movement_penalty = np.linalg.norm(state[3:6])  # 线速度
        movement_reward = np.exp(-movement_penalty)

        # 新增速度奖励项，鼓励智能体在接近目标时减速
        # 将速度奖励和移动惩罚区分开来，速度奖励强调接近零速度
        speed_reward = np.exp(-movement_penalty ** 2)

        # 组合所有奖励和惩罚
        reward = base_reward * attitude_reward * speed_reward

        # 确保奖励值在合理范围内
        reward = min(reward, 1.0)  # 设定奖励的上限值，防止过高

        return reward


    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        if np.linalg.norm(self.TARGET_POS-state[0:3]) < .0001:
            return True
        else:
            return False
        
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        state = self._getDroneStateVector(0)
        if (abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > 2.0 # Truncate when the drone is too far away
             or abs(state[7]) > .4 or abs(state[8]) > .4 # Truncate when the drone is too tilted
        ):
            return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years
