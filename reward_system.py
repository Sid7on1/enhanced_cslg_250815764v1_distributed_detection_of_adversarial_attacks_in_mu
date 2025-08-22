import logging
import numpy as np
from typing import Dict, List, Tuple
from reward_system.config import Config
from reward_system.exceptions import RewardSystemError
from reward_system.models import RewardModel
from reward_system.utils import calculate_velocity, calculate_flow_theory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RewardSystem:
    """
    Reward calculation and shaping system.

    This class is responsible for calculating and shaping rewards based on the agent's actions and observations.
    """

    def __init__(self, config: Config):
        """
        Initialize the reward system.

        Args:
            config (Config): Configuration object containing reward system settings.
        """
        self.config = config
        self.reward_model = RewardModel(config)

    def calculate_reward(self, observation: np.ndarray, action: np.ndarray) -> float:
        """
        Calculate the reward based on the observation and action.

        Args:
            observation (np.ndarray): Observation from the environment.
            action (np.ndarray): Action taken by the agent.

        Returns:
            float: Calculated reward.
        """
        try:
            # Calculate velocity
            velocity = calculate_velocity(observation, action)

            # Calculate flow theory
            flow_theory = calculate_flow_theory(observation, action)

            # Calculate reward using the reward model
            reward = self.reward_model.calculate_reward(velocity, flow_theory)

            return reward

        except RewardSystemError as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0

    def shape_reward(self, reward: float) -> float:
        """
        Shape the reward based on the reward system settings.

        Args:
            reward (float): Reward to be shaped.

        Returns:
            float: Shaped reward.
        """
        try:
            # Apply reward shaping based on the reward system settings
            shaped_reward = self.config.apply_reward_shaping(reward)

            return shaped_reward

        except RewardSystemError as e:
            logger.error(f"Error shaping reward: {e}")
            return 0.0

class RewardModel:
    """
    Reward model used for calculating rewards.

    This class is responsible for calculating rewards based on the velocity and flow theory.
    """

    def __init__(self, config: Config):
        """
        Initialize the reward model.

        Args:
            config (Config): Configuration object containing reward model settings.
        """
        self.config = config

    def calculate_reward(self, velocity: float, flow_theory: float) -> float:
        """
        Calculate the reward based on the velocity and flow theory.

        Args:
            velocity (float): Velocity calculated from the observation and action.
            flow_theory (float): Flow theory calculated from the observation and action.

        Returns:
            float: Calculated reward.
        """
        try:
            # Calculate reward using the reward model formula
            reward = self.config.calculate_reward_formula(velocity, flow_theory)

            return reward

        except RewardSystemError as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0

class Config:
    """
    Configuration object for the reward system.

    This class is responsible for storing and providing reward system settings.
    """

    def __init__(self):
        """
        Initialize the configuration object.
        """
        self.reward_shaping_enabled = False
        self.reward_shaping_formula = "linear"

    def apply_reward_shaping(self, reward: float) -> float:
        """
        Apply reward shaping based on the reward system settings.

        Args:
            reward (float): Reward to be shaped.

        Returns:
            float: Shaped reward.
        """
        try:
            # Apply reward shaping based on the reward system settings
            if self.reward_shaping_enabled:
                if self.reward_shaping_formula == "linear":
                    shaped_reward = reward * 2
                elif self.reward_shaping_formula == "exponential":
                    shaped_reward = np.exp(reward)
                else:
                    raise RewardSystemError("Invalid reward shaping formula")

                return shaped_reward

            return reward

        except RewardSystemError as e:
            logger.error(f"Error applying reward shaping: {e}")
            return 0.0

    def calculate_reward_formula(self, velocity: float, flow_theory: float) -> float:
        """
        Calculate the reward using the reward model formula.

        Args:
            velocity (float): Velocity calculated from the observation and action.
            flow_theory (float): Flow theory calculated from the observation and action.

        Returns:
            float: Calculated reward.
        """
        try:
            # Calculate reward using the reward model formula
            reward = velocity + flow_theory

            return reward

        except RewardSystemError as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0

class RewardSystemError(Exception):
    """
    Custom exception for reward system errors.
    """

    pass

def calculate_velocity(observation: np.ndarray, action: np.ndarray) -> float:
    """
    Calculate velocity based on the observation and action.

    Args:
        observation (np.ndarray): Observation from the environment.
        action (np.ndarray): Action taken by the agent.

    Returns:
        float: Calculated velocity.
    """
    try:
        # Calculate velocity using the velocity formula
        velocity = np.linalg.norm(action)

        return velocity

    except RewardSystemError as e:
        logger.error(f"Error calculating velocity: {e}")
        return 0.0

def calculate_flow_theory(observation: np.ndarray, action: np.ndarray) -> float:
    """
    Calculate flow theory based on the observation and action.

    Args:
        observation (np.ndarray): Observation from the environment.
        action (np.ndarray): Action taken by the agent.

    Returns:
        float: Calculated flow theory.
    """
    try:
        # Calculate flow theory using the flow theory formula
        flow_theory = np.dot(observation, action)

        return flow_theory

    except RewardSystemError as e:
        logger.error(f"Error calculating flow theory: {e}")
        return 0.0