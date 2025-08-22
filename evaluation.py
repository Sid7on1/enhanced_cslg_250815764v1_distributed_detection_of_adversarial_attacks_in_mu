import logging
import numpy as np
import pandas as pd
import torch
from torch import nn
from typing import Dict, List, Tuple

# Define constants
VELOCITY_THRESHOLD = 0.5
FLOW_THEORY_THRESHOLD = 0.8

# Define logger
logger = logging.getLogger(__name__)

class AgentEvaluationException(Exception):
    """Base exception class for agent evaluation."""
    pass

class InvalidAgentDataException(AgentEvaluationException):
    """Exception raised when agent data is invalid."""
    pass

class AgentEvaluation:
    """Class for evaluating agent performance."""

    def __init__(self, agent_data: Dict[str, List[float]], config: Dict[str, float]):
        """
        Initialize the AgentEvaluation class.

        Args:
        - agent_data (Dict[str, List[float]]): Agent data, including observations and actions.
        - config (Dict[str, float]): Configuration dictionary, including velocity threshold and flow theory threshold.

        Raises:
        - InvalidAgentDataException: If agent data is invalid.
        """
        self.agent_data = agent_data
        self.config = config
        self.velocity_threshold = config.get("velocity_threshold", VELOCITY_THRESHOLD)
        self.flow_theory_threshold = config.get("flow_theory_threshold", FLOW_THEORY_THRESHOLD)

        # Validate agent data
        if not self._validate_agent_data():
            raise InvalidAgentDataException("Invalid agent data")

    def _validate_agent_data(self) -> bool:
        """
        Validate agent data.

        Returns:
        - bool: True if agent data is valid, False otherwise.
        """
        # Check if agent data is a dictionary
        if not isinstance(self.agent_data, dict):
            return False

        # Check if agent data contains observations and actions
        if "observations" not in self.agent_data or "actions" not in self.agent_data:
            return False

        # Check if observations and actions are lists
        if not isinstance(self.agent_data["observations"], list) or not isinstance(self.agent_data["actions"], list):
            return False

        return True

    def calculate_velocity(self) -> float:
        """
        Calculate the velocity of the agent.

        Returns:
        - float: The velocity of the agent.
        """
        # Calculate the difference between consecutive observations
        observation_diff = np.diff(self.agent_data["observations"])

        # Calculate the velocity
        velocity = np.mean(np.abs(observation_diff))

        return velocity

    def calculate_flow_theory(self) -> float:
        """
        Calculate the flow theory of the agent.

        Returns:
        - float: The flow theory of the agent.
        """
        # Calculate the difference between consecutive actions
        action_diff = np.diff(self.agent_data["actions"])

        # Calculate the flow theory
        flow_theory = np.mean(np.abs(action_diff))

        return flow_theory

    def evaluate_agent(self) -> Dict[str, float]:
        """
        Evaluate the agent's performance.

        Returns:
        - Dict[str, float]: A dictionary containing the evaluation metrics.
        """
        # Calculate the velocity and flow theory
        velocity = self.calculate_velocity()
        flow_theory = self.calculate_flow_theory()

        # Evaluate the agent's performance based on the velocity and flow theory
        evaluation_metrics = {
            "velocity": velocity,
            "flow_theory": flow_theory,
            "velocity_threshold": self.velocity_threshold,
            "flow_theory_threshold": self.flow_theory_threshold,
        }

        # Check if the agent's velocity and flow theory exceed the thresholds
        if velocity > self.velocity_threshold:
            evaluation_metrics["velocity_exceeds_threshold"] = True
        else:
            evaluation_metrics["velocity_exceeds_threshold"] = False

        if flow_theory > self.flow_theory_threshold:
            evaluation_metrics["flow_theory_exceeds_threshold"] = True
        else:
            evaluation_metrics["flow_theory_exceeds_threshold"] = False

        return evaluation_metrics

    def save_evaluation_metrics(self, evaluation_metrics: Dict[str, float]) -> None:
        """
        Save the evaluation metrics to a file.

        Args:
        - evaluation_metrics (Dict[str, float]): A dictionary containing the evaluation metrics.
        """
        # Save the evaluation metrics to a CSV file
        pd.DataFrame([evaluation_metrics]).to_csv("evaluation_metrics.csv", index=False)

class AgentEvaluationConfig:
    """Class for configuring the agent evaluation."""

    def __init__(self, velocity_threshold: float = VELOCITY_THRESHOLD, flow_theory_threshold: float = FLOW_THEORY_THRESHOLD):
        """
        Initialize the AgentEvaluationConfig class.

        Args:
        - velocity_threshold (float): The velocity threshold. Defaults to VELOCITY_THRESHOLD.
        - flow_theory_threshold (float): The flow theory threshold. Defaults to FLOW_THEORY_THRESHOLD.
        """
        self.velocity_threshold = velocity_threshold
        self.flow_theory_threshold = flow_theory_threshold

    def to_dict(self) -> Dict[str, float]:
        """
        Convert the configuration to a dictionary.

        Returns:
        - Dict[str, float]: A dictionary containing the configuration.
        """
        return {
            "velocity_threshold": self.velocity_threshold,
            "flow_theory_threshold": self.flow_theory_threshold,
        }

def main() -> None:
    # Create a sample agent data
    agent_data = {
        "observations": [1.0, 2.0, 3.0, 4.0, 5.0],
        "actions": [0.5, 1.0, 1.5, 2.0, 2.5],
    }

    # Create a configuration
    config = AgentEvaluationConfig()

    # Create an agent evaluation instance
    agent_evaluation = AgentEvaluation(agent_data, config.to_dict())

    # Evaluate the agent
    evaluation_metrics = agent_evaluation.evaluate_agent()

    # Save the evaluation metrics
    agent_evaluation.save_evaluation_metrics(evaluation_metrics)

    # Log the evaluation metrics
    logger.info("Evaluation metrics: %s", evaluation_metrics)

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Run the main function
    main()