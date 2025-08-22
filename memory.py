import logging
import numpy as np
import torch
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
from collections import deque
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Memory:
    def __init__(self, config: Config):
        self.config = config
        self.memory = deque(maxlen=self.config.memory_size)
        self.experience = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "dones": []
        }

    def add_experience(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        self.experience["states"].append(state)
        self.experience["actions"].append(action)
        self.experience["rewards"].append(reward)
        self.experience["next_states"].append(next_state)
        self.experience["dones"].append(done)

    def sample_experience(self) -> Dict[str, np.ndarray]:
        if len(self.experience["states"]) < self.config.batch_size:
            return None
        indices = np.random.choice(len(self.experience["states"]), size=self.config.batch_size, replace=False)
        states = np.array([self.experience["states"][i] for i in indices])
        actions = np.array([self.experience["actions"][i] for i in indices])
        rewards = np.array([self.experience["rewards"][i] for i in indices])
        next_states = np.array([self.experience["next_states"][i] for i in indices])
        dones = np.array([self.experience["dones"][i] for i in indices])
        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones
        }

    def get_memory_size(self) -> int:
        return len(self.experience["states"])

    def clear_memory(self):
        self.memory.clear()
        self.experience = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "dones": []
        }

class ExperienceReplayBuffer(Memory):
    def __init__(self, config: Config):
        super().__init__(config)
        self.buffer = deque(maxlen=self.config.memory_size)

    def add_experience(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample_experience(self) -> List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]:
        if len(self.buffer) < self.config.batch_size:
            return []
        indices = np.random.choice(len(self.buffer), size=self.config.batch_size, replace=False)
        return [self.buffer[i] for i in indices]

class PrioritizedExperienceReplayBuffer(Memory):
    def __init__(self, config: Config):
        super().__init__(config)
        self.buffer = deque(maxlen=self.config.memory_size)
        self.priorities = deque(maxlen=self.config.memory_size)

    def add_experience(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(1.0)

    def sample_experience(self) -> List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]:
        if len(self.buffer) < self.config.batch_size:
            return []
        indices = np.random.choice(len(self.buffer), size=self.config.batch_size, replace=False)
        return [self.buffer[i] for i in indices]

class ExperienceReplayBufferWithPriorities(Memory):
    def __init__(self, config: Config):
        super().__init__(config)
        self.buffer = deque(maxlen=self.config.memory_size)
        self.priorities = deque(maxlen=self.config.memory_size)

    def add_experience(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(1.0)

    def sample_experience(self) -> List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]:
        if len(self.buffer) < self.config.batch_size:
            return []
        indices = np.random.choice(len(self.buffer), size=self.config.batch_size, replace=False, p=self.get_priorities())
        return [self.buffer[i] for i in indices]

    def get_priorities(self) -> np.ndarray:
        return np.array(self.priorities) / np.sum(self.priorities)

    def update_priority(self, index: int, priority: float):
        self.priorities[index] = priority

class ExperienceReplayBufferWithVelocityThreshold(Memory):
    def __init__(self, config: Config):
        super().__init__(config)
        self.buffer = deque(maxlen=self.config.memory_size)
        self.velocity_threshold = self.config.velocity_threshold

    def add_experience(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        velocity = np.linalg.norm(state - next_state)
        if velocity > self.velocity_threshold:
            self.buffer.append((state, action, reward, next_state, done))

    def sample_experience(self) -> List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]:
        if len(self.buffer) < self.config.batch_size:
            return []
        indices = np.random.choice(len(self.buffer), size=self.config.batch_size, replace=False)
        return [self.buffer[i] for i in indices]

class ExperienceReplayBufferWithFlowTheory(Memory):
    def __init__(self, config: Config):
        super().__init__(config)
        self.buffer = deque(maxlen=self.config.memory_size)
        self.flow_threshold = self.config.flow_threshold

    def add_experience(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        flow = np.linalg.norm(np.cross(state, next_state))
        if flow > self.flow_threshold:
            self.buffer.append((state, action, reward, next_state, done))

    def sample_experience(self) -> List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]:
        if len(self.buffer) < self.config.batch_size:
            return []
        indices = np.random.choice(len(self.buffer), size=self.config.batch_size, replace=False)
        return [self.buffer[i] for i in indices]