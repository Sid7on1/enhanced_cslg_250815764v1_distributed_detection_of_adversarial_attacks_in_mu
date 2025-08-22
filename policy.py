import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
from policy_config import PolicyConfig
from utils import get_logger, get_device
from models import PolicyNetwork
from metrics import calculate_normality_score, calculate_velocity_threshold

class Policy:
    def __init__(self, config: PolicyConfig):
        self.config = config
        self.logger = get_logger(self.config.log_level)
        self.device = get_device()
        self.policy_network = PolicyNetwork(self.config.input_dim, self.config.output_dim)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.config.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.normality_score = 0.0
        self.velocity_threshold = 0.0

    def train(self, data_loader: DataLoader):
        self.policy_network.train()
        total_loss = 0.0
        for batch in data_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.policy_network(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        self.logger.info(f'Training loss: {total_loss / len(data_loader)}')
        self.normality_score, self.velocity_threshold = calculate_normality_score(self.policy_network, self.config.input_dim, self.config.output_dim)
        self.logger.info(f'Normality score: {self.normality_score}, Velocity threshold: {self.velocity_threshold}')

    def evaluate(self, data_loader: DataLoader):
        self.policy_network.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in data_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.policy_network(inputs)
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()
        self.logger.info(f'Evaluation loss: {total_loss / len(data_loader)}')
        self.normality_score, self.velocity_threshold = calculate_normality_score(self.policy_network, self.config.input_dim, self.config.output_dim)
        self.logger.info(f'Normality score: {self.normality_score}, Velocity threshold: {self.velocity_threshold}')

    def predict(self, inputs: torch.Tensor):
        self.policy_network.eval()
        with torch.no_grad():
            outputs = self.policy_network(inputs)
        return outputs

    def get_normality_score(self):
        return self.normality_score

    def get_velocity_threshold(self):
        return self.velocity_threshold

class PolicyConfig:
    def __init__(self):
        self.input_dim = 10
        self.output_dim = 5
        self.learning_rate = 0.001
        self.log_level = 'INFO'
        self.batch_size = 32
        self.num_epochs = 10

class PolicyDataset(Dataset):
    def __init__(self, data: List[Tuple[torch.Tensor, torch.Tensor]], config: PolicyConfig):
        self.data = data
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        inputs, labels = self.data[index]
        return inputs, labels

def main():
    config = PolicyConfig()
    policy = Policy(config)
    data = [(torch.randn(10), torch.randn(5)) for _ in range(100)]
    data_loader = DataLoader(PolicyDataset(data, config), batch_size=config.batch_size, shuffle=True)
    policy.train(data_loader)
    policy.evaluate(data_loader)
    inputs = torch.randn(10)
    outputs = policy.predict(inputs)
    print(policy.get_normality_score())
    print(policy.get_velocity_threshold())

if __name__ == '__main__':
    main()