import logging
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants and configuration
CONFIG = {
    'model': {
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2
    },
    'training': {
        'batch_size': 32,
        'num_epochs': 10,
        'learning_rate': 0.001
    },
    'data': {
        'path': 'data.csv',
        'num_samples': 1000
    }
}

class AgentDataset(Dataset):
    def __init__(self, data_path: str, num_samples: int):
        self.data = pd.read_csv(data_path)
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        data = self.data.iloc[idx]
        return {
            'state': torch.tensor(data['state'], dtype=torch.float32),
            'action': torch.tensor(data['action'], dtype=torch.float32),
            'reward': torch.tensor(data['reward'], dtype=torch.float32)
        }

class AgentModel(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, dropout: float):
        super(AgentModel, self).__init__()
        self.fc1 = nn.Linear(4, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        for _ in range(self.num_layers - 1):
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
        x = self.fc3(x)
        return x

class AgentTrainer:
    def __init__(self, model: AgentModel, device: torch.device):
        self.model = model
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=CONFIG['training']['learning_rate'])

    def train(self, data_loader: DataLoader):
        self.model.train()
        for epoch in range(CONFIG['training']['num_epochs']):
            for batch in data_loader:
                state = batch['state'].to(self.device)
                action = batch['action'].to(self.device)
                reward = batch['reward'].to(self.device)
                self.optimizer.zero_grad()
                output = self.model(state)
                loss = nn.MSELoss()(output, reward)
                loss.backward()
                self.optimizer.step()
            logger.info(f'Epoch {epoch+1}, Loss: {loss.item()}')

    def evaluate(self, data_loader: DataLoader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in data_loader:
                state = batch['state'].to(self.device)
                action = batch['action'].to(self.device)
                reward = batch['reward'].to(self.device)
                output = self.model(state)
                loss = nn.MSELoss()(output, reward)
                total_loss += loss.item()
        return total_loss / len(data_loader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    dataset = AgentDataset(CONFIG['data']['path'], CONFIG['data']['num_samples'])
    data_loader = DataLoader(dataset, batch_size=CONFIG['training']['batch_size'], shuffle=True)

    model = AgentModel(CONFIG['model']['hidden_size'], CONFIG['model']['num_layers'], CONFIG['model']['dropout'])
    model.to(device)

    trainer = AgentTrainer(model, device)
    trainer.train(data_loader)
    trainer.evaluate(data_loader)

if __name__ == '__main__':
    main()