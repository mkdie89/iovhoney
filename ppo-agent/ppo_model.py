import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class PPONetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPONetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.actor = nn.Linear(128, output_dim)
        self.critic = nn.Linear(128, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value

class PPOAgent:
    def __init__(self, input_dim, output_dim, lr=0.0003, gamma=0.99, clip_epsilon=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PPONetwork(input_dim, output_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.policy(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)
    
    def update_policy(self, states, actions, log_probs_old, returns, advantages):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        log_probs_old = torch.FloatTensor(log_probs_old).to(self.device)
        returns = torch.FloatTensor(returns).unsqueeze(1).to(self.device)
        advantages = torch.FloatTensor(advantages).unsqueeze(1).to(self.device)
        
        # 计算新策略的概率
        action_probs, state_values = self.policy(states)
        dist = torch.distributions.Categorical(action_probs)
        log_probs_new = dist.log_prob(actions)
        
        # PPO损失函数
        ratio = torch.exp(log_probs_new - log_probs_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Critic损失
        critic_loss = F.mse_loss(state_values, returns)
        
        # 总损失
        loss = actor_loss + 0.5 * critic_loss
        
        # 更新策略
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save_model(self, path):
        torch.save(self.policy.state_dict(), path)
    
    def load_model(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))