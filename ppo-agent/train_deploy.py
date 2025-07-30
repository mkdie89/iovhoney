import time
import random
import numpy as np
import requests
import base64
import os
from ppo_model import PPOAgent
from attack_graph import AttackGraphGenerator

# 环境参数
NUM_NODES = 12
MAX_HONEYPOTS = 3  # 最多部署3个蜜罐
STATE_DIM = NUM_NODES * 4 + 5  # 节点特征(攻击次数, 资产值, 蜜罐交互, 蜜罐状态) + 全局特征
ACTION_DIM = NUM_NODES  # 每个节点都可以部署蜜罐

# RSU集群地址
RSU_API = "http://rsu-cluster:5000"

class HoneypotEnv:
    def __init__(self):
        self.num_nodes = NUM_NODES
        self.max_honeypots = MAX_HONEYPOTS
        self.current_deployments = []
        self.attack_graph = AttackGraphGenerator(NUM_NODES)
        
        # 初始化攻击图到RSU集群
        self.init_attack_graph()
    
    def init_attack_graph(self):
        try:
            requests.post(f"{RSU_API}/set_attack_graph")
        except:
            print("Failed to initialize attack graph on RSU cluster")
    
    def get_state(self):
        """获取当前环境状态"""
        try:
            response = requests.get(f"{RSU_API}/status")
            if response.status_code == 200:
                data = response.json()
                nodes = data['nodes']
                attack_history = data.get('attack_history', {})
                honeypot_history = data.get('honeypot_history', {})
                
                # 构建状态向量: 
                # 节点特征: [攻击次数, 资产值, 蜜罐交互次数, 蜜罐状态] * NUM_NODES
                state = []
                for i in range(self.num_nodes):
                    node = next((n for n in nodes if n['id'] == i), None)
                    if node:
                        state.append(node['attack_count'])
                        state.append(node['assets'])
                        state.append(node['honeypot_interactions'])
                        state.append(1 if node['is_honeypot'] else 0)
                    else:
                        state.extend([0, 0, 0, 0])
                
                # 添加攻击图特征
                graph_features = self.attack_graph.get_graph_features()
                state.extend(graph_features.tolist())
                
                return np.array(state, dtype=np.float32)
        except requests.exceptions.ConnectionError:
            pass
        return np.zeros(STATE_DIM, dtype=np.float32)
    
    def deploy_honeypot(self, node_id):
        """部署蜜罐到指定节点"""
        if node_id in self.current_deployments:
            return True  # 已经部署
        
        if len(self.current_deployments) >= self.max_honeypots:
            # 先移除一个蜜罐（选择最少交互的）
            honeypot_interactions = [
                (i, self.get_honeypot_interactions(i)) 
                for i in self.current_deployments
            ]
            remove_id = min(honeypot_interactions, key=lambda x: x[1])[0]
            requests.post(f"{RSU_API}/remove/{remove_id}")
            self.current_deployments.remove(remove_id)
        
        try:
            response = requests.post(f"{RSU_API}/deploy/{node_id}")
            if response.status_code == 200 and response.json().get('success', False):
                self.current_deployments.append(node_id)
                return True
        except requests.exceptions.ConnectionError:
            pass
        return False
    
    def get_honeypot_interactions(self, node_id):
        """获取节点的蜜罐交互次数"""
        try:
            response = requests.get(f"{RSU_API}/status")
            if response.status_code == 200:
                data = response.json()
                for node in data['nodes']:
                    if node['id'] == node_id:
                        return node['honeypot_interactions']
        except:
            pass
        return 0
    
    def get_reward(self, last_state, current_state):
        """计算奖励函数"""
        reward = 0
        
        # 计算资产保护奖励
        asset_reward = 0
        for i in range(0, self.num_nodes * 4, 4):
            asset_current = current_state[i+1]
            asset_last = last_state[i+1] if last_state is not None else asset_current
            # 资产增加或减少幅度小则奖励
            asset_reward += asset_current - asset_last
        
        # 计算蜜罐诱捕奖励
        honeypot_reward = 0
        for i in range(0, self.num_nodes * 4, 4):
            honeypot_interactions = current_state[i+2]
            # 如果该节点部署了蜜罐且有交互，给予奖励
            if current_state[i+3] == 1 and honeypot_interactions > 0:
                honeypot_reward += honeypot_interactions * 0.5
        
        # 蜜罐部署成本惩罚
        deployment_cost = -0.1 * len(self.current_deployments)
        
        # 攻击图复杂性奖励（鼓励发现更多攻击路径）
        graph_complexity = current_state[-5]  # 攻击图边数
        graph_reward = graph_complexity * 0.01
        
        reward = asset_reward + honeypot_reward + deployment_cost + graph_reward
        return reward
    
    def visualize_attack_graph(self):
        """可视化攻击图"""
        return self.attack_graph.visualize()

def train_ppo_agent(episodes=100, steps_per_episode=50):
    env = HoneypotEnv()
    agent = PPOAgent(STATE_DIM, ACTION_DIM, lr=0.0003)
    
    print("Starting PPO training...")
    
    # 创建目录保存模型和可视化
    os.makedirs("results", exist_ok=True)
    
    for episode in range(episodes):
        state = env.get_state()
        last_state = None
        episode_reward = 0
        states, actions, log_probs, rewards = [], [], [], []
        
        for step in range(steps_per_episode):
            # 选择动作
            action, log_prob = agent.get_action(state)
            
            # 执行动作 (部署蜜罐)
            env.deploy_honeypot(action)
            
            # 等待一段时间让攻击发生
            time.sleep(3)
            
            # 获取新状态
            new_state = env.get_state()
            
            # 计算奖励
            reward = env.get_reward(last_state, new_state)
            episode_reward += reward
            
            # 存储经验
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob.item())
            rewards.append(reward)
            
            last_state = state
            state = new_state
        
        # 计算回报和优势
        returns = []
        discounted_reward = 0
        for r in reversed(rewards):
            discounted_reward = r + agent.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        
        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        
        # 更新策略
        loss = agent.update_policy(states, actions, log_probs, returns, returns)
        
        print(f"Episode {episode+1}/{episodes}, Reward: {episode_reward:.2f}, Loss: {loss:.4f}")
        
        # 每10个episode保存模型和攻击图
        if (episode + 1) % 10 == 0:
            agent.save_model(f"results/ppo_honeypot_ep{episode+1}.pth")
            img_data = env.visualize_attack_graph()
            with open(f"results/attack_graph_ep{episode+1}.png", "wb") as f:
                f.write(base64.b64decode(img_data))
    
    # 保存最终模型
    agent.save_model("results/ppo_honeypot_final.pth")
    return agent

def deploy_strategy(agent):
    """使用训练好的代理部署蜜罐策略"""
    env = HoneypotEnv()
    print("Starting honeypot deployment strategy...")
    
    # 创建目录保存可视化
    os.makedirs("deployment_results", exist_ok=True)
    episode_count = 0
    
    while True:
        state = env.get_state()
        
        # 获取动作 (节点部署)
        action, _ = agent.get_action(state)
        
        # 部署蜜罐
        success = env.deploy_honeypot(action)
        if success:
            print(f"Deployed honeypot to node {action}")
        
        # 每5分钟保存一次攻击图
        if episode_count % 10 == 0:
            img_data = env.visualize_attack_graph()
            with open(f"deployment_results/attack_graph_{int(time.time())}.png", "wb") as f:
                f.write(base64.b64decode(img_data))
        
        # 每30秒重新评估部署
        time.sleep(30)
        episode_count += 1

if __name__ == "__main__":
    # 训练或部署
    # 在实际中，我们可能先训练然后部署，但为简化，这里直接使用代理
    try:
        agent = PPOAgent(STATE_DIM, ACTION_DIM)
        agent.load_model("results/ppo_honeypot_final.pth")
        print("Loaded pretrained model")
    except Exception as e:
        print(f"No pretrained model found: {e}, starting training...")
        agent = train_ppo_agent(episodes=30, steps_per_episode=20)
    
    # 开始部署策略
    deploy_strategy(agent)