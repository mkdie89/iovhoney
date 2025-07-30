import time
import random
import requests
import numpy as np

# RSU集群地址
RSU_API = "http://rsu-cluster:5000"

def get_rsu_status():
    try:
        response = requests.get(f"{RSU_API}/status")
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.ConnectionError:
        return None
    return None

def get_attacker_position():
    try:
        response = requests.get(f"{RSU_API}/status")
        if response.status_code == 200:
            data = response.json()
            return data.get('current_attacker_position', random.randint(0, 11))
    except:
        return random.randint(0, 11)
    return random.randint(0, 11)

def select_target(status, current_position):
    """选择攻击目标：使用攻击图策略"""
    if not status or 'nodes' not in status:
        return random.randint(0, 11)
    
    nodes = status['nodes']
    
    # 基于攻击图策略选择目标
    # 在实际中，这里会使用攻击图算法
    # 简化：偏好高资产且未被频繁攻击的节点
    
    # 计算攻击概率：资产价值越高，被攻击概率越大
    assets = [node['assets'] for node in nodes]
    total_assets = sum(assets)
    
    # 如果总资产为0，随机选择
    if total_assets <= 0:
        return random.randint(0, 11)
    
    # 计算攻击概率（资产占比）
    probabilities = [asset / total_assets for asset in assets]
    
    # 增加未被攻击过的节点概率
    for i in range(len(probabilities)):
        if nodes[i]['attack_count'] == 0:
            probabilities[i] *= 1.5
    
    # 减少当前所在节点的攻击概率
    if current_position < len(probabilities):
        probabilities[current_position] *= 0.3
    
    # 归一化
    total_prob = sum(probabilities)
    probabilities = [p / total_prob for p in probabilities]
    
    return np.random.choice(len(nodes), p=probabilities)

def simulate_attack(source, target):
    try:
        response = requests.get(f"{RSU_API}/attack/{source}/{target}")
        if response.status_code == 200:
            result = response.json()
            if result.get('success', False):
                if result.get('is_honeypot', False):
                    print(f"Attack from RSU-{source} to RSU-{target}: TRAPPED in honeypot!")
                else:
                    print(f"Attack from RSU-{source} to RSU-{target}: Success! Asset loss: {result.get('asset_loss', 0):.2f}")
            else:
                print(f"Attack from RSU-{source} to RSU-{target} failed: {result.get('error', 'Unknown error')}")
    except requests.exceptions.ConnectionError:
        print("Failed to connect to RSU cluster")

def main():
    print("Starting attacker simulation...")
    
    while True:
        # 获取当前攻击者位置
        current_position = get_attacker_position()
        
        # 获取当前RSU状态
        status = get_rsu_status()
        
        # 选择攻击目标
        target = select_target(status, current_position)
        
        # 模拟攻击
        simulate_attack(current_position, target)
        
        # 更新攻击者位置
        try:
            requests.post(f"{RSU_API}/set_attacker_position/{target}")
        except:
            pass
        
        # 随机间隔 (泊松分布，平均2秒)
        time.sleep(max(0.5, random.expovariate(0.5)))

if __name__ == "__main__":
    main()