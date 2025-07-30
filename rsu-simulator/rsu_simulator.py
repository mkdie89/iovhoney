import random
import time
import json
from flask import Flask, jsonify, request
from prometheus_client import start_http_server, Counter, Gauge
import threading

app = Flask(__name__)

# 监控指标
ATTACK_COUNTER = Counter('rsu_attacks_total', 'Total attacks on RSUs', ['rsu_id'])
DEPLOYMENT_GAUGE = Gauge('honeypot_deployment', 'Honeypot deployment status', ['rsu_id'])
ASSET_GAUGE = Gauge('rsu_assets', 'Asset value of RSU', ['rsu_id'])
ATTACK_GRAPH_GAUGE = Gauge('attack_graph_edges', 'Number of edges in attack graph')

# 模拟RSU集群
class RSUCluster:
    def __init__(self, num_nodes=12):
        self.nodes = []
        self.attack_history = {}
        self.honeypot_history = {}
        self.init_nodes(num_nodes)
        self.attack_graph = None
        self.current_attacker_position = random.randint(0, num_nodes-1)
    
    def set_attack_graph(self, attack_graph):
        self.attack_graph = attack_graph
    
    def init_nodes(self, num_nodes):
        for i in range(num_nodes):
            # 随机分配资产值 (5-15)
            assets = random.randint(5, 15)
            self.nodes.append({
                'id': i,
                'assets': assets,
                'is_honeypot': False,
                'attack_count': 0,
                'honeypot_interactions': 0
            })
            ASSET_GAUGE.labels(rsu_id=i).set(assets)
            DEPLOYMENT_GAUGE.labels(rsu_id=i).set(0)
            self.attack_history[i] = 0
            self.honeypot_history[i] = 0
    
    def deploy_honeypot(self, node_id):
        if 0 <= node_id < len(self.nodes):
            # 确保不超过最大蜜罐数量
            current_honeypots = sum(1 for node in self.nodes if node['is_honeypot'])
            if current_honeypots >= 3 and not self.nodes[node_id]['is_honeypot']:
                # 移除一个随机蜜罐
                honeypot_nodes = [i for i, node in enumerate(self.nodes) if node['is_honeypot']]
                remove_id = random.choice(honeypot_nodes)
                self.remove_honeypot(remove_id)
            
            self.nodes[node_id]['is_honeypot'] = True
            DEPLOYMENT_GAUGE.labels(rsu_id=node_id).set(1)
            return True
        return False
    
    def remove_honeypot(self, node_id):
        if 0 <= node_id < len(self.nodes):
            self.nodes[node_id]['is_honeypot'] = False
            DEPLOYMENT_GAUGE.labels(rsu_id=node_id).set(0)
            return True
        return False
    
    def simulate_attack(self, source_id, target_id):
        if 0 <= target_id < len(self.nodes):
            # 更新攻击图
            if self.attack_graph:
                success = False
                if self.nodes[target_id]['is_honeypot']:
                    success = True
                    self.nodes[target_id]['honeypot_interactions'] += 1
                    self.honeypot_history[target_id] = self.honeypot_history.get(target_id, 0) + 1
                self.attack_graph.update_attack(source_id, target_id, success)
                ATTACK_GRAPH_GAUGE.set(len(self.attack_graph.graph.edges))
            
            self.nodes[target_id]['attack_count'] += 1
            self.attack_history[target_id] = self.attack_history.get(target_id, 0) + 1
            ATTACK_COUNTER.labels(rsu_id=target_id).inc()
            
            # 检查是否是蜜罐
            if self.nodes[target_id]['is_honeypot']:
                return {
                    'success': True,
                    'is_honeypot': True,
                    'message': 'Attacker trapped in honeypot!'
                }
            else:
                # 资产损失与攻击次数成正比
                asset_loss = min(1, self.nodes[target_id]['attack_count'] * 0.1)
                self.nodes[target_id]['assets'] = max(0, self.nodes[target_id]['assets'] - asset_loss)
                ASSET_GAUGE.labels(rsu_id=target_id).set(self.nodes[target_id]['assets'])
                return {
                    'success': True,
                    'is_honeypot': False,
                    'asset_loss': asset_loss
                }
        return {'success': False, 'error': 'Invalid node ID'}

# 创建RSU集群
rsu_cluster = RSUCluster(12)

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({
        'nodes': rsu_cluster.nodes,
        'attack_history': rsu_cluster.attack_history,
        'honeypot_history': rsu_cluster.honeypot_history,
        'current_attacker_position': rsu_cluster.current_attacker_position
    })

@app.route('/deploy/<int:node_id>', methods=['POST'])
def deploy_honeypot(node_id):
    success = rsu_cluster.deploy_honeypot(node_id)
    return jsonify({'success': success, 'node_id': node_id})

@app.route('/remove/<int:node_id>', methods=['POST'])
def remove_honeypot(node_id):
    success = rsu_cluster.remove_honeypot(node_id)
    return jsonify({'success': success, 'node_id': node_id})

@app.route('/attack/<int:source_id>/<int:target_id>', methods=['GET'])
def attack_node(source_id, target_id):
    result = rsu_cluster.simulate_attack(source_id, target_id)
    return jsonify(result)

@app.route('/set_attacker_position/<int:position>', methods=['POST'])
def set_attacker_position(position):
    if 0 <= position < len(rsu_cluster.nodes):
        rsu_cluster.current_attacker_position = position
        return jsonify({'success': True, 'position': position})
    return jsonify({'success': False, 'error': 'Invalid position'})

@app.route('/set_attack_graph', methods=['POST'])
def set_attack_graph():
    rsu_cluster.attack_graph = None  # 实际中这里会接收图数据
    return jsonify({'success': True})

def run_flask():
    app.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    # 启动Prometheus指标服务器
    start_http_server(8000)
    
    # 在单独线程中运行Flask应用
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    
    # 保持主线程运行
    while True:
        time.sleep(1)