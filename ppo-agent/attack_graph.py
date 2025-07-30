import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
import base64

class AttackGraphGenerator:
    def __init__(self, num_nodes=12):
        self.num_nodes = num_nodes
        self.graph = nx.DiGraph()
        self.init_graph()
    
    def init_graph(self):
        # 添加节点
        for i in range(self.num_nodes):
            self.graph.add_node(i, type="RSU", value=0, attacks=0)
        
        # 创建随机拓扑结构
        # 确保所有节点连接（车联网特性）
        for i in range(self.num_nodes):
            # 每个节点连接2-4个其他节点
            connections = np.random.choice(
                [j for j in range(self.num_nodes) if j != i],
                size=np.random.randint(2, 5),
                replace=False
            )
            for conn in connections:
                # 随机权重表示连接强度
                weight = np.random.uniform(0.5, 1.0)
                self.graph.add_edge(i, conn, weight=weight)
    
    def update_attack(self, source, target, success):
        """更新攻击图"""
        # 更新节点攻击次数
        self.graph.nodes[target]["attacks"] += 1
        
        # 更新边（攻击路径）
        if self.graph.has_edge(source, target):
            # 增加攻击路径权重
            self.graph[source][target]["weight"] += 0.1
        else:
            # 添加新的攻击路径
            self.graph.add_edge(source, target, weight=0.5)
            
        # 如果攻击成功，增加节点价值（攻击者更可能返回）
        if success:
            self.graph.nodes[target]["value"] += 0.2
    
    def get_attack_probs(self, current_position=None):
        """获取每个节点的攻击概率"""
        if current_position is None:
            current_position = np.random.choice(self.num_nodes)
        
        # 基于图结构计算攻击概率
        probs = np.zeros(self.num_nodes)
        
        # 1. 节点价值因素
        values = np.array([self.graph.nodes[i]["value"] for i in range(self.num_nodes)])
        value_factor = values / (values.sum() + 1e-5)
        
        # 2. 攻击次数因素（攻击者偏好未攻击过的节点）
        attacks = np.array([self.graph.nodes[i]["attacks"] for i in range(self.num_nodes)])
        attack_factor = 1.0 / (attacks + 1)
        attack_factor = attack_factor / (attack_factor.sum() + 1e-5)
        
        # 3. 拓扑接近度因素
        if current_position is not None:
            distances = np.zeros(self.num_nodes)
            for i in range(self.num_nodes):
                try:
                    # 计算最短路径长度
                    path_len = nx.shortest_path_length(self.graph, source=current_position, target=i)
                    distances[i] = path_len
                except:
                    distances[i] = 10  # 不可达节点
            distance_factor = 1.0 / (distances + 0.1)
            distance_factor = distance_factor / (distance_factor.sum() + 1e-5)
        else:
            distance_factor = np.ones(self.num_nodes) / self.num_nodes
        
        # 组合因素
        probs = 0.4 * value_factor + 0.3 * attack_factor + 0.3 * distance_factor
        probs = probs / probs.sum()
        
        return probs
    
    def get_next_target(self, current_position=None):
        """获取下一个攻击目标"""
        probs = self.get_attack_probs(current_position)
        return np.random.choice(self.num_nodes, p=probs)
    
    def visualize(self):
        """生成攻击图可视化"""
        plt.figure(figsize=(10, 8))
        
        # 节点大小基于攻击次数
        node_size = [1000 + 500 * self.graph.nodes[i]["attacks"] for i in self.graph.nodes]
        
        # 节点颜色基于价值
        node_color = [self.graph.nodes[i]["value"] for i in self.graph.nodes]
        
        # 边宽度基于攻击路径权重
        edge_width = [2 * self.graph[u][v]["weight"] for u, v in self.graph.edges]
        
        # 绘制图形
        pos = nx.spring_layout(self.graph, seed=42)
        nx.draw_networkx_nodes(self.graph, pos, node_size=node_size, node_color=node_color, 
                              cmap=plt.cm.Reds, alpha=0.8)
        nx.draw_networkx_edges(self.graph, pos, width=edge_width, alpha=0.5, edge_color="gray")
        nx.draw_networkx_labels(self.graph, pos, font_size=10)
        
        # 添加边权重标签
        edge_labels = {(u, v): f"{self.graph[u][v]['weight']:.1f}" for u, v in self.graph.edges}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title("车联网攻击图")
        plt.axis("off")
        
        # 转换为Base64编码图像
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        return img_base64
    
    def get_graph_features(self):
        """获取图特征用于PPO状态"""
        # 节点特征: [攻击次数, 价值]
        node_features = []
        for i in range(self.num_nodes):
            node_features.append(self.graph.nodes[i]["attacks"])
            node_features.append(self.graph.nodes[i]["value"])
        
        # 全局特征: 平均攻击次数, 最大价值等
        attacks = [self.graph.nodes[i]["attacks"] for i in range(self.num_nodes)]
        values = [self.graph.nodes[i]["value"] for i in range(self.num_nodes)]
        
        global_features = [
            np.mean(attacks),
            np.max(attacks),
            np.mean(values),
            np.max(values),
            self.graph.number_of_edges()
        ]
        
        return np.array(node_features + global_features, dtype=np.float32)