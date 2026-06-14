import argparse
import os
import time
import csv
import json
import networkx as nx
import numpy as np
import warnings
import pandas as pd
from collections import defaultdict, Counter
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import logging
import glob
import joblib

# 配置日志
import os
import logging
from logging.handlers import RotatingFileHandler

# 创建日志目录
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# 配置根日志
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# 控制台处理器 - 只输出INFO及以上
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# 文件处理器 - 输出DEBUG及以上
file_handler = RotatingFileHandler(
    os.path.join(log_dir, 'debug.log'),
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# 清除现有处理器并添加新处理器
if logger.hasHandlers():
    logger.handlers = []
logger.addHandler(console_handler)
logger.addHandler(file_handler)
logger = logging.getLogger(__name__)

class SuspiciousAccountDetector:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.description = '区块链可疑账户检测系统'
        parser.add_argument('-x', '--txs_file', help='交易数据CSV文件路径', dest='txs_file', type=str, required=True)
        parser.add_argument('-n', '--nodes_file', help='节点列表CSV文件路径', dest='nodes_file', type=str, required=True)
        parser.add_argument('-o', '--output', help='输出结果文件夹', dest='out_dir', type=str, required=True)
        # 移除重复的-t参数定义
        parser.add_argument('-t', '--threshold', help='洗钱判定阈值', dest='threshold', type=float, default=0.37)
        parser.add_argument('--rule_weight', help='规则检测权重', type=float, default=0.5)
        parser.add_argument('--model_weight', help='模型检测权重', type=float, default=0.5)
        self.args = parser.parse_args()
        # 定义要加载的模型路径
        # 定义模型训练时使用的特征集
        self.original_features = ['is_same', 'out_tx_count', 'total_out', 'avg_out_amount', 'std_out_amount', 'max_out_amount', 'min_out_amount', 'out_amount_variation', 'tx_frequency', 'in_tx_count', 'total_in', 'avg_in_amount', 'std_in_amount', 'max_in_amount', 'min_in_amount', 'in_amount_variation', 'time_diff_variation']
        
        # 定义模型目录和获取最新模型路径
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Real-CATS-master', 'models')
        self.model_paths = {
            'lightgbm': self.get_latest_model_path(model_dir, 'lightgbm'),
            'random_forest': self.get_latest_model_path(model_dir, 'random_forest'),
            'xgboost': self.get_latest_model_path(model_dir, 'xgboost')
        }
        # 降低规则检测阈值
        self.threshold = self.args.threshold  # 使用命令行参数或默认值0.37
        
        # 确保输出目录存在
        if not os.path.exists(self.args.out_dir):
            os.makedirs(self.args.out_dir)
            
        # 归一化权重
        total_weight = self.args.rule_weight + self.args.model_weight
        self.rule_weight = self.args.rule_weight / total_weight
        self.model_weight = self.args.model_weight / total_weight
            
        # 加载预训练的洗钱识别模型
        self.load_model()
        
        # 加载scaler
        self.scaler_path = self.get_latest_model_path(model_dir, 'scaler')
        self.scaler = None
        if self.scaler_path and os.path.exists(self.scaler_path):
            with open(self.scaler_path, 'rb') as f:
                self.scaler = joblib.load(f)
            logger.info(f"成功加载特征标准化器: {self.scaler_path}")
        else:
            logger.warning("未找到特征标准化器scaler文件")
        
    def get_latest_model_path(self, model_dir, model_type):
        pattern = os.path.join(model_dir, f"{model_type}_*.pkl")
        files = glob.glob(pattern)
        if not files:
            return None
        # 按文件修改时间排序，取最新的
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return files[0]

    def load_model(self):
        """加载预训练的洗钱识别模型"""
        self.models = {}
        for name, path in self.model_paths.items():
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        self.models[name] = joblib.load(f)
                    # 配置GPU设备
                    if name == 'xgboost':
                        self.models[name].set_params(device='cuda')
                    elif name == 'lightgbm':
                        self.models[name].set_params(device='gpu')
                    logger.info(f"成功加载洗钱识别模型: {path} (已启用GPU加速)")
                except Exception as e:
                    logger.error(f"加载模型 {name} 失败: {e}")
            else:
                logger.warning(f"模型文件不存在: {path}")
        
        if not self.models:
            logger.warning("没有加载任何模型，将仅使用规则检测")
            self.models = None
    
    def load_graph(self, txs_file):
        """从交易数据加载有向图"""
        g = nx.DiGraph()
        with open(txs_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                tx = {header[i]: row[i] for i in range(len(header))}
                sender = tx.get('from')
                receiver = tx.get('to')
                amount = float(tx.get('value', 0))
                
                # 验证发送者和接收者不为空
                if not sender or not receiver:
                    logger.warning(f"跳过缺少发送者或接收者的交易: {tx}")
                    continue
                
                # 添加节点
                if sender not in g:
                    g.add_node(sender, transactions=[])
                if receiver not in g:
                    g.add_node(receiver, transactions=[])
                
                # 添加边，包含交易金额和时间戳
                if g.has_edge(sender, receiver):
                    g[sender][receiver]['weight'] += amount
                    g[sender][receiver]['tx_count'] += 1
                else:
                    g.add_edge(sender, receiver, weight=amount, tx_count=1)
                
                # 记录节点的交易
                g.nodes[sender]['transactions'].append({
                    'type': 'out',
                    'to': receiver,
                    'amount': amount,
                    'timestamp': tx.get('timestamp', 0)
                })
                g.nodes[receiver]['transactions'].append({
                    'type': 'in',
                    'from': sender,
                    'amount': amount,
                    'timestamp': tx.get('timestamp', 0)
                })
        
        logger.info(f"图加载完成，节点数: {g.number_of_nodes()}, 边数: {g.number_of_edges()}")
        return g
    
    def extract_features(self, g):
        """从图中提取节点特征"""
        logger.info("开始提取节点特征...")
        features = {}
        total_nodes = g.number_of_nodes()
        logger.info(f"开始处理 {total_nodes} 个节点的特征提取")
        
        # 预计算图级网络结构特征
        undirected_g = g.to_undirected()
        clustering_coeffs = nx.clustering(undirected_g) if total_nodes > 1 else {}
        pagerank_scores = nx.pagerank(g) if total_nodes > 1 else {}
        
        for idx, node in enumerate(g.nodes()):
            if idx % 5000 == 0:
                logger.info(f"已处理 {idx}/{total_nodes} 个节点 ({idx/total_nodes*100:.2f}%)")
            # 初始化所有特征为0
            node_features = {f: 0 for f in self.original_features}
            
            # 1. 基本交易特征
            # 检查是否存在交易双方相同的情况
            transactions = g.nodes[node].get('transactions', [])
            is_same = 1 if any(tx.get('from') == tx.get('to') for tx in transactions) else 0
            node_features['is_same'] = is_same
            
            in_degree = g.in_degree(node)
            out_degree = g.out_degree(node)
            total_in = sum(g[src][node]['weight'] for src in g.predecessors(node))
            total_out = sum(g[node][dst]['weight'] for dst in g.successors(node))
            
            node_features['in_degree'] = in_degree
            node_features['out_degree'] = out_degree
            node_features['degree_ratio'] = out_degree / (in_degree + 1)
            node_features['total_in'] = total_in
            node_features['total_out'] = total_out
            node_features['balance'] = total_out - total_in
            logger.debug(f"节点 {node} 基本交易特征提取完成")
            
            # 2. 交易模式特征
            transactions = g.nodes[node].get('transactions', [])
            in_txs = [tx for tx in transactions if tx['type'] == 'in']
            out_txs = [tx for tx in transactions if tx['type'] == 'out']
            
            node_features['tx_count'] = len(transactions)
            node_features['in_tx_count'] = len(in_txs)
            node_features['out_tx_count'] = len(out_txs)
            
            if out_txs:
                out_amounts = [tx['amount'] for tx in out_txs]
                node_features['avg_out_amount'] = sum(out_amounts) / len(out_amounts)
                node_features['std_out_amount'] = np.std(out_amounts) if len(out_amounts) > 1 else 0
                node_features['max_out_amount'] = max(out_amounts)
                node_features['min_out_amount'] = min(out_amounts)
                node_features['out_amount_variation'] = node_features['std_out_amount'] / (node_features['avg_out_amount'] + 1)
                # 计算账户余额
                node_features['balance'] = node_features.get('total_in', 0) - node_features.get('total_out', 0)
                
                # 检查是否有整额交易
                round_amounts = sum(1 for a in out_amounts if a % 1000000000000000000 == 0)  # 检查是否为整ETH (1 ETH = 1e18 wei)
                node_features['round_amount_ratio'] = round_amounts / len(out_amounts)
            logger.debug(f"节点 {node} 交易模式特征提取完成")
            # 计算度比率 (出度/入度)
            in_degree = node_features.get('in_degree', 0)
            out_degree = node_features.get('out_degree', 0)
            if in_degree == 0:
                node_features['degree_ratio'] = 1000.0 if out_degree > 0 else 0  # 使用大数值替代无穷大避免数值错误
            else:
                node_features['degree_ratio'] = out_degree / in_degree
            
            if in_txs:
                in_amounts = [tx['amount'] for tx in in_txs]
                node_features['avg_in_amount'] = sum(in_amounts) / len(in_amounts)
                node_features['std_in_amount'] = np.std(in_amounts) if len(in_amounts) > 1 else 0
                node_features['max_in_amount'] = max(in_amounts)
                node_features['min_in_amount'] = min(in_amounts)
                node_features['in_amount_variation'] = node_features['std_in_amount'] / (node_features['avg_in_amount'] + 1)
            
            # 3. 网络结构特征
            if total_nodes > 1:
                try:
                    node_features['clustering'] = clustering_coeffs.get(node, 0)
                    node_features['pagerank'] = pagerank_scores.get(node, 0)
                except:
                    node_features['clustering'] = 0
                    node_features['pagerank'] = 0
            logger.debug(f"节点 {node} 网络结构特征提取完成")
            
            # 4. 时间特征
            if transactions:
                # 将时间戳转换为浮点数
                timestamps = sorted([float(tx['timestamp']) for tx in transactions])
                if len(timestamps) > 1:
                    time_diffs = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
                    node_features['avg_time_diff'] = sum(time_diffs) / len(time_diffs)
                    node_features['std_time_diff'] = np.std(time_diffs) if len(time_diffs) > 1 else 0
                    node_features['time_diff_variation'] = node_features['std_time_diff'] / (node_features['avg_time_diff'] + 1)
                
                # 计算交易频率
                if timestamps:
                    time_span = max(timestamps) - min(timestamps)
                    if time_span > 0:
                        # 避免时间跨度为0时的除零错误
                        if time_span == 0:
                            node_features['tx_frequency'] = len(timestamps)  # 所有交易在同一天
                        else:
                            node_features['tx_frequency'] = len(timestamps) / (time_span / 86400)  # 交易/天
            logger.debug(f"节点 {node} 时间特征提取完成")

            # 5. 二阶交易特征 (from_from和to_from特征)
            # 收集一级关联节点 (去重以减少重复计算)
            from_nodes = list(set([tx['from'] for tx in transactions if tx['type'] == 'in']))
            to_nodes = list(set([tx['to'] for tx in transactions if tx['type'] == 'out']))
            node_count = len(features)

            # 进度日志 (每处理500个节点记录一次)
            if idx % 500 == 0:
                if total_nodes == 0:
                    logger.warning("节点数量为零，无法计算进度百分比")
                    progress_log = f"正在提取二阶特征: {idx}/0 个节点"
                else:
                    progress_log = f"正在提取二阶特征: {idx}/{total_nodes} 个节点 ({idx/total_nodes*100:.2f}%)"
                logger.info(progress_log)

            # from_from特征 (节点的入边来源节点的交易特征)
            from_from_txs = []
            for from_node in from_nodes:
                if from_node in g.nodes:
                    from_from_txs.extend(g.nodes[from_node].get('transactions', []))
            if from_from_txs:
                from_from_values = np.array([tx['amount'] for tx in from_from_txs])
                node_features['from_from_transaction_count'] = len(from_from_txs)
                node_features['from_from_total_value'] = from_from_values.sum()
                node_features['from_from_avg_value'] = from_from_values.mean()
                node_features['from_from_std_value'] = from_from_values.std() if len(from_from_values) > 1 else 0
                node_features['from_from_max_value'] = from_from_values.max()
                node_features['from_from_min_value'] = from_from_values.min()
                node_features['from_from_avg_log_value'] = np.log1p(from_from_values).mean()
                # 交易小时模式
                timestamps = np.array([float(tx['timestamp']) for tx in from_from_txs])
                hours = pd.to_datetime(timestamps, unit='s').hour
                node_features['from_from_transaction_hour_mode'] = pd.Series(hours).mode()[0] if len(hours) > 0 else 0
            else:
                node_features.update({f:0 for f in ['from_from_transaction_count', 'from_from_total_value', 'from_from_avg_value', 'from_from_std_value', 'from_from_max_value', 'from_from_min_value', 'from_from_avg_log_value', 'from_from_transaction_hour_mode']})

            # to_from特征 (节点的出边目标节点的入交易特征)
            to_from_txs = []
            for to_node in to_nodes:
                if to_node in g.nodes:
                    to_from_txs.extend([tx for tx in g.nodes[to_node].get('transactions', []) if tx['type'] == 'in'])
            if to_from_txs:
                to_from_values = np.array([tx['amount'] for tx in to_from_txs])
                node_features['to_from_transaction_count'] = len(to_from_txs)
                node_features['to_from_total_value'] = to_from_values.sum()
                node_features['to_from_avg_value'] = to_from_values.mean()
                node_features['to_from_std_value'] = to_from_values.std() if len(to_from_values) > 1 else 0
                node_features['to_from_max_value'] = to_from_values.max()
                node_features['to_from_min_value'] = to_from_values.min()
                node_features['to_from_avg_log_value'] = np.log1p(to_from_values).mean()
                # 交易小时模式
                timestamps = np.array([float(tx['timestamp']) for tx in to_from_txs])
                hours = pd.to_datetime(timestamps, unit='s').hour
                node_features['to_from_transaction_hour_mode'] = pd.Series(hours).mode()[0] if len(hours) > 0 else 0
            else:
                node_features.update({f:0 for f in ['to_from_transaction_count', 'to_from_total_value', 'to_from_avg_value', 'to_from_std_value', 'to_from_max_value', 'to_from_min_value', 'to_from_avg_log_value', 'to_from_transaction_hour_mode']})

            logger.debug(f"节点 {node} 二阶交易特征提取完成")
            
            features[node] = node_features
        
        logger.info(f"特征提取完成，共提取 {len(features)} 个节点的特征")
        return features
    
    def rule_based_detection(self, features):
        """基于规则的洗钱检测方法"""
        results = {}
        total_nodes = len(features)
        logger.info(f"开始基于规则的洗钱检测，共处理 {total_nodes} 个节点")
        
        # 从配置文件加载规则和黑名单
        import json
        import os
        
        # 加载规则配置
        rules_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rules_config.json')
        with open(rules_config_path, 'r', encoding='utf-8') as f:
            rules_config = json.load(f)
        
        # 加载黑名单
        blacklist_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'blacklist.json')
        with open(blacklist_path, 'r') as f:
            blacklist = json.load(f)
        suspicious_addresses = set(blacklist.get('suspicious_addresses', []))
        
        for idx, (node, node_features) in enumerate(features.items()):
            if idx % 5000 == 0:
                logger.info(f"规则检测进度: {idx}/{total_nodes} 个节点 ({idx/total_nodes*100:.2f}%)")
            
            # 动态构建规则
            rules = []
            # 优先检查黑名单
            if node in suspicious_addresses:
                rules.append((True, 5.0))  # 黑名单地址直接赋予最高权重
            
            for rule in rules_config:
                # 解析规则条件
                condition = eval(rule['condition'], {"node_features": node_features, "self": self, "params": rule.get('params', {})})
                rules.append((condition, rule['weight']))
            
            # 计算加权可疑度分数
            matched_score = sum(weight for cond, weight in rules if cond)
            total_weight = sum(weight for cond, weight in rules)  # 使用所有规则权重总和作为分母
            score = matched_score / total_weight if total_weight > 0 else 0
            
            # 判断是否可疑
            adjusted_threshold = self.threshold
            is_suspicious = score >= adjusted_threshold
            
            results[node] = {
                'probability': score,
                'matched_rules': matched_score,
                'total_rules': total_weight,
                'is_suspicious': is_suspicious
            }
        
        suspicious_count = sum(1 for res in results.values() if res['is_suspicious'])
        logger.info(f"基于规则的洗钱检测完成，共发现 {suspicious_count} 个可疑节点")
        return results
    
    def model_based_detection(self, features):
        """基于模型的可疑账户检测"""
        if not self.models or len(self.models) == 0:
            return {}
        
        results = defaultdict(float)
        # 准备特征矩阵
        X = []
        nodes = list(features.keys())
        
        # 如果没有节点数据，直接返回
        if not nodes:
            logger.warning("没有可处理的节点特征数据，模型检测将跳过")
            return {}
        
        # 使用类定义的原始特征集
        original_features = self.original_features

        # 确保所有特征都存在
        for node in nodes:
            feats = features[node]
            row = []
            for f in original_features:
                row.append(feats.get(f, 0))
            X.append(row)
        
        # 检查特征矩阵是否为空
        if not X:
            logger.warning("特征矩阵为空，无法进行模型检测")
            return {}
        
        # 特征标准化
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        
        # 确保输入是2D数组
        if X_scaled.ndim == 1:
            X_scaled = X_scaled.reshape(-1, 1)
        elif X_scaled.ndim == 0:
            logger.warning("特征矩阵为空，无法进行模型检测")
            return {}
        
        # 检查是否有特征列
        if X_scaled.shape[1] == 0:
            logger.warning("特征矩阵没有特征列，无法进行模型检测")
            return {}

        # 使用所有模型进行预测并平均
        for name, model in self.models.items():
            try:
                # 抑制特征名称警告
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="X does not have valid feature names")
                    y_pred = model.predict_proba(X_scaled)[:, 1]
                for i, node in enumerate(nodes):
                    results[node] += y_pred[i] / len(self.models)
            except Exception as e:
                logger.error(f"模型 {name} 预测失败: {e}")
        
        return dict(results)
    
    def load_common_nodes(self):
        """加载需要检测的节点列表"""
        nodes = set()
        try:
            with open(self.args.nodes_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    node = row.get('node')
                    if node:
                        nodes.add(node)
            logger.info(f"从 {self.args.nodes_file} 加载了 {len(nodes)} 个节点")
        except Exception as e:
            logger.error(f"加载节点文件失败: {e}")
        return nodes

    def detect_money_laundering(self, features):
        """融合规则与模型的洗钱账户检测方法"""
        logger.info("开始融合规则与模型的洗钱检测...")
        
        # 执行规则检测
        rule_results = self.rule_based_detection(features)
        
        # 执行模型检测
        model_results = self.model_based_detection(features) if self.models else {}

        
        # 获取节点列表
        common_nodes = self.load_common_nodes()
        final_results = {}
        
        # 打印基于规则检测的所有可疑账户及其得分
        logger.info("基于规则检测的可疑账户:")
        rule_suspicious_count = 0
        for node, result in rule_results.items():
            if result['probability'] >= self.threshold:
                rule_suspicious_count += 1
                logger.info(f"账户 {node}: 规则得分 {result['probability']:.4f}")
        logger.info(f"共检测出 {rule_suspicious_count} 个基于规则的可疑账户")

        # 新增：打印位于common_nodes内的规则检测可疑账户
        logger.info("位于common_nodes内的规则检测可疑账户:")
        common_count = 0
        for node in common_nodes:
            if node in rule_results and rule_results[node]['probability'] >= self.threshold:
                logger.info(f"账户 {node}: 规则得分 {rule_results[node]['probability']:.4f}")
                common_count += 1
        logger.info(f"位于common_nodes内的可疑账户总数: {common_count}")

        for node in common_nodes:
            if node not in features:
                logger.warning(f"节点 {node} 不在交易数据中，已跳过")
                continue
            
            # 获取规则得分和模型得分
            rule_score = rule_results[node]['probability']
            model_score = model_results.get(node, 0)
            
            # 加权融合
            if self.models:
                final_score = self.rule_weight * rule_score + self.model_weight * model_score
            else:
                final_score = rule_score
            
            # 使用原始阈值判断
            is_suspicious = final_score >= self.threshold
            
            final_results[node] = {
                'rule_score': rule_score,
                'model_score': model_score,
                'final_score': final_score,
                'is_suspicious': is_suspicious,
                'features': features[node]
            }
        
        logger.info(f"融合检测完成,疑似洗钱账户数量: {sum(1 for r in final_results.values() if r['is_suspicious'])}")
        
        # 打印基于模型检测的可疑账户及其得分
        logger.info("基于模型检测的可疑账户:")
        suspicious_count = 0
        for node, res in final_results.items():
            if res['is_suspicious']:
                suspicious_count += 1
                logger.info(f"账户 {node}: 模型得分 {res['model_score']:.4f}")
        logger.info(f"共检测出 {suspicious_count} 个可疑账户")
        return final_results
    
    def save_results(self, results):
        """保存检测结果到CSV文件"""
        # 确保输出目录存在
        os.makedirs(self.args.out_dir, exist_ok=True)
        
        # 保存详细结果
        detailed_path = os.path.join(self.args.out_dir, 'suspicious_accounts_detailed.csv')
        with open(detailed_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Account', 'SuspiciousScore', 'IsSuspicious', 'RuleScore', 'ModelScore'])
            for node, res in results.items():
                writer.writerow([
                    node,
                    res['final_score'],
                    res['is_suspicious'],
                    res['rule_score'],
                    res['model_score']
                ])
        logger.info(f"详细检测结果已保存至: {detailed_path}")
        
        # 保存仅包含可疑账户的结果
        suspicious_path = os.path.join(self.args.out_dir, 'suspicious_accounts.csv')
        with open(suspicious_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Account', 'SuspiciousScore'])
            for node, res in results.items():
                if res['is_suspicious']:
                    writer.writerow([node, res['final_score']])
        logger.info(f"可疑账户结果已保存至: {suspicious_path}")
        
        return detailed_path, suspicious_path
    
    def run(self):
        """运行可疑账户检测流程"""
        start_time = time.time()
        logger.info("===== 可疑账户检测系统开始运行 ====")
        
        try:
            # 加载交易数据
            tx_files = [self.args.txs_file]
            if not tx_files:
                logger.error("未找到交易数据文件")
                return
            
            # 合并所有交易数据
            logger.info(f"找到{len(tx_files)}个交易数据文件，开始合并...")
            all_txs = []
            for tx_file in tx_files:
                tx_path = tx_file
                with open(tx_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    if not all_txs:
                        all_txs.append(header)
                    all_txs.extend(list(reader))
            
            # 保存合并后的交易数据
            merged_tx_path = os.path.join(self.args.out_dir, 'merged_transactions.csv')
            with open(merged_tx_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(all_txs)
            logger.info(f"交易数据合并完成，共{len(all_txs)-1}条交易，保存至: {merged_tx_path}")
            
            # 构建交易图
            logger.info("开始构建交易图...")
            g = self.load_graph(self.args.txs_file)
            
            # 提取特征
            features = self.extract_features(g)
            
            # 检测可疑账户
            results = self.detect_money_laundering(features)
            
            # 保存结果
            detailed_path, suspicious_path = self.save_results(results)
            
            # 统计结果
            suspicious_count = sum(1 for res in results.values() if res['is_suspicious'])
            total_count = len(results)
            if total_count > 0:
                logger.info(f"检测完成: 共{total_count}个账户，其中{str(suspicious_count)}个可疑账户 ({suspicious_count/total_count*100:.2f}%)")
            else:
                logger.info("检测完成: 未处理任何账户")
            
        except Exception as e:
            logger.error(f"检测过程中发生错误: {e}", exc_info=True)
        finally:
            end_time = time.time()
            logger.info(f"===== 可疑账户检测系统运行结束，耗时{end_time - start_time:.2f}秒 ====")

if __name__ == '__main__':
    detector = SuspiciousAccountDetector()
    detector.run()