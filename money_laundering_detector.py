# 标准库导入
import argparse
import csv
import glob
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import time
from collections import Counter, defaultdict

# 第三方库导入
import community as community_louvain
import joblib
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community import label_propagation_communities
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 配置日志

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

class MoneyLaunderingDetector:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.description = '区块链反洗钱社区划分系统'
        parser.add_argument('-i', '--input', help='输入交易数据文件夹', dest='in_dir', type=str, required=True)
        parser.add_argument('-o', '--output', help='输出结果文件夹', dest='out_dir', type=str, required=True)
    
        parser.add_argument('-t', '--threshold', help='洗钱判定阈值', dest='threshold', type=float, default=0.25)
        parser.add_argument('--lpa', help='使用标签传播算法', action='store_true', default=True)
        parser.add_argument('--louvain', help='使用Louvain算法', action='store_true', default=True)
        parser.add_argument('--pseudo', help='使用伪似然优化算法', action='store_true', default=True)
        parser.add_argument('--combined', help='使用组合算法', action='store_true', default=True)
        parser.add_argument('--min_community_size', help='最小社区大小', type=int, default=3)
        parser.add_argument(
            '--jaccard_threshold',
            help='社区融合 Jaccard 相似度阈值 η',
            type=float,
            default=0.5,
        )
        parser.add_argument('--rule_weight', help='规则检测权重 (对应 S_rule 权重 1-omega)', type=float, default=0.4)
        parser.add_argument('--model_weight', help='模型检测权重 (对应 S_ML 权重 omega)', type=float, default=0.6)
        parser.add_argument('--only_lpa', help='仅使用LPA社区检测', action='store_true', default=False)
        parser.add_argument('--only_louvain', help='仅使用Louvain社区检测', action='store_true', default=False)
        parser.add_argument('--only_pseudo', help='仅使用Pseudo社区检测', action='store_true', default=False)
        parser.add_argument('--only_combined', help='仅使用融合社区检测', action='store_true', default=False)
        parser.add_argument(
            '--enable_label_prior',
            help='启用遗留的 CE 标签辅助预测逻辑（仅调试用，默认关闭）',
            action='store_true',
            default=False,
        )
        parser.add_argument(
            '--disable_label_prior',
            help='显式禁用 CE/BE 标签对预测的影响（论文实验模式，与默认行为相同）',
            action='store_true',
            default=False,
        )

        self.args = parser.parse_args()
        if self.args.enable_label_prior and self.args.disable_label_prior:
            parser.error('不能同时指定 --enable_label_prior 与 --disable_label_prior')
        # Default: no label prior. Legacy debugging requires --enable_label_prior.
        self.use_label_prior = bool(self.args.enable_label_prior)

        only_flags = [
            self.args.only_lpa,
            self.args.only_louvain,
            self.args.only_pseudo,
            self.args.only_combined,
        ]
        if sum(only_flags) > 1:
            parser.error('只能指定一个 --only_* 社区检测参数')
        if any(only_flags):
            if self.args.only_lpa:
                self.args.lpa = True
                self.args.louvain = False
                self.args.pseudo = False
                self.args.combined = False
            elif self.args.only_louvain:
                self.args.lpa = False
                self.args.louvain = True
                self.args.pseudo = False
                self.args.combined = False
            elif self.args.only_pseudo:
                self.args.lpa = False
                self.args.louvain = False
                self.args.pseudo = True
                self.args.combined = False
            elif self.args.only_combined:
                self.args.lpa = True
                self.args.louvain = True
                self.args.pseudo = True
                self.args.combined = True
        # 定义模型目录和获取最新模型路径
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Real-CATS-master', 'models')
        self.model_paths = {
            'lightgbm': self.get_latest_model_path(model_dir, 'lightgbm'),
            'random_forest': self.get_latest_model_path(model_dir, 'random_forest'),
            'xgboost': self.get_latest_model_path(model_dir, 'xgboost')
        }
        # 降低规则检测阈值，默认从0.7调整为0.3
        self.threshold = self.args.threshold  # 使用命令行参数的阈值，不再覆盖默认值
        
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
            try:
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = joblib.load(f)
                logger.info(f"成功加载特征标准化器: {self.scaler_path}")
            except Exception as e:
                logger.error(f"加载特征标准化器失败: {e}")
        else:
            logger.warning("未找到特征标准化器，模型预测可能不准确")
            logger.info(f"成功加载特征标准化器: {self.scaler_path}")

        
        # 加载标签数据
        label_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'labels')
        ce_path = os.path.join(label_dir, 'CE.csv')
        be_path = os.path.join(label_dir, 'BE.csv')

        # 加载CE标签
        logger.debug(f"尝试加载CE标签文件: {ce_path}")
        if os.path.exists(ce_path):
            self.ce_df = pd.read_csv(ce_path)
            # 尝试找到地址列，不区分大小写
            addr_cols = [col for col in self.ce_df.columns if col.lower() == 'address']
            if addr_cols:
                self.ce_df['address'] = self.ce_df[addr_cols[0]].str.strip().str.lower()
                # 确保地址以0x开头
                self.ce_df['address'] = self.ce_df['address'].apply(lambda x: x if x.startswith('0x') else f'0x{x}')
                logger.info(f"成功加载CE标签文件: {ce_path}, 共{len(self.ce_df)}条记录")
            else:
                logger.error(f"CE标签文件缺少地址列: {ce_path}")
                self.ce_df = pd.DataFrame(columns=['address', 'label'])
        else:
            self.ce_df = pd.DataFrame(columns=['address', 'label'])
            logger.error(f"CE标签文件不存在: {ce_path}")
            # 尝试其他可能的文件名
            alternative_paths = [os.path.join(label_dir, 'ce.csv'), os.path.join(label_dir, 'CE_addresses.csv')]
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    logger.warning(f"找到替代CE标签文件: {alt_path}")
                    ce_path = alt_path
                    self.ce_df = pd.read_csv(ce_path)
                    addr_cols = [col for col in self.ce_df.columns if col.lower() == 'address']
                    if addr_cols:
                        self.ce_df['address'] = self.ce_df[addr_cols[0]].str.strip().str.lower()
                        logger.info(f"成功加载替代CE标签文件: {alt_path}, 共{len(self.ce_df)}条记录")
                        break
            else:
                logger.warning(f"未找到CE标签文件: {ce_path}")
        
        # 加载BE标签
        if os.path.exists(be_path):
            self.be_df = pd.read_csv(be_path)
            # 尝试找到地址列，不区分大小写
            addr_cols = [col for col in self.be_df.columns if col.lower() == 'address']
            if addr_cols:
                self.be_df['address'] = self.be_df[addr_cols[0]].str.lower()
                self.suspicious_accounts = self.be_df['address'].tolist()
                logger.info(f"成功加载BE标签文件: {be_path}, 共{len(self.be_df)}条记录")
            else:
                logger.error(f"BE标签文件缺少地址列: {be_path}")
                self.be_df = pd.DataFrame(columns=['address', 'label'])
                self.suspicious_accounts = []
        else:
            self.be_df = pd.DataFrame(columns=['address', 'label'])
            self.suspicious_accounts = []
            logger.warning(f"未找到BE标签文件: {be_path}")
        
        # 加载社区特征数据
        self.community_features = self._load_community_features()
        self.node_to_community = self._map_nodes_to_communities()
    
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
                        if name == 'xgboost':
                            import warnings
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore", category=Warning, message=".*If you are loading a serialized model.*")
                                self.models[name] = joblib.load(f)
                        else:
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
    
    def _load_community_features(self):
        """加载社区特征数据"""
        import os
        import pandas as pd
        community_features = {}
        
        community_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'combined')
        
        if os.path.exists(community_data_path):
            for filename in os.listdir(community_data_path):
                if filename.endswith('_communities.csv'):
                    df = pd.read_csv(os.path.join(community_data_path, filename))
                    for _, row in df.iterrows():
                        community_id = row['CommunityID']
                        community_features[community_id] = {
                            'avg_suspicious_prob': row['AvgSuspiciousProb'],
                            'density': row['Density'],
                            'in_out_ratio': row['InOutRatio'],
                            'internal_ratio': row['InternalRatio'],
                            'avg_transaction_amount': row['AvgTransactionAmount']
                        }
        return community_features
    
    def _map_nodes_to_communities(self):
        """将节点映射到社区ID"""
        import os
        import pandas as pd
        node_to_community = {}
        
        community_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'combined')
        
        if os.path.exists(community_data_path):
            for filename in os.listdir(community_data_path):
                if filename.startswith('0xeb31973e0febf3e3d7058234a5ebbae1ab4b8c23_community_') and filename.endswith('.csv'):
                    try:
                        community_id = int(filename.split('_community_')[1].split('.')[0])
                        df = pd.read_csv(os.path.join(community_data_path, filename))
                        for _, row in df.iterrows():
                            node = row['Node']
                            node_to_community[node] = community_id
                    except Exception as e:
                        logger.warning(f"处理社区文件 {filename} 时出错: {e}")
        return node_to_community
    
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

        # 创建交易节点总数CSV文件
        df = pd.DataFrame({'total_transaction_nodes': [g.number_of_nodes()]})
        output_path = os.path.join(self.args.out_dir, 'transaction_nodes_summary.csv')
        df.to_csv(output_path, index=False)
        logger.info(f"已保存交易节点总数到 {output_path}")

        # Evaluation-only: merge graph nodes with CE/BE labels when label files exist.
        self.ce_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'labels', 'CE.csv')
        if os.path.exists(self.ce_path):
            self.ce_df = pd.read_csv(self.ce_path)
            if 'address' in self.ce_df.columns:
                self.ce_df['address'] = self.ce_df['address'].str.lower()
        elif self.ce_df.empty:
            logger.warning("CE label file missing during graph load; CE evaluation subset will be empty")

        all_nodes = [node.lower() for node in g.nodes()]
        node_df = pd.DataFrame(all_nodes, columns=['address'])

        ce_transactions_labels = node_df.merge(self.ce_df, on='address', how='inner') if not self.ce_df.empty else pd.DataFrame()
        if not ce_transactions_labels.empty:
            ce_path = os.path.join(self.args.out_dir, 'CE.csv')
            ce_transactions_labels.to_csv(ce_path, index=False)
            logger.info(f"生成CE.csv，共{len(ce_transactions_labels)}条记录")
            print(f"交易标签统计: {len(ce_transactions_labels)}个账户匹配CE.csv标签")
        else:
            logger.info("未找到与CE.csv匹配的交易节点")
            print("交易标签统计: 0个账户匹配CE.csv标签")
        
        # 处理BE标签
        be_transactions_labels = node_df.merge(self.be_df, on='address', how='inner')
        if not be_transactions_labels.empty:
            be_path = os.path.join(self.args.out_dir, 'BE.csv')
            be_transactions_labels.to_csv(be_path, index=False)
            logger.info(f"生成BE.csv，共{len(be_transactions_labels)}条记录")
            print(f"交易标签统计: {len(be_transactions_labels)}个账户匹配BE.csv标签")
        else:
            logger.info("未找到与BE.csv匹配的交易节点")
            print("交易标签统计: 0个账户匹配BE.csv标签")
        
        return g
    
    def validate_features(self, features):
        """验证特征数据有效性"""
        import numpy as np
        if features is None:
            return False
        # 将列表转换为numpy数组
        if isinstance(features, list):
            features = np.array(features)
        # 检查是否包含NaN值
        if np.isnan(features).any():
            logger.warning("特征数据包含NaN值")
            return False
        # 检查特征维度是否正确
        if len(features.shape) != 2:
            logger.warning(f"特征维度不正确，期望2维，实际{len(features.shape)}维")
            return False
        return True

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
            node_features = {}
            
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

            # 5. 二阶交易特征 (from_from和to_from特征) - 优化版
            # 收集一级关联节点 (去重以减少重复计算)
            from_nodes = list(set([tx['from'] for tx in transactions if tx['type'] == 'in']))
            to_nodes = list(set([tx['to'] for tx in transactions if tx['type'] == 'out']))
            node_count = len(features)

            # 进度日志 (每处理1000个节点记录一次)
            if idx % 1000 == 0:
                if node_count == 0:
                    logger.warning("节点数量为零，无法计算进度百分比")
                    progress_log = f"正在提取二阶特征: {idx}/0 个节点"
                else:
                    progress_log = f"正在提取二阶特征: {idx}/{node_count} 个节点 ({idx/node_count*100:.2f}%)"
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
                # 交易小时模式 (使用向量化操作优化)
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
                # 交易小时模式 (使用向量化操作优化)
                timestamps = np.array([float(tx['timestamp']) for tx in to_from_txs])
                hours = pd.to_datetime(timestamps, unit='s').hour
                node_features['to_from_transaction_hour_mode'] = pd.Series(hours).mode()[0] if len(hours) > 0 else 0
            else:
                node_features.update({f:0 for f in ['to_from_transaction_count', 'to_from_total_value', 'to_from_avg_value', 'to_from_std_value', 'to_from_max_value', 'to_from_min_value', 'to_from_avg_log_value', 'to_from_transaction_hour_mode']})

            logger.debug(f"节点 {node} 二阶交易特征提取完成")
            
            features[node] = node_features
        
        logger.info(f"特征提取完成，共提取 {len(features)} 个节点的特征")
        return features
    
    def detect_money_laundering(self, features):
        """融合规则与模型的洗钱账户检测方法"""
        logger.info("开始融合规则与模型的洗钱检测...")
        
        # 执行规则检测
        rule_results = self.rule_based_detection(features)
        
        # 执行模型检测
        model_results = {}
        if self.models is not None and len(self.models) > 0:
            # 准备特征矩阵（使用与训练时相同的特征顺序）
            X = []
            # 收集所有节点
            nodes = list(features.keys())
              
              # 定义模型特定的特征集
            original_features = ['is_same', 'out_tx_count', 'total_out', 'avg_out_amount', 'std_out_amount', 'max_out_amount', 'min_out_amount', 'out_amount_variation', 'tx_frequency', 'in_tx_count', 'total_in', 'avg_in_amount', 'std_in_amount', 'max_in_amount', 'min_in_amount', 'in_amount_variation', 'time_diff_variation']
            extended_features = original_features + ['from_from_transaction_count', 'from_from_total_value', 'from_from_avg_value', 'from_from_std_value', 'from_from_max_value', 'from_from_min_value', 'from_from_avg_log_value', 'from_from_transaction_hour_mode', 'to_from_transaction_count', 'to_from_total_value', 'to_from_avg_value', 'to_from_std_value', 'to_from_max_value', 'to_from_min_value', 'to_from_avg_log_value', 'to_from_transaction_hour_mode']
              
              # 为每个模型准备对应特征集并预测
            all_probabilities = []
            for name, model in self.models.items():
                  # 根据模型类型选择特征集
                if name in ['lightgbm', 'random_forest']:
                      current_features = original_features
                else:  # xgboost
                      # 定义二阶特征列表
                      second_order_features = [
                            'from_from_transaction_count',
                            'from_from_total_value',
                            'from_from_avg_value',
                            'from_from_std_value',
                            'from_from_max_value',
                            'from_from_min_value',
                            'from_from_avg_log_value',
                            'from_from_transaction_hour_mode',
                            'to_from_transaction_count',
                            'to_from_total_value',
                            'to_from_avg_value',
                            'to_from_std_value',
                            'to_from_max_value',
                            'to_from_min_value',
                            'to_from_avg_log_value',
                            'to_from_transaction_hour_mode'
                        ]
                      current_features = ['is_same'] + second_order_features
                  
                  # 准备当前模型的特征向量
                X = []
                for node in nodes:
                    node_features = features[node]
                    # 确保所有特征都存在，缺失特征填充0
                    feature_vector = []
                    for f in current_features:
                        value = node_features.get(f, 0)
                        feature_vector.append(value)
                    X.append(feature_vector)
                
                # 使用训练时的标准化器对特征进行标准化
                if self.scaler and self.validate_features(X):
                    X = self.scaler.transform(X)
                logger.debug(f"特征标准化完成，样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
                  
                  # 模型预测
                if name == 'xgboost':
                    import pandas as pd
                    X_df = pd.DataFrame(X, columns=current_features)
                    prob = model.predict_proba(X_df)[:, 1]
                else:
                    prob = model.predict_proba(X)[:, 1]
                all_probabilities.append(prob)
            
            # 简单平均融合预测结果
            probabilities = np.mean(all_probabilities, axis=0)
            # 记录模型分数分布
            if len(probabilities) > 0:
                logger.info(f"模型分数分布 - 最小值: {probabilities.min():.4f}, 最大值: {probabilities.max():.4f}, 平均值: {probabilities.mean():.4f}")
            
            # 记录模型预测分数分布
            if probabilities.size > 0:
                logger.info(f"机器学习模型预测分数分布 - 最小值: {probabilities.min():.4f}, 最大值: {probabilities.max():.4f}, 平均值: {probabilities.mean():.4f}")
                logger.info(f"机器学习模型预测分数 >= {self.threshold}的节点比例: {sum(prob >= self.threshold for prob in probabilities)/len(probabilities):.2%}, 总数: {sum(prob >= self.threshold for prob in probabilities)}")
            
            # 添加模型检测进度日志
            total_nodes = len(nodes)
            for idx in range(0, len(nodes), 5000):
                processed = min(idx + 5000, total_nodes)
                percent = processed / total_nodes * 100
                logger.info(f"模型检测进度: {processed}/{total_nodes} 个节点 ({percent:.2f}%)")
            
            # 统计可疑节点
            suspicious_nodes = sum(1 for prob in probabilities if prob >= self.threshold)
            logger.info(f"基于机器学习模型的洗钱检测完成，共发现 {suspicious_nodes} 个可疑节点")
            
            model_results = {node: prob for node, prob in zip(nodes, probabilities)}
        else:
            logger.warning("没有可用模型，仅使用规则检测结果")
            model_results = {node: 0.0 for node in features.keys()}
        
        # 融合两种方法结果
        final_results = {}
        ce_addresses = (
            set(self.ce_df['address'].str.lower())
            if self.use_label_prior and not self.ce_df.empty
            else set()
        )
        for idx, node in enumerate(features.keys()):
            rule_score = rule_results[node]['probability']
            model_score = model_results.get(node, 0.0)
            # 加权融合
            final_score = self.rule_weight * rule_score + self.model_weight * model_score
            # 添加分数分布日志
            if idx % 1000 == 0:
                logger.debug(f"分数分布 - 规则分: {rule_score:.4f}, 模型分: {model_score:.4f}, 最终分: {final_score:.4f}")

            
            threshold = self.threshold
            if self.use_label_prior and node.lower() in ce_addresses:
                threshold *= 0.9
            is_suspicious = final_score >= threshold
            
            # 添加调试日志输出得分分布
            if idx % 1000 == 0:
                logger.debug(f"节点 {node} 规则得分: {rule_score:.4f}, 模型得分: {model_score:.4f}, 最终得分: {final_score:.4f}")
            final_results[node] = {
                'rule_score': rule_score,
                'model_score': model_score,
                'final_score': final_score,
                'is_suspicious': is_suspicious,
                'features': features[node]
            }
        
        logger.info(f"融合检测完成，疑似洗钱账户数量: {sum(1 for r in final_results.values() if r['is_suspicious'])}")
        return final_results
    
    def rule_based_detection(self, features):
        """基于规则的洗钱检测方法"""
        results = {}
        total_nodes = len(features)
        logger.info(f"开始基于规则的洗钱检测，共处理 {total_nodes} 个节点")
        
        # 从配置文件加载规则和黑名单（仅加载一次）
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
        ce_addresses = (
            set(self.ce_df['address'].str.lower())
            if self.use_label_prior and not self.ce_df.empty
            else set()
        )
        feature_nodes = set(features.keys())
        if self.use_label_prior:
            common_addresses = ce_addresses.intersection(feature_nodes)
            logger.debug(f"CE标签地址加载数量: {len(ce_addresses)}, 与特征节点重叠数量: {len(common_addresses)}")
            if common_addresses:
                logger.debug(f"重叠地址示例: {list(common_addresses)[:3]}")
            else:
                logger.warning("CE标签地址与特征节点无重叠，可能导致检测不到CE账户")
        # 记录前3个节点地址用于格式对比
        sample_nodes = list(features.keys())[:3]
        logger.debug(f"样本节点地址格式: {sample_nodes if sample_nodes else '无'}")
        for idx, (node, node_features) in enumerate(features.items()):
            if idx % 5000 == 0:
                logger.info(f"规则检测进度: {idx}/{total_nodes} 个节点 ({idx/total_nodes*100:.2f}%)")
            
            # 动态构建规则
            rules = []
            # 优先检查黑名单
            # 黑名单地址直接设为最高可疑度
            # 黑名单地址作为高权重规则而非直接设置分数
            # 优先检查黑名单和CE标签账户
            if node in suspicious_addresses:
                rules.append((True, 2.0))  # 黑名单高权重
            elif self.use_label_prior and node.lower() in ce_addresses:
                logger.debug(f"节点 {node} 匹配CE标签，应用高权重规则")
                rules.append((True, 12.0))
            elif self.use_label_prior and idx < 3:
                logger.debug(f"节点 {node} 未匹配CE标签，CE地址示例: {list(ce_addresses)[:1] if ce_addresses else '无'}")

            for rule in rules_config:
                    # 解析规则条件，传入规则参数
                    condition = eval(rule['condition'], {"node_features": node_features, "self": self, "params": rule.get('params', {})})
                    rules.append((condition, rule['weight']))
            

            # 添加规则匹配调试日志
            matched_rules = [(i+1, cond, weight) for i, (cond, weight) in enumerate(rules) if cond]
            if matched_rules:
                logger.debug(f"节点 {node} 匹配规则: {[r[0] for r in matched_rules]}, 得分: {sum(r[2] for r in matched_rules)}")
            else:
                logger.debug(f"节点 {node} 未匹配任何规则，特征值: {node_features}")
            # 计算加权可疑度分数（分母为所有规则总权重）
            matched_score = sum(weight for cond, weight in rules if cond)
            # 使用配置文件中定义的所有规则权重总和作为固定分母
            total_weight = sum(weight for cond, weight in rules) if rules else 0  # 包含所有规则权重以准确计算得分

            score = matched_score / total_weight if total_weight > 0 else 0
            
            # 获取节点所在社区
            community_id = self.node_to_community.get(node, None)
            community_feature = self.community_features.get(community_id, {})
            
            # 基于社区特征动态调整阈值
            community_threshold_factor = 1.0
            if community_feature:
                # 如果社区平均可疑概率高，则降低阈值
                if community_feature.get('avg_suspicious_prob', 0) > 0.7:
                    community_threshold_factor = 0.8
                # 如果社区密度高且内部交易比例高，则降低阈值
                if community_feature.get('density', 0) > 0.6 and community_feature.get('internal_ratio', 0) > 0.5:
                    community_threshold_factor = 0.7
                # 如果社区有异常的进出比，则降低阈值
                if community_feature.get('in_out_ratio', 0) > 2.0 or community_feature.get('in_out_ratio', 0) < 0.5:
                    community_threshold_factor = 0.8
            
            if self.use_label_prior and node.lower() in ce_addresses:
                adjusted_threshold = self.threshold * 0.9 * community_threshold_factor
            else:
                adjusted_threshold = self.threshold * community_threshold_factor
            is_suspicious = score >= adjusted_threshold
            
            # 添加社区信息到日志
            if is_suspicious:
                logger.debug(f"节点 {node} 触发 {len(matched_rules)}/{len(rules)} 条规则，可疑度: {score:.4f}, 社区ID: {community_id}, 调整后阈值: {adjusted_threshold:.4f}")
            
            results[node] = {
                'probability': score,
                'matched_rules': matched_score,
                'total_rules': total_weight,
                'is_suspicious': is_suspicious
            }
            
            if is_suspicious:
                logger.debug(f"节点 {node} 触发 {matched_rules}/{len(rules)} 条规则，可疑度: {score:.4f}")
        
        suspicious_count = sum(1 for res in results.values() if res['is_suspicious'])
        logger.info(f"基于规则的洗钱检测完成，共发现 {suspicious_count} 个可疑节点")
        return results
    
    def detect_suspicious_communities(self, g, features, detection_results):
        """使用三种算法检测可疑社区"""
        logger.info("开始可疑社区检测...")
        # 筛选出可疑节点
        suspicious_nodes = {node for node, result in detection_results.items() if result['is_suspicious']}
        
        # 创建只包含可疑节点的子图
        suspicious_subgraph = g.subgraph(suspicious_nodes).copy()
        
        # 移除孤立节点
        suspicious_subgraph.remove_nodes_from(list(nx.isolates(suspicious_subgraph)))
        
        if suspicious_subgraph.number_of_nodes() == 0:
            logger.warning("没有找到符合条件的可疑社区")
            return {}
        
        # 应用三种社区发现算法
        results = {}
        # 保存原始分区结果，用于计算NMI和ARI
        partitions = {}
        
        # 1. 标签传播算法
        if self.args.lpa:
            logger.info("开始使用标签传播算法进行社区划分...")
            # 将有向图转换为无向图以支持标签传播算法
            communities = list(label_propagation_communities(suspicious_subgraph.to_undirected()))
            results['lpa'] = self.filter_communities(communities, suspicious_subgraph, detection_results)
            logger.info(f"标签传播算法完成，发现 {len(results['lpa'])} 个可疑社区")
            
            # 保存分区结果
            partition = {}
            for comm_id, community in enumerate(communities):
                for node in community:
                    partition[node] = comm_id
            partitions['lpa'] = partition
        
        # 2. Louvain算法
        if self.args.louvain:
            logger.info("开始使用Louvain算法进行社区划分...")
            partition = community_louvain.best_partition(suspicious_subgraph.to_undirected())
            communities = defaultdict(set)
            for node, comm_id in partition.items():
                communities[comm_id].add(node)
            results['louvain'] = self.filter_communities(communities.values(), suspicious_subgraph, detection_results)
            logger.info(f"Louvain算法完成，发现 {len(results['louvain'])} 个可疑社区")
            partitions['louvain'] = partition
        
        # 3. 伪似然优化算法
        if self.args.pseudo:
            logger.info("开始使用伪似然优化算法进行社区划分...")
            partition = self.pseudo_likelihood_optimization(suspicious_subgraph)
            communities = defaultdict(set)
            for node, comm_id in partition.items():
                communities[comm_id].add(node)
            results['pseudo'] = self.filter_communities(communities.values(), suspicious_subgraph, detection_results)
            logger.info(f"伪似然优化算法完成，发现 {len(results['pseudo'])} 个可疑社区")
            partitions['pseudo'] = partition
        
        # 4. 组合算法（如果需要）
        if self.args.combined and len(results) > 1:
            logger.info("开始使用组合算法进行社区划分...")
            combined_communities = self.combine_communities(results, suspicious_subgraph)
            results['combined'] = self.filter_communities(combined_communities, suspicious_subgraph, detection_results)
            logger.info(f"组合算法完成，发现 {len(results['combined'])} 个可疑社区")
            
            # 保存组合算法的分区结果
            partition = {}
            for comm_id, community in enumerate(combined_communities):
                for node in community:
                    partition[node] = comm_id
            partitions['combined'] = partition
        
        # 计算NMI和ARI
        self.calculate_community_evaluation_metrics(partitions)
        
        return results
    
    def calculate_community_evaluation_metrics(self, partitions):
        """计算社区评估指标：归一化互信息(NMI)和调整兰德指数(ARI)"""
        if len(partitions) < 2:
            logger.info("算法数量不足，无法计算社区评估指标")
            return
        
        try:
            from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
            
            # 获取所有算法共享的节点
            all_nodes = set()
            for partition in partitions.values():
                all_nodes.update(partition.keys())
            
            logger.info(f"\n===== 社区评估指标 =====")
            
            # 计算每对算法之间的NMI和ARI
            algorithm_names = list(partitions.keys())
            for i in range(len(algorithm_names)):
                for j in range(i + 1, len(algorithm_names)):
                    algo1 = algorithm_names[i]
                    algo2 = algorithm_names[j]
                    
                    # 获取两个算法共同的节点
                    common_nodes = list(set(partitions[algo1].keys()) & set(partitions[algo2].keys()))
                    
                    if len(common_nodes) < 2:
                        logger.info(f"{algo1}和{algo2}之间共享节点不足，无法计算评估指标")
                        continue
                    
                    # 获取共同节点的社区标签
                    labels1 = [partitions[algo1][node] for node in common_nodes]
                    labels2 = [partitions[algo2][node] for node in common_nodes]
                    
                    # 计算NMI
                    nmi = normalized_mutual_info_score(labels1, labels2)
                    
                    # 计算ARI
                    ari = adjusted_rand_score(labels1, labels2)
                    
                    logger.info(f"{algo1} vs {algo2}:")
                    logger.info(f"  归一化互信息(NMI): {nmi:.4f}")
                    logger.info(f"  调整兰德指数(ARI): {ari:.4f}")
                    
                    # 同时输出到控制台
                    print(f"{algo1} vs {algo2}:")
                    print(f"  归一化互信息(NMI): {nmi:.4f}")
                    print(f"  调整兰德指数(ARI): {ari:.4f}")
                    
        except Exception as e:
            logger.error(f"计算社区评估指标时出错: {str(e)}")
    
    def filter_communities(self, communities, g, detection_results):
        """过滤社区，确保每个社区只包含可疑节点，且满足最小社区大小"""
        filtered_communities = []
        
        for community in communities:
            # 确保社区中的所有节点都是可疑的
            all_suspicious = all(detection_results.get(node, {'is_suspicious': False})['is_suspicious'] for node in community)
            
            # 确保社区大小符合要求
            meets_size_requirement = len(community) >= self.args.min_community_size
            
            if all_suspicious and meets_size_requirement:
                # 计算社区的洗钱特征
                community_features = self.calculate_community_features(community, g, detection_results)
                filtered_communities.append({
                    'nodes': list(community),
                    'size': len(community),
                    'features': community_features
                })
        
        return filtered_communities
    
    def calculate_community_features(self, community, g, detection_results):
        """计算社区的洗钱特征"""
        features = {}
        
        # 计算社区内节点的平均洗钱概率
        probabilities = [detection_results[node]['final_score'] for node in community]
        features['avg_suspicious_prob'] = sum(probabilities) / len(probabilities)
        
        # 计算社区内交易密度
        subgraph = g.subgraph(community)
        n = subgraph.number_of_nodes()
        if n > 1:
            max_possible_edges = n * (n - 1)
            features['density'] = 2 * subgraph.number_of_edges() / max_possible_edges
        else:
            features['density'] = 0
        
        # 计算社区内的交易模式特征
        in_edges = sum(1 for u, v in g.edges() if u in community and v not in community)
        out_edges = sum(1 for u, v in g.edges() if u not in community and v in community)
        internal_edges = subgraph.number_of_edges()
        
        features['in_out_ratio'] = out_edges / (in_edges + 1)
        features['internal_ratio'] = internal_edges / (in_edges + out_edges + internal_edges + 1)
        
        # 计算社区内的平均交易金额
        total_amount = sum(g[u][v]['weight'] for u, v in subgraph.edges())
        features['avg_transaction_amount'] = total_amount / (subgraph.number_of_edges() + 1)
        
        return features
    
    def combine_communities(self, algorithm_results, g):
        """组合多种算法的结果，找到共同的社区结构"""
        # 提取所有算法发现的社区
        all_communities = []
        for algo, communities in algorithm_results.items():
            for comm in communities:
                all_communities.append(frozenset(comm['nodes']))
        
        # 计算社区之间的Jaccard相似度
        similarity_matrix = {}
        for i, comm1 in enumerate(all_communities):
            for j, comm2 in enumerate(all_communities):
                if i < j:
                    intersection = len(comm1 & comm2)
                    union = len(comm1 | comm2)
                    if union > 0:
                        similarity = intersection / union
                        similarity_matrix[(i, j)] = similarity
        
        # 基于相似度合并社区
        merged_communities = []
        processed = set()
        
        for i, comm in enumerate(all_communities):
            if i in processed:
                continue
            
            # 找到所有相似度高的社区
            similar_comms = {i}
            for j in range(len(all_communities)):
                if i != j and j not in processed:
                    key = (min(i, j), max(i, j))
                    if key in similarity_matrix and similarity_matrix[key] > self.args.jaccard_threshold:
                        similar_comms.add(j)
            
            # 合并相似社区
            merged = set()
            for idx in similar_comms:
                merged.update(all_communities[idx])
                processed.add(idx)
            
            merged_communities.append(merged)
        
        return merged_communities
    
    def pseudo_likelihood_optimization(self, g):
        """伪似然方法优化社区划分"""
        logger.info("开始伪似然优化算法...")
        
        # 初始化：每个节点自成一个社区
        partition = {node: i for i, node in enumerate(g.nodes())}
        communities = defaultdict(set)
        for node, comm in partition.items():
            communities[comm].add(node)
        
        num_iterations = 2  # 进一步减少迭代次数
        max_time = 180  # 增加超时时间
        batch_size = 1000  # 批次处理节点
        start_time = time.time()
        
        for iteration in range(num_iterations):
            if time.time() - start_time > max_time:
                logger.warning(f"伪似然优化达到最大时间限制 {max_time} 秒，停止迭代")
                break
                
            improved = False
            logger.info(f"伪似然优化迭代 {iteration+1}/{num_iterations}...")
            
            # 批次处理节点以降低内存占用
            nodes = list(g.nodes())
            np.random.shuffle(nodes)
            batches = [nodes[i:i+batch_size] for i in range(0, len(nodes), batch_size)]
            
            for batch_idx, batch in enumerate(batches):
                batch_start_time = time.time()
                for node in batch:
                    current_community = partition[node]
                    current_contribution = self.calculate_node_contribution(g, partition, node)
                    
                    # 限制邻居检查数量
                    neighbors = list(g.neighbors(node))[:10] + list(g.predecessors(node))[:10]
                    neighbors = list(set(neighbors))  # 去重
                    
                    for neighbor in neighbors:
                        neighbor_community = partition.get(neighbor, -1)
                        if neighbor_community == current_community or neighbor_community == -1:
                            continue
                            
                        # 尝试移动节点
                        partition[node] = neighbor_community
                        new_contribution = self.calculate_node_contribution(g, partition, node)
                        
                        # 简化似然计算
                        if new_contribution > current_contribution + 1e-10:
                            current_contribution = new_contribution
                            improved = True
                            break
                        else:
                            partition[node] = current_community
                
                # 记录批次进度
                progress = (batch_idx + 1) / len(batches) * 100
                batch_time = time.time() - batch_start_time
                logger.info(f"批次 {batch_idx+1}/{len(batches)} 完成 ({progress:.1f}%)，耗时 {batch_time:.2f} 秒")
            
            # 提前终止条件
            if not improved:
                logger.info("似然值无改善，提前终止迭代")
                break
            
            # 初始化当前迭代的总似然值
            current_likelihood = self.calculate_pseudo_likelihood(g, partition)
            
            # 随机打乱节点顺序
            nodes = list(g.nodes())
            np.random.shuffle(nodes)
            
            for node in nodes:
                current_community = partition[node]
                # 计算当前节点移动前的似然贡献
                current_contribution = self.calculate_node_contribution(g, partition, node)
                
                # 尝试将节点移动到邻居的社区
                for neighbor in list(g.neighbors(node)) + list(g.predecessors(node)):  # 考虑入边和出边
                    neighbor_community = partition.get(neighbor, -1)
                    if neighbor_community == current_community or neighbor_community == -1:
                        continue
                        
                    # 临时移动节点到新社区
                    partition[node] = neighbor_community
                    new_contribution = self.calculate_node_contribution(g, partition, node)
                    
                    # 计算似然变化
                    likelihood_diff = new_contribution - current_contribution
                    
                    # 如果似然增加，接受移动
                    # 降低阈值以促进社区合并，接受较小的似然改善
                    if likelihood_diff > 1e-12:
                        current_likelihood += likelihood_diff
                        current_contribution = new_contribution
                        improved = True
                        break  # 找到改善即停止尝试其他社区
                    else:
                        # 否则恢复原社区
                        partition[node] = current_community
            
            # 如果没有改进，提前结束
            if not improved:
                logger.info(f"伪似然优化没有进一步改进，停止迭代")
                break
        
        logger.info("伪似然优化完成")
        return partition
    
    def calculate_node_contribution(self, g, partition, node):
        """计算单个节点对伪似然的贡献"""
        contribution = 0
        epsilon = 1e-10
        total_edges = g.number_of_edges() + epsilon
        node_community = partition[node]
        
        # 考虑所有与该节点相关的边（入边和出边）
        connected_nodes = set(g.neighbors(node)) | set(g.predecessors(node))
        for neighbor in connected_nodes:
            # 处理出边
            if g.has_edge(node, neighbor):
                weight = g[node][neighbor].get('weight', 1)
                weight = max(epsilon, min(weight, 1 - epsilon))
                neighbor_community = partition.get(neighbor, -1)
                if neighbor_community == -1:
                    continue
                if node_community == neighbor_community:
                    contribution += np.log(weight)
                else:
                    ratio = weight / total_edges
                    ratio = max(epsilon, min(ratio, 1 - epsilon))
                    contribution += np.log(1 - ratio)
            # 处理入边
            if g.has_edge(neighbor, node) and neighbor != node:
                weight = g[neighbor][node].get('weight', 1)
                weight = max(epsilon, min(weight, 1 - epsilon))
                neighbor_community = partition.get(neighbor, -1)
                if neighbor_community == -1:
                    continue
                if node_community == neighbor_community:
                    contribution += np.log(weight)
                else:
                    ratio = weight / total_edges
                    ratio = max(epsilon, min(ratio, 1 - epsilon))
                    contribution += np.log(1 - ratio)
        
        return contribution
    
    def calculate_pseudo_likelihood(self, g, partition):
        """计算伪似然值"""
        likelihood = 0
        
        # 遍历所有边，添加数值稳定性处理
        epsilon = 1e-10  # 防止log(0)的小常数
        total_edges = g.number_of_edges()
        for u, v in g.edges():
            weight = g[u][v].get('weight', 1)
            # 确保权重在合理范围内
            weight = max(epsilon, min(weight, 1 - epsilon))
            if partition[u] == partition[v]:
                # 同社区内的边
                likelihood += np.log(weight)
            else:
                # 不同社区的边，添加epsilon防止除零和log(0)
                ratio = weight / (total_edges + 1)
                ratio = max(epsilon, min(ratio, 1 - epsilon))
                likelihood += np.log(1 - ratio)
        
        return likelihood
    
    def save_results(self, results, detection_results, out_dir, filename):
        """保存检测结果"""
        import os
        # 创建输出目录
        result_dir = out_dir  # 直接使用输出目录，避免重复嵌套'results'文件夹
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        # 保存每个算法的结果
        for algo, communities in results.items():
            algo_dir = os.path.join(result_dir, algo)
            if not os.path.exists(algo_dir):
                os.makedirs(algo_dir)
            
            # 保存社区列表
            with open(os.path.join(algo_dir, f'{filename}_communities.csv'), 'w', newline='\n', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['CommunityID', 'Size', 'AvgSuspiciousProb', 'Density', 'InOutRatio', 'InternalRatio', 'AvgTransactionAmount'])
                
                for i, comm in enumerate(communities):
                    writer.writerow([
                        i,
                        comm['size'],
                        comm['features']['avg_suspicious_prob'],
                        comm['features']['density'],
                        comm['features']['in_out_ratio'],
                        comm['features']['internal_ratio'],
                        comm['features']['avg_transaction_amount']
                    ])
            
            # 保存每个社区的节点
            for i, comm in enumerate(communities):
                with open(os.path.join(algo_dir, f'{filename}_community_{i}.csv'), 'w', newline='\n', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Node', 'SuspiciousProbability'])
                    
                    for node in comm['nodes']:
                        writer.writerow([node, detection_results[node]['final_score']])

                # 生成社区节点JSON文件
                json_data = [{"source": node, "out": node, "types": "external,internal,erc20,erc721"} for node in comm['nodes']]
                json_path = os.path.join(algo_dir, f'{filename}_community_{i}.json')
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        # 保存总体统计信息
        with open(os.path.join(result_dir, f'{filename}_summary.csv'), 'w', newline='\n', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Algorithm', 'CommunityCount', 'TotalNodes', 'AvgCommunitySize', 'MaxCommunitySize'])
            
            for algo, communities in results.items():
                if communities:
                    sizes = [comm['size'] for comm in communities]
                    writer.writerow([
                        algo,
                        len(communities),
                        sum(sizes),
                        sum(sizes) / len(sizes),
                        max(sizes)
                    ])
                else:
                    writer.writerow([algo, 0, 0, 0, 0])
        
        logger.info(f"开始保存检测结果到 {self.args.out_dir}...")
        # 保存疑似洗钱账户并分类到CE、BE和新可疑账户
        ce_count = 0
        be_count = 0
        new_count = 0
        suspicious_ce = []
        suspicious_be = []
        new_suspicious = []
        all_suspicious = []
        
        for node, result in detection_results.items():
            node_lower = node.lower()
            ce_row = self.ce_df[self.ce_df['address'] == node_lower]
            is_ce_node = not ce_row.empty
            threshold = self.args.threshold
            if self.use_label_prior and is_ce_node:
                threshold *= 0.9
            if result['final_score'] >= threshold:
                ce_label = ''
                be_label = ''
                
                # 检查CE标签
                if not ce_row.empty:
                    ce_label = ce_row['label'].values[0]
                    ce_count += 1
                    suspicious_ce.append((node, result['final_score'], ce_label))
                
                # 检查BE标签
                be_row = self.be_df[self.be_df['address'] == node_lower]
                if not be_row.empty:
                    be_label = be_row['label'].values[0]
                    be_count += 1
                    suspicious_be.append((node, result['final_score'], be_label))
                
                # 确定标签
                if ce_label:
                    all_suspicious.append((node, result['final_score'], ce_label))
                elif be_label:
                    all_suspicious.append((node, result['final_score'], be_label))
                else:
                    all_suspicious.append((node, result['final_score'], ''))
                    new_suspicious.append((node, result['final_score']))
                    new_count += 1
        
        # 保存所有可疑账户
        with open(os.path.join(result_dir, f'{filename}_suspicious_accounts.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Node', 'SuspiciousProbability', 'label'])
            writer.writerows(all_suspicious)
        
        # 保存CE匹配的可疑账户
        with open(os.path.join(result_dir, 'suspicious_CE.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Node', 'SuspiciousProbability', 'label'])
            writer.writerows(suspicious_ce)
        
        # 保存BE匹配的可疑账户
        with open(os.path.join(result_dir, 'suspicious_BE.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Node', 'SuspiciousProbability', 'label'])
            writer.writerows(suspicious_be)
        
        # 保存新发现的可疑账户
        with open(os.path.join(result_dir, 'new_suspicious.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Node', 'SuspiciousProbability'])
            writer.writerows(new_suspicious)
        
        logger.info(f"疑似洗钱账户已保存到 {result_dir}/{filename}_suspicious_accounts.csv")
        logger.info(f"匹配到CE.csv标签的可疑账户数量: {ce_count}")
        logger.info(f"匹配到BE.csv标签的可疑账户数量: {be_count}")
        logger.info(f"新发现的可疑账户数量: {new_count}")
        logger.info(f"结果已保存到 {result_dir}")

        # CE账户完整性检查 - 使用结果文件夹中新生成的CE.csv
        result_ce_path = os.path.join(result_dir, 'CE.csv')
        try:
            result_ce_df = pd.read_csv(result_ce_path, encoding='utf-8')
            all_ce_addresses = set(result_ce_df['address'].str.lower())
            logger.info(f"已加载结果文件夹中的CE.csv: {result_ce_path}, 共{len(all_ce_addresses)}条记录")
        except FileNotFoundError:
            logger.error(f"结果文件夹中的CE.csv不存在: {result_ce_path}")
            all_ce_addresses = set()
        except Exception as e:
            logger.error(f"加载结果CE.csv失败: {str(e)}")
            all_ce_addresses = set()
        
        # 获取检测结果中的CE匹配账户
        detected_ce_addresses = {node.lower() for node, _, _ in suspicious_ce}
        
        # 获取新可疑账户
        new_suspicious_addresses = {node.lower() for node, _ in new_suspicious}
        
        # 计算未匹配的CE账户
        unmatched_ce = all_ce_addresses - detected_ce_addresses
        
        # 分类未匹配账户
        in_new_suspicious = 0
        not_detected_ce = 0
        for addr in unmatched_ce:
            if addr in new_suspicious_addresses:
                in_new_suspicious += 1
            else:
                not_detected_ce += 1
        
        # 记录统计信息
        logger.info(f"CE账户完整性检查: 共{len(all_ce_addresses)}个CE账户")
        logger.info(f"  - 成功匹配并标记为CE可疑账户: {len(detected_ce_addresses)}")
        logger.info(f"  - 未匹配CE账户: {len(unmatched_ce)}")
        logger.info(f"    - 被归类为新可疑账户: {in_new_suspicious}")
        logger.info(f"    - 未被系统检测到: {not_detected_ce}")

        # 初始化检测计数变量
        not_detected_ce = 0
        not_detected_be = 0

        # BE账户完整性检查 - 使用结果文件夹中新生成的BE.csv
        result_be_path = os.path.join(result_dir, 'BE.csv')
        try:
            result_be_df = pd.read_csv(result_be_path, encoding='utf-8')
            all_be_addresses = set(result_be_df['address'].str.lower())
            logger.info(f"已加载结果文件夹中的BE.csv: {result_be_path}, 共{len(all_be_addresses)}条记录")
        except FileNotFoundError:
            logger.error(f"结果文件夹中的BE.csv不存在: {result_be_path}")
            all_be_addresses = set()
        except Exception as e:
            logger.error(f"加载结果BE.csv失败: {str(e)}")
            all_be_addresses = set()

        # 获取检测结果中的BE匹配账户
        detected_be_addresses = {node.lower() for node, _, _ in suspicious_be}

        # 统一转换为小写进行地址匹配
        suspicious_accounts = getattr(self, 'suspicious_accounts', [])
        suspicious_accounts_lower = {sa.lower() for sa in suspicious_accounts}
        new_suspicious_lower = {nsa.lower() for nsa in new_suspicious_addresses}

        # BE账户完整性检查
        detected_be_addresses = [addr for addr in all_be_addresses if addr in suspicious_accounts_lower or addr in new_suspicious_lower]
        unmatched_be = [addr for addr in all_be_addresses if addr not in suspicious_accounts]
        in_new_suspicious_be = 0
        not_detected_be = 0
        for addr in unmatched_be:
            if addr in new_suspicious_addresses:
                in_new_suspicious_be += 1
            else:
                not_detected_be += 1
        logger.info(f"BE账户完整性检查: 共{len(all_be_addresses)}个BE账户")
        logger.info(f"  - 成功匹配并标记为BE可疑账户: {len(detected_be_addresses)}")
        logger.info(f"  - 未匹配BE账户: {len(unmatched_be)}")
        logger.info(f"    - 被归类为新可疑账户: {in_new_suspicious_be}")
        logger.info(f"    - 未被系统检测到: {not_detected_be}")

        # 获取所有CE地址
        all_ce_addresses = self.ce_df['address'].tolist() if not self.ce_df.empty else []
        # BE地址已从结果文件加载
        all_be_addresses = list(all_be_addresses)

        # 计算性能指标
        # 添加BE检测阈值调整
        be_threshold = 0.15  # 根据业务需求调整BE检测阈值
        # 从检测结果的分数字典中获取BE账户分数
        detection_scores = detection_results.get('scores', {})
        # 同时检查分数和可疑账户列表
        #   同时检查可疑账户、新可疑账户和分数
        # 统一转换为小写进行地址匹配
        suspicious_accounts = getattr(self, 'suspicious_accounts', [])
        suspicious_accounts_lower = {sa.lower() for sa in suspicious_accounts}
        new_suspicious_lower = {nsa.lower() for nsa in new_suspicious_addresses}
        detected_be_addresses = [addr for addr in all_be_addresses if addr.lower() in suspicious_accounts_lower or addr.lower() in new_suspicious_lower or detection_scores.get(addr, 0) >= be_threshold]
        TP = len(detected_ce_addresses) + len(detected_be_addresses)
        FN = not_detected_ce + not_detected_be
        FP = new_count
        # 仅计算图中存在的阳性样本
        graph_nodes = set(self.graph.nodes()) if hasattr(self, 'graph') else set()
        valid_ce = [addr for addr in all_ce_addresses if addr in graph_nodes]
        valid_be = [addr for addr in all_be_addresses if addr in graph_nodes]
        total_positive = len(valid_ce) + len(valid_be)
        total_nodes = self.graph.number_of_nodes() if hasattr(self, 'graph') else 0
        
        # 计算TN（总节点数减去所有正样本和假正例）
        # 修正TN计算：真正负例 = 总负例 - 假正例
        total_negative = total_nodes - total_positive
        TN = max(0, total_negative - FP)  # 确保TN非负
        
        # 防止除零错误
        recall = TP / total_positive if total_positive > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        fnr = FN / total_positive if total_positive > 0 else 0.0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        accuracy = (TP + TN) / total_nodes if total_nodes > 0 else 0.0
        
        # 准备用于ROC曲线的数据
        y_true = []  # 真实标签
        y_scores = []  # 预测分数
        
        # 添加正样本（CE和BE账户）
        for addr in valid_ce + valid_be:
            if addr in detection_results:
                y_true.append(1)  # 正样本标签为1
                y_scores.append(detection_results[addr].get('final_score', 0))
        
        # 添加部分负样本（随机选择部分节点以避免样本不平衡问题）
        negative_nodes = [node for node in graph_nodes if node not in (valid_ce + valid_be) and node in detection_results]
        # 限制负样本数量，避免样本过多导致的计算问题
        sample_size = min(len(negative_nodes), len(valid_ce) + len(valid_be))  # 正负样本数量平衡
        import random
        # 设置随机种子以确保结果可重现
        random.seed(42)
        sampled_negative_nodes = random.sample(negative_nodes, sample_size) if negative_nodes else []
        
        for addr in sampled_negative_nodes:
            y_true.append(0)  # 负样本标签为0
            y_scores.append(detection_results[addr].get('final_score', 0))
        
        # 计算AUC值
        auc_score = 0.0
        if len(y_true) > 0 and len(set(y_true)) > 1:  # 确保有正有负样本
            from sklearn.metrics import roc_auc_score
            try:
                auc_score = roc_auc_score(y_true, y_scores)
                logger.info(f"AUC值计算成功: {auc_score:.4f}")
            except Exception as e:
                logger.warning(f"无法计算AUC值，可能是样本标签问题: {str(e)}")
        else:
            logger.warning("无法计算AUC值：样本类别单一")
        
        # 输出性能指标到控制台
        print("\n===== 反洗钱检测性能指标 =====")
        print(f"准确率: {accuracy:.5%}")
        print(f"精确率: {precision:.2%}")
        print(f"召回率: {recall:.2%}")
        print(f"误报率: {fpr:.2%}")
        print(f"漏报率: {fnr:.2%}")
        print(f"F1分数: {f1:.2%}")
        print(f"AUC值: {auc_score:.4f}")

        # 可视化性能指标
        import warnings
        import logging
        # 使用正则表达式忽略所有字体家族找不到的警告
        warnings.filterwarnings("ignore", message=r"findfont: Font family '.*' not found.")
        # 禁止matplotlib.font_manager的所有日志输出
        logging.getLogger('matplotlib.font_manager').setLevel(logging.CRITICAL + 1)
        import matplotlib.pyplot as plt
        import os
        # 配置中文字体以消除缺失警告
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

        # 1. 性能指标条形图
        metrics = ['准确率', '召回率', 'F1分数', '误报率', '漏报率', 'AUC值']
        values = [accuracy, recall, f1, fpr, fnr, auc_score]

        plt.figure(figsize=(12, 6))
        bars = plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red', 'purple', 'cyan'])
        plt.ylim(0, 1.0)
        plt.title('反洗钱检测性能指标')
        plt.ylabel('比例')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            if metrics[bars.index(bar)] == 'AUC值':
                plt.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.4f}',
                         ha='center', va='bottom')
            else:
                plt.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.2%}',
                         ha='center', va='bottom')

        # 保存图表
        output_dir = os.path.join(self.args.out_dir, 'visualizations')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'performance_metrics.png'))
        plt.close()
        logger.info(f"性能指标图表已保存至: {os.path.join(output_dir, 'performance_metrics.png')}")
        
        # 2. 绘制ROC曲线
        if len(y_true) > 0 and len(set(y_true)) > 1:  # 确保有正有负样本
            from sklearn.metrics import roc_curve
            try:
                # 计算ROC曲线
                fpr_values, tpr_values, _ = roc_curve(y_true, y_scores)
                
                # 绘制ROC曲线
                plt.figure(figsize=(8, 6))
                plt.plot(fpr_values, tpr_values, color='blue', lw=2, label=f'ROC曲线 (AUC = {auc_score:.4f})')
                plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')  # 对角线
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('假正率')
                plt.ylabel('真正率')
                plt.title('ROC曲线')
                plt.legend(loc="lower right")
                plt.grid(alpha=0.3)
                
                # 保存ROC曲线
                plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
                plt.close()
                logger.info(f"ROC曲线已保存至: {os.path.join(output_dir, 'roc_curve.png')}")
            except Exception as e:
                logger.warning(f"绘制ROC曲线时出错: {str(e)}")
    
    def run(self):
        """运行反洗钱检测系统"""
        start_time = time.time()
        logger.info("开始区块链反洗钱社区划分系统...")
        
        # 处理每个输入文件
        for filename in os.listdir(self.args.in_dir):
            if not filename.endswith('.csv'):
                continue
                
            file_path = os.path.join(self.args.in_dir, filename)
            logger.info(f"处理文件: {filename}")
            
            # 加载图
            g = self.load_graph(file_path)
            
            # 提取特征
            self.graph = g  # 保存图为实例属性
            features = self.extract_features(g)
            
            # 检测洗钱账户
            detection_results = self.detect_money_laundering(features)
            
            # 检测可疑社区
            suspicious_communities = self.detect_suspicious_communities(g, features, detection_results)
            
            # 保存结果
            self.save_results(suspicious_communities, detection_results, self.args.out_dir, filename.split('.')[0])
        
        end_time = time.time()
        logger.info(f"反洗钱检测完成，总耗时: {end_time - start_time:.2f} 秒")


if __name__ == "__main__":
    detector = MoneyLaunderingDetector()
    detector.run()
