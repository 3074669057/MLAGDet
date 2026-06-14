import pandas as pd
import numpy as np
import logging
import joblib
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 配置类 - 集中管理所有参数
class Config:
    def __init__(self):
        self.dataset_path = os.path.dirname(os.path.abspath(__file__))
        self.ce_file = os.path.join(self.dataset_path, 'CE.tsv')
        self.be_file = os.path.join(self.dataset_path, 'BE.tsv')
        self.test_size = 0.2
        self.random_state = 42
        self.model_save_dir = os.path.join(self.dataset_path, 'models')
        self.model_version = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 创建模型保存目录
        os.makedirs(self.model_save_dir, exist_ok=True)

# 数据加载器类
class DataLoader:
    @staticmethod
    def load_data(config):
        try:
            logger.info(f'加载恶意地址数据: {config.ce_file}')
            df_criminal = pd.read_csv(config.ce_file, sep='\t')
            df_criminal['label'] = 1
            logger.info(f'恶意地址数据加载完成，共 {len(df_criminal)} 条记录')

            logger.info(f'加载良性地址数据: {config.be_file}')
            df_benign = pd.read_csv(config.be_file, sep='\t')
            df_benign['label'] = 0
            logger.info(f'良性地址数据加载完成，共 {len(df_benign)} 条记录')

            # 合并数据集
            df_combined = pd.concat([df_criminal, df_benign], ignore_index=True)
            logger.info(f'合并后数据集大小: {df_combined.shape}')

            # 加载额外的交易数据文件
            additional_file = os.path.join(config.dataset_path, '0xeb31973e0febf3e3d7058234a5ebbae1ab4b8c23.csv')
            logger.info(f'加载额外数据文件: {additional_file}')
            df_additional = pd.read_csv(additional_file)
            df_combined = pd.concat([df_combined, df_additional], ignore_index=True)
            logger.info(f'合并额外数据后数据集大小: {df_combined.shape}')

            # 处理重复列
            df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]
            logger.info(f'处理重复列后数据集大小: {df_combined.shape}')

            return df_combined
        except Exception as e:
            logger.error(f'数据加载失败: {str(e)}', exc_info=True)
            raise

# 数据预处理类
class DataPreprocessor:
    @staticmethod
    def preprocess_data(df):
        try:
            logger.info('开始数据预处理')
            
            # 处理缺失值
            df = df.fillna(0)
            logger.info('缺失值处理完成')

            # 确保数值列转换为正确类型
            numeric_cols = ['total_in', 'total_out', 'avg_out_amount', 'std_out_amount', 'max_out_amount', 'min_out_amount', 'avg_in_amount', 'value']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.fillna(0)

            # 移除非特征列
            if 'address' in df.columns:
                df = df.drop('address', axis=1)
                logger.info('已移除address列')

            # 分离特征和标签
            if 'label' in df.columns:
                X = df.drop('label', axis=1)
                y = df['label']
                logger.info(f'特征集大小: {X.shape}, 标签集大小: {y.shape}')
            else:
                X = df
                y = None
                logger.info(f'特征集大小: {X.shape}, 无标签数据')

            # 自动检测并处理所有 datetime 列
            # 注释掉日期时间特征扩展以减少特征数量
            # for col in X.columns:
            #     # 尝试将列转换为 datetime 类型
            #     try:
            #         # 尝试多种日期格式解析
            #         datetime_series = pd.to_datetime(X[col], errors='coerce', infer_datetime_format=True)
            #         # 检查是否有超过50%的值成功解析为日期
            #         if datetime_series.notna().mean() > 0.5:
            #             logger.info(f'检测到日期时间列: {col}，开始特征提取')
            #             # 转换为时间戳(秒级)
            #             X[f'{col}_timestamp'] = datetime_series.view('int64') // 10**9
            #             # 提取时间特征
            #             X[f'{col}_hour'] = datetime_series.dt.hour
            #             X[f'{col}_dayofweek'] = datetime_series.dt.dayofweek
            #             X[f'{col}_month'] = datetime_series.dt.month
            #             X[f'{col}_day'] = datetime_series.dt.day
            #             X[f'{col}_year'] = datetime_series.dt.year
            #             # 删除原始日期列
            #             X = X.drop(columns=[col])
            #             logger.info(f'已处理日期时间列: {col}，新增 {len([c for c in X.columns if c.startswith(f"{col}_")])} 个时间特征')
            #     except Exception as e:
            #         logger.warning(f'处理列 {col} 时出错: {str(e)}，跳过该列')

            # 验证必要列是否存在
            required_columns = ['from', 'to', 'value']
            missing_columns = [col for col in required_columns if col not in X.columns]
            if missing_columns:
                raise ValueError(f'训练数据缺少必要列: {missing_columns}')

            # 实现与检测器相同的特征工程逻辑
            # 1. 基本交易特征
            X['is_same'] = (X['from'] == X['to']).astype(int)
            X['in_degree'] = X.groupby('to')['to'].transform('count')
            X['out_degree'] = X.groupby('from')['from'].transform('count')
            X['total_in'] = X.groupby('to')['value'].transform('sum')
            X['total_out'] = X.groupby('from')['value'].transform('sum')
            X['balance'] = X['total_in'] - X['total_out']
            X['tx_count'] = X.groupby('from')['from'].transform('count') + X.groupby('to')['to'].transform('count')
            X['in_tx_count'] = X.groupby('to')['to'].transform('count')
            X['out_tx_count'] = X.groupby('from')['from'].transform('count')
            X['degree_ratio'] = X['out_degree'] / (X['in_degree'] + 1)

            # 2. 交易模式特征
            X['avg_out_amount'] = X.groupby('from')['value'].transform('mean')
            X['std_out_amount'] = X.groupby('from')['value'].transform('std').fillna(0)
            X['max_out_amount'] = X.groupby('from')['value'].transform('max')
            X['min_out_amount'] = X.groupby('from')['value'].transform('min')
            X['out_amount_variation'] = X['std_out_amount'] / (X['avg_out_amount'] + 1)
            X['round_amount_ratio'] = X['value'].apply(lambda x: 1 if x % 1e18 == 0 else 0)
            X['avg_in_amount'] = X.groupby('to')['value'].transform('mean')

            # 选择与检测器匹配的17个特征
            selected_features = [
                'is_same', 'in_degree', 'out_degree', 'degree_ratio', 'total_in', 
                'total_out', 'balance', 'tx_count', 'in_tx_count', 'out_tx_count', 
                'avg_out_amount', 'std_out_amount', 'max_out_amount', 'min_out_amount', 
                'out_amount_variation', 'round_amount_ratio', 'avg_in_amount'
            ]
            X = X[selected_features].fillna(0)
            logger.info(f'特征选择后保留 {X.shape[1]} 个特征')

            # 处理对象类型列，尝试转换为数值型
            for col in X.columns:
                if X[col].dtype == 'object':
                    logger.info(f'尝试将对象列 {col} 转换为数值型')
                    X[col] = pd.to_numeric(X[col], errors='coerce')

            # 删除转换后仍为对象类型的列
            X = X.select_dtypes(exclude=['object'])
            logger.info(f'转换并删除非数值列后特征集大小: {X.shape}')

            # 处理转换过程中产生的缺失值
            X = X.fillna(0)
            logger.info('已填充转换产生的缺失值')

            # 处理所有非数值列，包括object和string类型
            non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric_cols:
                logger.info(f'发现非数值列: {non_numeric_cols}，尝试转换为数值型')
                for col in non_numeric_cols:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                # 再次检查并移除仍为非数值的列
                remaining_non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
                if remaining_non_numeric:
                    logger.warning(f'无法转换以下非数值列，将被移除: {remaining_non_numeric}')
                    X = X.select_dtypes(include=[np.number])
            logger.info(f'选择数值型列后特征集大小: {X.shape}')

            # 保存特征名称用于后续分析
            feature_names = X.columns.tolist()

            # 特征缩放
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            logger.info('特征缩放完成')

            return X_scaled, y, feature_names
        except Exception as e:
            logger.error(f'数据预处理失败: {str(e)}', exc_info=True)
            raise

# 模型训练器类
class ModelTrainer:
    @staticmethod
    def train_random_forest(X_train, y_train):
        logger.info('开始训练随机森林模型')
        # 简单网格搜索调参
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20]
        }
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=Config().random_state),
            param_grid,
            cv=3,
            scoring='f1'
        )
        grid_search.fit(X_train, y_train)
        logger.info(f'随机森林最佳参数: {grid_search.best_params_}')
        return grid_search.best_estimator_

    @staticmethod
    def train_xgboost(X_train, y_train):
        logger.info('开始训练XGBoost模型')
        param_grid = {
            'n_estimators': [50, 100],
            'learning_rate': [0.1, 0.01]
        }
        grid_search = GridSearchCV(
            xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=Config().random_state),
            param_grid,
            cv=3,
            scoring='f1'
        )
        grid_search.fit(X_train, y_train)
        logger.info(f'XGBoost最佳参数: {grid_search.best_params_}')
        return grid_search.best_estimator_

    @staticmethod
    def train_lightgbm(X_train, y_train):
        logger.info('开始训练LightGBM模型')
        param_grid = {
            'n_estimators': [50, 100],
            'learning_rate': [0.1, 0.01]
        }
        grid_search = GridSearchCV(
            lgb.LGBMClassifier(random_state=Config().random_state),
            param_grid,
            cv=3,
            scoring='f1'
        )
        grid_search.fit(X_train, y_train)
        logger.info(f'LightGBM最佳参数: {grid_search.best_params_}')
        return grid_search.best_estimator_

# 模型评估器类
class ModelEvaluator:
    @staticmethod
    def evaluate_model(model, X_test, y_test, model_name, feature_names):
        logger.info(f'开始评估{model_name}模型')
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        # 基本评估指标
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }

        # AUC指标（如果模型支持概率预测）
        if y_proba is not None:
            metrics['auc'] = roc_auc_score(y_test, y_proba)

        # 打印详细分类报告
        logger.info(f'\n{model_name}分类报告:\n{classification_report(y_test, y_pred)}')
        logger.info(f'混淆矩阵:\n{confusion_matrix(y_test, y_pred)}')

        # 交叉验证
        cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='f1')
        metrics['cv_f1_mean'] = np.mean(cv_scores)
        metrics['cv_f1_std'] = np.std(cv_scores)
        logger.info(f'交叉验证F1分数: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

        # 特征重要性分析（仅树模型）
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            logger.info(f'{model_name} 特征重要性（前10名）:')
            # 限制显示数量为特征名称列表长度和10的较小值
            top_n = min(10, len(feature_names))
            for f in range(top_n):
                if indices[f] < len(feature_names):
                    logger.info(f'  {feature_names[indices[f]]}: {importances[indices[f]]:.4f}')
                else:
                    logger.warning(f'特征索引 {indices[f]} 超出特征名称列表范围，跳过')

        return metrics

    @staticmethod
    def log_metrics(model_name, metrics):
        logger.info(f'\n{model_name}评估指标:')
        for metric, value in metrics.items():
            logger.info(f'  {metric}: {value:.4f}')

# 模型保存工具类
class ModelSaver:
    @staticmethod
    def save_model(model, model_name, config, timestamp):
        model_prefix = model_name.lower().replace(" ", "_")
        model_path = os.path.join(config.model_save_dir, f"{model_prefix}_{timestamp}.pkl")
        import joblib
        joblib.dump(model, model_path)
        logger.info(f"{model_name}模型已保存至: {model_path}")
        return model_path

# 主函数
def main():
    try:
        # 初始化配置
        config = Config()
        logger.info('模型训练流程开始')

        # 加载并预处理数据
        df = DataLoader.load_data(config)
        X, y, feature_names = DataPreprocessor.preprocess_data(df)

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size, random_state=config.random_state, stratify=y
        )
        logger.info(f'训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}')
        # 检查标签分布并输出到诊断日志
        train_dist = pd.Series(y_train).value_counts(normalize=True)
        test_dist = pd.Series(y_test).value_counts(normalize=True)
        logger.info(f'训练集标签分布: {train_dist}')
        logger.info(f'测试集标签分布: {test_dist}')
        
        # 检查特征相关性并输出到诊断日志
        corr_matrix = pd.DataFrame(X_train).corr().abs()
        high_corr = np.sum(corr_matrix.values > 0.9) - X_train.shape[1]
        logger.info(f'高度相关特征对数(>0.9): {high_corr//2}')
        
        # 写入独立诊断日志文件
        with open('data_diagnostics.log', 'w') as f:
            f.write('=== 数据诊断报告 ===\n')
            f.write(f'训练集标签分布:\n{train_dist.to_string()}\n\n')
            f.write(f'测试集标签分布:\n{test_dist.to_string()}\n\n')
            f.write(f'高度相关特征对数(>0.9): {high_corr//2}\n\n')
            f.write('特征相关性矩阵(前5行5列):\n')
            f.write(corr_matrix.iloc[:5,:5].to_string())

        # 特征标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # 生成统一时间戳
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        logger.info(f"使用统一时间戳: {timestamp} 保存所有模型和标准化器")
        
        # 保存特征标准化器
        scaler_path = os.path.join(config.model_save_dir, f"scaler_{timestamp}.pkl")
        joblib.dump(scaler, scaler_path)
        logger.info(f"特征标准化器已保存至: {scaler_path}")
        
        # 训练模型
        models = {
            'Random Forest': ModelTrainer.train_random_forest(X_train, y_train),
            'XGBoost': ModelTrainer.train_xgboost(X_train, y_train),
            'LightGBM': ModelTrainer.train_lightgbm(X_train, y_train)
        }

        # 评估模型
        results = {}
        for model_name, model in models.items():
            metrics = ModelEvaluator.evaluate_model(model, X_test, y_test, model_name, feature_names)
            ModelEvaluator.log_metrics(model_name, metrics)
            results[model_name] = metrics

        # 保存模型
        for model_name, model in models.items():
            ModelSaver.save_model(model, model_name, config, timestamp)

        logger.info('模型训练流程完成')
        return X_train, X_test, y_train, y_test, scaler_path, timestamp

    except Exception as e:
        logger.error(f'训练流程失败: {str(e)}', exc_info=True)
        raise

if __name__ == '__main__':
    main()