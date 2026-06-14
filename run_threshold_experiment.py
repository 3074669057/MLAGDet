import subprocess
import re
import pandas as pd
import os
import time
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'threshold_experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 设置要测试的threshold值
thresholds = [0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.28, 0.31, 0.33, 0.35]

# 存储结果
results = []

# 定义正则表达式来提取指标
precision_pattern = r'精确率:\s+(\d+\.\d+)%'
recall_pattern = r'召回率:\s+(\d+\.\d+)%'
f1_pattern = r'F1分数:\s+(\d+\.\d+)%'
auc_pattern = r'AUC值:\s+(\d+\.\d+)'

# 遍历所有threshold值
for threshold in thresholds:
    logger.info(f"\n=== 开始运行阈值: {threshold} ===")
    
    command = f'python money_laundering_detector.py -i transactions -o results_{threshold} -t {threshold}'
    
    # 运行命令并捕获输出
    try:
        logger.info(f"执行命令: {command}")
        start_time = time.time()
        
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            universal_newlines=True
        )
        
        # 设置超时时间为600秒（10分钟）
        timeout = 600
        stdout, stderr = process.communicate(timeout=timeout)
        
        elapsed_time = time.time() - start_time
        logger.info(f"命令执行完成，耗时: {elapsed_time:.2f}秒")
        
        # 检查是否有错误
        if process.returncode != 0:
            logger.error(f"命令执行失败，返回码: {process.returncode}")
            logger.error(f"错误信息: {stderr}")
            results.append({
                'threshold': threshold,
                'precision': None,
                'recall': None,
                'f1_score': None,
                'auc': None,
                'error': stderr,
                'status': 'error'
            })
            continue
        
        # 提取指标
        precision_match = re.search(precision_pattern, stdout)
        recall_match = re.search(recall_pattern, stdout)
        f1_match = re.search(f1_pattern, stdout)
        auc_match = re.search(auc_pattern, stdout)
        
        # 解析指标
        precision = float(precision_match.group(1)) if precision_match else None
        recall = float(recall_match.group(1)) if recall_match else None
        f1_score = float(f1_match.group(1)) if f1_match else None
        auc = float(auc_match.group(1)) if auc_match else None
        
        # 记录关键输出信息
        if precision_match:
            logger.info(f"提取到精确率: {precision}%")
        else:
            logger.warning("未找到精确率信息")
            logger.debug(f"标准输出内容前2000字符: {stdout[:2000]}")
            
        if recall_match:
            logger.info(f"提取到召回率: {recall}%")
        else:
            logger.warning("未找到召回率信息")
            
        if f1_match:
            logger.info(f"提取到F1分数: {f1_score}%")
        else:
            logger.warning("未找到F1分数信息")
            
        if auc_match:
            logger.info(f"提取到AUC值: {auc}")
        else:
            logger.warning("未找到AUC值信息")
        
        # 存储结果
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'auc': auc,
            'status': 'success',
            'execution_time': elapsed_time
        })
        
        # 打印提取的指标
        print(f"精确率: {precision}%")
        print(f"召回率: {recall}%")
        print(f"F1分数: {f1_score}%")
        print(f"AUC值: {auc}")
        
        # 等待一段时间，避免系统负载过高
        time.sleep(3)
        
    except subprocess.TimeoutExpired:
        logger.error(f"命令执行超时 ({timeout}秒)")
        process.kill()
        results.append({
            'threshold': threshold,
            'precision': None,
            'recall': None,
            'f1_score': None,
            'auc': None,
            'error': f'Command timed out after {timeout} seconds',
            'status': 'timeout'
        })
    except Exception as e:
        logger.error(f"运行时出错: {e}", exc_info=True)
        results.append({
            'threshold': threshold,
            'precision': None,
            'recall': None,
            'f1_score': None,
            'auc': None,
            'error': str(e),
            'status': 'exception'
        })

# 将结果转换为DataFrame
df = pd.DataFrame(results)

# 保存到CSV文件
csv_file = 'threshold_experiment_results.csv'
df.to_csv(csv_file, index=False, encoding='utf-8-sig')

logger.info(f"\n=== 实验完成 ===")
logger.info(f"结果已保存到: {csv_file}")
logger.info(f"\n结果汇总:")
logger.info(df)

print(f"\n=== 实验完成 ===")
print(f"结果已保存到: {csv_file}")
print("\n结果汇总:")
print(df)
