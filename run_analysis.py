import argparse
import os
import subprocess
import csv
import logging
from datetime import datetime

def main():
    # 配置日志
    log_filename = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(os.getcwd(), 'logs', log_filename)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    logging.info("===== 开始执行洗钱检测分析脚本 ====")
    parser = argparse.ArgumentParser(description='Run Scrapy spider and find common nodes in importance CSVs.')
    parser.add_argument('-a', required=True, help='Input JSON file name for Scrapy spider')
    args = parser.parse_args()
    json_file = args.a
    logging.info(f"接收到输入参数: JSON文件={json_file}")

    original_dir = os.getcwd()
    logging.info(f"当前工作目录: {original_dir}")

    try:
        spider_dir = os.path.join(original_dir, 'Spider')
        if not os.path.isdir(spider_dir):
            logging.error(f"错误: 未找到Spider目录 - {spider_dir}")
            return

        os.chdir(spider_dir)
        logging.info(f"已切换到Spider目录: {os.getcwd()}")

        # 添加调试日志并实时输出
        scrapy_command = f'scrapy crawl txs.eth.ttr -a file={os.path.basename(json_file)} --loglevel=DEBUG'
        logging.info(f"准备执行Scrapy命令: {scrapy_command}")

        # 使用Popen实时捕获输出
        process = subprocess.Popen(
            scrapy_command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # 实时记录输出
        for line in process.stdout:
            logging.info(f"Scrapy输出: {line.strip()}")

        # 等待进程完成并检查返回码
        try:
            process.wait(timeout=300)  # 5分钟超时
            if process.returncode != 0:
                logging.error(f"Scrapy命令执行失败，返回码: {process.returncode}")
                raise subprocess.CalledProcessError(process.returncode, scrapy_command)
        except subprocess.TimeoutExpired:
            logging.error("Scrapy命令执行超时")
            process.kill()
            raise

        logging.info("Scrapy命令执行成功")

        node_sets = []
        logging.info("开始搜索importance目录下的CSV文件...")
        for root, dirs, files in os.walk('.'):
            if os.path.basename(root) == 'importance':
                logging.info(f"发现importance目录: {root}")
                for file in files:
                    if file.endswith('.csv'):
                        csv_path = os.path.join(root, file)
                        logging.info(f"正在处理CSV文件: {csv_path}")
                        with open(csv_path, 'r', newline='') as f:
                            reader = csv.DictReader(f)
                            if 'node' not in reader.fieldnames:
                                logging.warning(f"警告: 在{csv_path}中未找到'node'列，已跳过")
                                continue
                            nodes = {row['node'] for row in reader}
                            logging.info(f"从{csv_path}中读取到{len(nodes)}个节点")
                            node_sets.append(nodes)

        if not node_sets:
            logging.warning("未在importance目录中找到包含'node'列的CSV文件")
            return

        logging.info(f"找到{len(node_sets)}个CSV文件，开始计算节点并集...")
        common_nodes = set.union(*node_sets)
        logging.info(f"计算完成，共找到{len(common_nodes)}个合并节点")

        input_path = os.path.abspath(json_file)
        input_dir = os.path.dirname(input_path)
        input_basename = os.path.basename(input_path)
        input_filename = os.path.splitext(input_basename)[0]
        output_file = os.path.join(input_dir, f"{input_filename}_common_nodes.csv")
        os.makedirs(input_dir, exist_ok=True)

        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['node'])
            for node in sorted(common_nodes):
                writer.writerow([node])

        logging.info(f"共同节点已保存到: {output_file}")

        # 合并交易CSV文件并去重
        logging.info("开始合并交易CSV文件...")
        tx_files = []
        spider_dir = os.path.join(original_dir, 'Spider')
        for subdir in os.listdir(spider_dir):
            subdir_path = os.path.join(spider_dir, subdir)
            if os.path.isdir(subdir_path):
                csv_file = os.path.join(subdir_path, f"{subdir}.csv")
                if os.path.exists(csv_file):
                    tx_files.append(csv_file)
                    logging.info(f"找到交易文件: {csv_file}")
        
        if not tx_files:
            logging.warning("未找到交易CSV文件")
        else:
            transactions = {}
            fieldnames = None
            for csv_path in tx_files:
                try:
                    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        if not fieldnames:
                            fieldnames = reader.fieldnames
                        # 检查是否包含必要的交易哈希字段
                        if 'hash' not in reader.fieldnames:
                            logging.warning(f"交易文件 {csv_path} 缺少 'hash' 字段，已跳过")
                            continue
                        file_transactions = 0
                        for row in reader:
                            file_transactions += 1
                            tx_hash = row['hash']
                            transactions[tx_hash] = row
                        logging.info(f"从 {csv_path} 读取了 {file_transactions} 条交易记录")
                except Exception as e:
                    logging.error(f"处理交易文件 {csv_path} 时出错: {str(e)}", exc_info=True)
            
            if transactions and fieldnames:
                output_txs_file = os.path.join(input_dir, f"{input_filename}_common_txs.csv")
                with open(output_txs_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(transactions.values())
                logging.info(f"合并去重后的交易数据已保存到: {output_txs_file}，共 {len(transactions)} 条记录")
            else:
                logging.warning("未收集到有效的交易数据，无法生成交易文件")

        logging.info("===== 脚本执行完成 ====")

    except subprocess.CalledProcessError as e:
        logging.error(f"执行Scrapy命令时出错: {e.stderr}", exc_info=True)
    except Exception as e:
        logging.error(f"发生错误: {str(e)}", exc_info=True)
    finally:
        os.chdir(original_dir)
        logging.info(f"已切换回原始目录: {original_dir}")

if __name__ == "__main__":
    main()