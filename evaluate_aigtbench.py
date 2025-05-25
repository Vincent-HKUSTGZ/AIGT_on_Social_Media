import argparse
import warnings
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import os

# 关闭特定的FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module='transformers')

def load_model_and_tokenizer(checkpoint_path, device):
    """加载模型和tokenizer"""
    print(f"Loading model from {checkpoint_path}...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, num_labels=2)
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    return tokenizer, model

def load_aigtbench_dataset(data_source):
    """
    加载AIGTBench数据集
    data_source: 可以是 'huggingface' 或本地路径
    """
    if data_source.lower() == 'huggingface':
        print("Loading AIGTBench dataset from Hugging Face...")
        try:
            dataset = load_dataset("tarryzhang/AIGTBench")
            return dataset
        except Exception as e:
            print(f"Error loading from Hugging Face: {e}")
            print("Please make sure you have internet connection and the dataset exists.")
            return None
    else:
        # 本地路径
        if os.path.exists(data_source):
            print(f"Loading dataset from local path: {data_source}")
            try:
                dataset = load_from_disk(data_source)
                return dataset
            except Exception as e:
                print(f"Error loading from local path: {e}")
                return None
        else:
            print(f"Local path does not exist: {data_source}")
            return None

def prepare_dataloader(dataset, tokenizer, max_length=4096, batch_size=32):
    """准备DataLoader"""
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False)

def evaluate_model(model, dataloader, device):
    """评估模型"""
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return all_preds, all_labels

def print_and_save_results(labels, preds, dataset_name="Dataset", save_dir=None):
    """打印并保存评估结果"""
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    
    print(f"\n=== {dataset_name} Results ===")
    print(f"Accuracy: {acc:.3f}")
    print(f"F1 Score: {f1:.3f}")
    
    classification_rep = classification_report(labels, preds, target_names=['Human', 'AI-Generated'], digits=3)
    print("\nDetailed Classification Report:")
    print(classification_rep)
    
    # 保存结果到文件
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        result_file = os.path.join(save_dir, f"{dataset_name.lower().replace(' ', '_')}_results.txt")
        
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"=== {dataset_name} Results ===\n")
            f.write(f"Accuracy: {acc:.3f}\n")
            f.write(f"F1 Score: {f1:.3f}\n\n")
            f.write("Detailed Classification Report:\n")
            f.write(classification_rep)
        
        print(f"Results saved to: {result_file}")
    
    return {
        'accuracy': acc,
        'f1_score': f1,
        'classification_report': classification_rep
    }

def evaluate_by_platform(dataset, model, tokenizer, device, args, save_dir=None):
    """按平台分别评估"""
    platforms = ['medium', 'quora', 'reddit']
    platform_results = {}
    
    for platform in platforms:
        print(f"\n{'=' * 60}")
        print(f"Evaluating {platform} Platform")
        print(f"{'=' * 60}")
        
        # 过滤出当前平台的数据
        platform_data = dataset.filter(lambda x: x['social_media_platform'] == platform)
        print(f"Found {len(platform_data)} samples for {platform}")
        
        if len(platform_data) == 0:
            print(f"No data found for {platform}, skipping...")
            continue
        
        # 准备DataLoader
        dataloader = prepare_dataloader(platform_data, tokenizer, args.max_length, args.batch_size)
        
        # 评估模型
        preds, labels = evaluate_model(model, dataloader, device)
        
        # 打印并保存结果
        results = print_and_save_results(labels, preds, f"{platform} Platform", save_dir)
        platform_results[platform] = results
    
    return platform_results

def save_summary_results(platform_results, save_dir):
    """保存汇总结果"""
    if not save_dir or not platform_results:
        return
    
    summary_file = os.path.join(save_dir, "summary_results.txt")
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=== AIGTBench Platform-wise Evaluation Summary ===\n\n")
        
        # 创建汇总表格
        f.write("Platform\t\tAccuracy\tF1 Score\n")
        f.write("-" * 50 + "\n")
        
        for platform, results in platform_results.items():
            f.write(f"{platform:<15}\t{results['accuracy']:.3f}\t\t{results['f1_score']:.3f}\n")
        
        f.write("\n" + "=" * 50 + "\n\n")
        
        # 详细结果
        for platform, results in platform_results.items():
            f.write(f"=== {platform} Platform Detailed Results ===\n")
            f.write(f"Accuracy: {results['accuracy']:.3f}\n")
            f.write(f"F1 Score: {results['f1_score']:.3f}\n\n")
            f.write("Classification Report:\n")
            f.write(results['classification_report'])
            f.write("\n" + "-" * 50 + "\n\n")
    
    print(f"\nSummary results saved to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate AIGT Detection Model on AIGTBench Dataset by Platform')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the model checkpoint (local path or Hugging Face model name)')
    parser.add_argument('--data_source', type=str, default='huggingface',
                       help='Data source: "huggingface" or local dataset path (default: huggingface)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation (default: 32)')
    parser.add_argument('--max_length', type=int, default=4096,
                       help='Maximum sequence length (default: 4096)')
    parser.add_argument('--save_results', type=str, default='./results',
                       help='Directory to save evaluation results (default: ./results)')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型和tokenizer
    tokenizer, model = load_model_and_tokenizer(args.model_path, device)
    
    # 加载数据集
    dataset = load_aigtbench_dataset(args.data_source)
    if dataset is None:
        print("Failed to load dataset. Exiting...")
        return
    
    # 检查是否有test分割
    if hasattr(dataset, 'keys') and 'test' in dataset.keys():
        test_dataset = dataset['test']
        print(f"Using test split with {len(test_dataset)} samples")
    else:
        print("No test split found. Please make sure your dataset has a 'test' split.")
        return
    
    # 检查数据集是否包含平台信息
    if 'social_media_platform' not in test_dataset.column_names:
        print("Error: Dataset does not contain 'social_media_platform' column.")
        print(f"Available columns: {test_dataset.column_names}")
        return
    
    # 创建保存目录
    save_dir = args.save_results
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved to: {save_dir}")
    
    # 按平台评估
    platform_results = evaluate_by_platform(test_dataset, model, tokenizer, device, args, save_dir)
    
    # 保存汇总结果
    save_summary_results(platform_results, save_dir)
    
    # 打印最终汇总
    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Platform':<15} {'Accuracy':<10} {'F1 Score':<10}")
    print("-" * 60)
    
    for platform, results in platform_results.items():
        print(f"{platform:<15} {results['accuracy']:<10.3f} {results['f1_score']:<10.3f}")
    
    print(f"\nAll results saved to: {save_dir}")

if __name__ == "__main__":
    main()