import argparse
import os
import warnings
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments
)
from sklearn.metrics import accuracy_score, f1_score
import torch

# 关闭警告
warnings.filterwarnings("ignore", category=FutureWarning, module='transformers')

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

def load_model_and_tokenizer(model_path):
    """加载模型和tokenizer"""
    print(f"Loading model and tokenizer from {model_path}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
        print("Model and tokenizer loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def preprocess_dataset(dataset, tokenizer, max_length=4096, cache_dir=None):
    """预处理数据集"""
    def preprocess_function(examples):
        return tokenizer(
            examples['text'], 
            truncation=True, 
            padding='max_length', 
            max_length=max_length
        )
    
    processed_datasets = {}
    
    for split_name in dataset.keys():
        print(f"Preprocessing {split_name} split...")
        
        # 检查是否有缓存
        if cache_dir:
            cache_path = os.path.join(cache_dir, f"{split_name}_dataset")
            if os.path.exists(cache_path):
                print(f"Loading cached {split_name} dataset...")
                processed_datasets[split_name] = load_from_disk(cache_path)
                continue
        
        # 预处理数据
        processed_dataset = dataset[split_name].map(preprocess_function, batched=True)
        processed_datasets[split_name] = processed_dataset
        
        # 保存缓存
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"{split_name}_dataset")
            processed_dataset.save_to_disk(cache_path)
            print(f"Cached {split_name} dataset saved to {cache_path}")
    
    return DatasetDict(processed_datasets)

def compute_metrics(eval_pred):
    """计算评估指标"""
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {
        'accuracy': acc,
        'f1_score': f1
    }

class OSMDetTrainer(Trainer):
    """自定义Trainer类，用于训练OSM-Det模型"""
    
    def on_epoch_end(self):
        """在每个epoch结束时的回调"""
        super().on_epoch_end()
        epoch = int(self.state.epoch)
        
        # 保存每个epoch的模型
        output_dir = os.path.join(self.args.output_dir, f"checkpoint-epoch-{epoch}")
        self.save_model(output_dir)
        print(f"Model saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Train OSM-Det Model on AIGTBench Dataset')
    
    # 数据相关参数
    parser.add_argument('--data_source', type=str, default='huggingface',
                       help='Data source: "huggingface" or local dataset path (default: huggingface)')
    parser.add_argument('--cache_dir', type=str, default='./preprocessed_cache',
                       help='Directory to cache preprocessed datasets (default: ./preprocessed_cache)')
    
    # 模型相关参数
    parser.add_argument('--model_path', type=str, default='allenai/longformer-base-4096',
                       help='Path to base model (default: allenai/longformer-base-4096)')
    parser.add_argument('--max_length', type=int, default=4096,
                       help='Maximum sequence length (default: 4096)')
    
    # 训练相关参数
    parser.add_argument('--output_dir', type=str, default='./osm-det-model',
                       help='Output directory for trained model (default: ./osm-det-model)')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--batch_size', type=int, default=5,
                       help='Batch size per device (default: 5)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                       help='Gradient accumulation steps (default: 8)')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate (default: 2e-5)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay (default: 0.01)')
    
    args = parser.parse_args()
    os.environ['WANDB_DISABLED'] = 'true'  # 禁用wandb

    # 检查GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据集
    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)
    
    dataset = load_aigtbench_dataset(args.data_source)
    if dataset is None:
        print("Failed to load dataset. Exiting...")
        return
    
    # 检查数据集结构
    print(f"Dataset structure: {dataset}")
    required_splits = {'train', 'validation', 'test'}
    available_splits = set(dataset.keys())
    
    if not required_splits.issubset(available_splits):
        missing_splits = required_splits - available_splits
        print(f"Warning: Missing splits {missing_splits}")
        
        # 如果缺少validation，从train中分割
        if 'validation' not in available_splits and 'train' in available_splits:
            print("Creating validation split from train data...")
            train_val_split = dataset['train'].train_test_split(test_size=0.1, seed=42)
            dataset = DatasetDict({
                'train': train_val_split['train'],
                'validation': train_val_split['test'],
                'test': dataset.get('test', train_val_split['test'])
            })
    
    # 加载模型和tokenizer
    print("=" * 60)
    print("LOADING MODEL")
    print("=" * 60)
    
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    if model is None or tokenizer is None:
        print("Failed to load model. Exiting...")
        return
    
    # 预处理数据集
    print("=" * 60)
    print("PREPROCESSING DATA")
    print("=" * 60)
    
    processed_dataset = preprocess_dataset(
        dataset, 
        tokenizer, 
        max_length=args.max_length,
        cache_dir=args.cache_dir
    )
    
    print(f"Processed dataset: {processed_dataset}")
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=None,
        load_best_model_at_end=True,
        metric_for_best_model="f1_score",
        greater_is_better=True,
        report_to=None,  # 禁用wandb等日志记录
        dataloader_num_workers=4,
    )
    
    # 初始化训练器
    trainer = OSMDetTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset['train'],
        eval_dataset=processed_dataset['validation'],
        compute_metrics=compute_metrics
    )
    
    # 开始训练
    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"Training samples: {len(processed_dataset['train'])}")
    print(f"Validation samples: {len(processed_dataset['validation'])}")
    print(f"Test samples: {len(processed_dataset['test'])}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    
    try:
        trainer.train()
        
        # 保存最终模型
        final_model_path = os.path.join(args.output_dir, "final_model")
        trainer.save_model(final_model_path)
        print(f"Final model saved to {final_model_path}")
        
        print("=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        return

if __name__ == "__main__":
    main()