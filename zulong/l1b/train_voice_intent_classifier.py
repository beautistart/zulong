"""
语音意图分类器训练脚本

训练一个 3 类分类头（TEXT_ONLY / AUTO_TTS / FORCED_TTS）在冻结的 ALBERT-tiny-chinese 基础模型上。

使用方法:
    python zulong/l1b/train_voice_intent_classifier.py

输出:
    - models/albert-tiny-chinese/voice_intent_head.pt - 分类头权重
    - models/albert-tiny-chinese/voice_intent_training_meta.json - 训练元数据
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# 训练数据
# ============================================================================

# 标签映射
LABEL2ID = {
    "TEXT_ONLY": 0,
    "AUTO_TTS": 1,
    "FORCED_TTS": 2,
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# 种子训练数据（约 150 条）
TRAINING_DATA: List[Tuple[str, str]] = [
    # FORCED_TTS - 用户明确要求语音回复
    ("读给我听", "FORCED_TTS"),
    ("用语音回答", "FORCED_TTS"),
    ("播报一下", "FORCED_TTS"),
    ("语音回复我", "FORCED_TTS"),
    ("说给我听", "FORCED_TTS"),
    ("念一下这段文字", "FORCED_TTS"),
    ("用声音告诉我", "FORCED_TTS"),
    ("语音告诉我", "FORCED_TTS"),
    ("用语音说", "FORCED_TTS"),
    ("用语音讲", "FORCED_TTS"),
    ("给我讲", "FORCED_TTS"),
    ("跟我说", "FORCED_TTS"),
    ("跟我讲", "FORCED_TTS"),
    ("大声说", "FORCED_TTS"),
    ("大声讲", "FORCED_TTS"),
    ("读一下", "FORCED_TTS"),
    ("讲出来", "FORCED_TTS"),
    ("说一遍", "FORCED_TTS"),
    ("讲一遍", "FORCED_TTS"),
    ("说一次", "FORCED_TTS"),
    ("讲一次", "FORCED_TTS"),
    ("念给我听", "FORCED_TTS"),
    ("说给我听", "FORCED_TTS"),
    ("讲给我听", "FORCED_TTS"),
    ("读给我", "FORCED_TTS"),
    ("念给我", "FORCED_TTS"),
    ("说给我", "FORCED_TTS"),
    ("讲给我", "FORCED_TTS"),
    ("语音跟我说", "FORCED_TTS"),
    ("语音讲给我听", "FORCED_TTS"),
    ("把重庆明天的天气情况读给我听", "FORCED_TTS"),
    ("用语音给我打个招呼", "FORCED_TTS"),
    ("用语音给我读一下这段代码", "FORCED_TTS"),
    ("播报今天的新闻", "FORCED_TTS"),
    ("用语音回答这个问题", "FORCED_TTS"),
    ("请你用语音告诉我", "FORCED_TTS"),
    ("我想听语音回复", "FORCED_TTS"),
    ("请用声音回答", "FORCED_TTS"),
    ("能不能用语音告诉我", "FORCED_TTS"),
    ("麻烦用语音念一下", "FORCED_TTS"),
    ("请朗读这段文字", "FORCED_TTS"),
    ("用语音播报", "FORCED_TTS"),
    ("语音念一下", "FORCED_TTS"),
    ("读出来", "FORCED_TTS"),
    ("念出来", "FORCED_TTS"),
    ("说出来", "FORCED_TTS"),
    ("讲出来", "FORCED_TTS"),
    ("用语音", "FORCED_TTS"),
    ("给我说", "FORCED_TTS"),
    ("大声点", "FORCED_TTS"),
    ("语音回答", "FORCED_TTS"),
    ("语音回复", "FORCED_TTS"),
    ("用语音说一遍", "FORCED_TTS"),
    ("用语音讲一遍", "FORCED_TTS"),
    ("把这段话读给我听", "FORCED_TTS"),
    ("把这个故事念给我听", "FORCED_TTS"),
    ("把新闻播报给我听", "FORCED_TTS"),
    ("用语音告诉我答案", "FORCED_TTS"),
    ("请语音回复", "FORCED_TTS"),
    ("我想听语音", "FORCED_TTS"),
    ("给我播放一下", "FORCED_TTS"),
    
    # TEXT_ONLY - 用户仅需要文字或明确不要语音
    ("我不想听语音", "TEXT_ONLY"),
    ("不用语音，文字就行", "TEXT_ONLY"),
    ("文字回复就行", "TEXT_ONLY"),
    ("帮我写个代码", "TEXT_ONLY"),
    ("今天天气怎么样", "TEXT_ONLY"),
    ("关闭语音输出", "TEXT_ONLY"),
    ("不要用语音回答", "TEXT_ONLY"),
    ("文字回答就好", "TEXT_ONLY"),
    ("不需要语音", "TEXT_ONLY"),
    ("只要文字", "TEXT_ONLY"),
    ("给我文字回复", "TEXT_ONLY"),
    ("我不想听声音", "TEXT_ONLY"),
    ("不用声音了", "TEXT_ONLY"),
    ("文字就可以了", "TEXT_ONLY"),
    ("帮我分析一下这段代码", "TEXT_ONLY"),
    ("这个函数是什么意思", "TEXT_ONLY"),
    ("搜索一下这个关键词", "TEXT_ONLY"),
    ("打开摄像头", "TEXT_ONLY"),
    ("设置音量", "TEXT_ONLY"),
    ("你好祖龙", "TEXT_ONLY"),
    ("在吗", "TEXT_ONLY"),
    ("帮我写一个Python函数", "TEXT_ONLY"),
    ("解释一下这段代码", "TEXT_ONLY"),
    ("查找文件", "TEXT_ONLY"),
    ("运行这个程序", "TEXT_ONLY"),
    ("读取文件内容", "TEXT_ONLY"),
    ("分析代码逻辑", "TEXT_ONLY"),
    ("搜索相关信息", "TEXT_ONLY"),
    ("配置系统参数", "TEXT_ONLY"),
    ("停止当前任务", "TEXT_ONLY"),
    ("开始新任务", "TEXT_ONLY"),
    ("修改设置", "TEXT_ONLY"),
    ("今天天气如何", "TEXT_ONLY"),
    ("帮我查一下", "TEXT_ONLY"),
    ("这段代码有问题吗", "TEXT_ONLY"),
    ("重构这个函数", "TEXT_ONLY"),
    ("生成报告", "TEXT_ONLY"),
    ("导出结果", "TEXT_ONLY"),
    ("查看当前状态", "TEXT_ONLY"),
    ("显示详细信息", "TEXT_ONLY"),
    ("不需要语音回复", "TEXT_ONLY"),
    ("别用语音了", "TEXT_ONLY"),
    ("语音不方便", "TEXT_ONLY"),
    ("看文字就好", "TEXT_ONLY"),
    ("不想听", "TEXT_ONLY"),
    ("只要文字回复", "TEXT_ONLY"),
    ("纯文字就行", "TEXT_ONLY"),
    ("不要声音", "TEXT_ONLY"),
    ("关掉语音", "TEXT_ONLY"),
    ("语音太吵了", "TEXT_ONLY"),
    ("安静点，文字就行", "TEXT_ONLY"),
    
    # AUTO_TTS - 隐式语音请求（通常由语音输入事件触发）
    # 注意：这类数据在实际训练中较少，主要依靠事件类型判断
    # 这里添加一些可能触发 AUTO_TTS 的文本模式
    ("听到了吗", "AUTO_TTS"),
    ("你能听到我说话吗", "AUTO_TTS"),
    ("麦克风测试", "AUTO_TTS"),
    ("语音输入测试", "AUTO_TTS"),
    ("我在说话", "AUTO_TTS"),
    ("听得见吗", "AUTO_TTS"),
    ("声音清楚吗", "AUTO_TTS"),
    ("语音识别准吗", "AUTO_TTS"),
    ("测试语音", "AUTO_TTS"),
    ("语音对话", "AUTO_TTS"),
    ("用语音聊天", "AUTO_TTS"),
    ("语音交流", "AUTO_TTS"),
    ("语音交互", "AUTO_TTS"),
]

# 修正数据格式（上面有一条格式错误）
TRAINING_DATA = [
    ("读给我听", "FORCED_TTS"),
    ("用语音回答", "FORCED_TTS"),
    ("播报一下", "FORCED_TTS"),
    ("语音回复我", "FORCED_TTS"),
    ("说给我听", "FORCED_TTS"),
    ("念一下这段文字", "FORCED_TTS"),
    ("用声音告诉我", "FORCED_TTS"),
    ("语音告诉我", "FORCED_TTS"),
    ("用语音说", "FORCED_TTS"),
    ("用语音讲", "FORCED_TTS"),
    ("给我讲", "FORCED_TTS"),
    ("跟我说", "FORCED_TTS"),
    ("跟我讲", "FORCED_TTS"),
    ("大声说", "FORCED_TTS"),
    ("大声讲", "FORCED_TTS"),
    ("读一下", "FORCED_TTS"),
    ("讲出来", "FORCED_TTS"),
    ("说一遍", "FORCED_TTS"),
    ("讲一遍", "FORCED_TTS"),
    ("说一次", "FORCED_TTS"),
    ("讲一次", "FORCED_TTS"),
    ("念给我听", "FORCED_TTS"),
    ("说给我听", "FORCED_TTS"),
    ("讲给我听", "FORCED_TTS"),
    ("读给我", "FORCED_TTS"),
    ("念给我", "FORCED_TTS"),
    ("说给我", "FORCED_TTS"),
    ("讲给我", "FORCED_TTS"),
    ("语音跟我说", "FORCED_TTS"),
    ("语音讲给我听", "FORCED_TTS"),
    ("把重庆明天的天气情况读给我听", "FORCED_TTS"),
    ("用语音给我打个招呼", "FORCED_TTS"),
    ("用语音给我读一下这段代码", "FORCED_TTS"),
    ("播报今天的新闻", "FORCED_TTS"),
    ("用语音回答这个问题", "FORCED_TTS"),
    ("请你用语音告诉我", "FORCED_TTS"),
    ("我想听语音回复", "FORCED_TTS"),
    ("请用声音回答", "FORCED_TTS"),
    ("能不能用语音告诉我", "FORCED_TTS"),
    ("麻烦用语音念一下", "FORCED_TTS"),
    ("请朗读这段文字", "FORCED_TTS"),
    ("用语音播报", "FORCED_TTS"),
    ("语音念一下", "FORCED_TTS"),
    ("读出来", "FORCED_TTS"),
    ("念出来", "FORCED_TTS"),
    ("说出来", "FORCED_TTS"),
    ("讲出来", "FORCED_TTS"),
    ("用语音", "FORCED_TTS"),
    ("给我说", "FORCED_TTS"),
    ("大声点", "FORCED_TTS"),
    ("语音回答", "FORCED_TTS"),
    ("语音回复", "FORCED_TTS"),
    ("用语音说一遍", "FORCED_TTS"),
    ("用语音讲一遍", "FORCED_TTS"),
    ("把这段话读给我听", "FORCED_TTS"),
    ("把这个故事念给我听", "FORCED_TTS"),
    ("把新闻播报给我听", "FORCED_TTS"),
    ("用语音告诉我答案", "FORCED_TTS"),
    ("请语音回复", "FORCED_TTS"),
    ("我想听语音", "FORCED_TTS"),
    ("给我播放一下", "FORCED_TTS"),
    ("我不想听语音", "TEXT_ONLY"),
    ("不用语音，文字就行", "TEXT_ONLY"),
    ("文字回复就行", "TEXT_ONLY"),
    ("帮我写个代码", "TEXT_ONLY"),
    ("今天天气怎么样", "TEXT_ONLY"),
    ("关闭语音输出", "TEXT_ONLY"),
    ("不要用语音回答", "TEXT_ONLY"),
    ("文字回答就好", "TEXT_ONLY"),
    ("不需要语音", "TEXT_ONLY"),
    ("只要文字", "TEXT_ONLY"),
    ("给我文字回复", "TEXT_ONLY"),
    ("我不想听声音", "TEXT_ONLY"),
    ("不用声音了", "TEXT_ONLY"),
    ("文字就可以了", "TEXT_ONLY"),
    ("帮我分析一下这段代码", "TEXT_ONLY"),
    ("这个函数是什么意思", "TEXT_ONLY"),
    ("搜索一下这个关键词", "TEXT_ONLY"),
    ("打开摄像头", "TEXT_ONLY"),
    ("设置音量", "TEXT_ONLY"),
    ("你好祖龙", "TEXT_ONLY"),
    ("在吗", "TEXT_ONLY"),
    ("帮我写一个Python函数", "TEXT_ONLY"),
    ("解释一下这段代码", "TEXT_ONLY"),
    ("查找文件", "TEXT_ONLY"),
    ("运行这个程序", "TEXT_ONLY"),
    ("读取文件内容", "TEXT_ONLY"),
    ("分析代码逻辑", "TEXT_ONLY"),
    ("搜索相关信息", "TEXT_ONLY"),
    ("配置系统参数", "TEXT_ONLY"),
    ("停止当前任务", "TEXT_ONLY"),
    ("开始新任务", "TEXT_ONLY"),
    ("修改设置", "TEXT_ONLY"),
    ("今天天气如何", "TEXT_ONLY"),
    ("帮我查一下", "TEXT_ONLY"),
    ("这段代码有问题吗", "TEXT_ONLY"),
    ("重构这个函数", "TEXT_ONLY"),
    ("生成报告", "TEXT_ONLY"),
    ("导出结果", "TEXT_ONLY"),
    ("查看当前状态", "TEXT_ONLY"),
    ("显示详细信息", "TEXT_ONLY"),
    ("不需要语音回复", "TEXT_ONLY"),
    ("别用语音了", "TEXT_ONLY"),
    ("语音不方便", "TEXT_ONLY"),
    ("看文字就好", "TEXT_ONLY"),
    ("不想听", "TEXT_ONLY"),
    ("只要文字回复", "TEXT_ONLY"),
    ("纯文字就行", "TEXT_ONLY"),
    ("不要声音", "TEXT_ONLY"),
    ("关掉语音", "TEXT_ONLY"),
    ("语音太吵了", "TEXT_ONLY"),
    ("安静点，文字就行", "TEXT_ONLY"),
    ("听到了吗", "AUTO_TTS"),
    ("你能听到我说话吗", "AUTO_TTS"),
    ("麦克风测试", "AUTO_TTS"),
    ("语音输入测试", "AUTO_TTS"),
    ("我在说话", "AUTO_TTS"),
    ("听得见吗", "AUTO_TTS"),
    ("声音清楚吗", "AUTO_TTS"),
    ("语音识别准吗", "AUTO_TTS"),
    ("测试语音", "AUTO_TTS"),
    ("语音对话", "AUTO_TTS"),
    ("用语音聊天", "AUTO_TTS"),
    ("语音交流", "AUTO_TTS"),
    ("语音交互", "AUTO_TTS"),
]


# ============================================================================
# 数据集
# ============================================================================

class VoiceIntentDataset(Dataset):
    """语音意图数据集"""
    
    def __init__(self, texts: List[str], labels: List[int]):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


# ============================================================================
# 训练函数
# ============================================================================

def train_voice_intent_classifier(
    model_path: str = "./models/albert-tiny-chinese",
    output_dir: str = None,
    batch_size: int = 16,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    device: str = "cpu",
):
    """
    训练语音意图分类器
    
    Args:
        model_path: ALBERT 基础模型路径
        output_dir: 输出目录（默认与 model_path 相同）
        batch_size: 批次大小
        epochs: 训练轮数
        learning_rate: 学习率
        device: 训练设备
    """
    if output_dir is None:
        output_dir = model_path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"[训练] 配置:")
    logger.info(f"  - 基础模型: {model_path}")
    logger.info(f"  - 输出目录: {output_dir}")
    logger.info(f"  - 批次大小: {batch_size}")
    logger.info(f"  - 训练轮数: {epochs}")
    logger.info(f"  - 学习率: {learning_rate}")
    logger.info(f"  - 设备: {device}")
    
    # 1. 准备数据
    texts = [item[0] for item in TRAINING_DATA]
    labels = [LABEL2ID[item[1]] for item in TRAINING_DATA]
    
    # 统计类别分布
    from collections import Counter
    label_counts = Counter(labels)
    logger.info(f"[训练] 数据集大小: {len(TRAINING_DATA)}")
    logger.info(f"[训练] 类别分布:")
    for label_id, count in sorted(label_counts.items()):
        logger.info(f"  - {ID2LABEL[label_id]}: {count} 条")
    
    # 2. 加载 ALBERT 基础模型（冻结参数）
    logger.info(f"[训练] 加载 ALBERT 基础模型...")
    from transformers import AutoModel, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    albert_model = AutoModel.from_pretrained(model_path)
    albert_model.to(device)
    albert_model.eval()
    
    # 冻结 ALBERT 参数
    for param in albert_model.parameters():
        param.requires_grad = False
    
    logger.info(f"[训练] ALBERT 模型已加载并冻结（{sum(p.numel() for p in albert_model.parameters()) / 1e6:.1f}M 参数）")
    
    # 3. 初始化分类头（768 → 3）
    # ALBERT-tiny 的 hidden_size 通常是 312 或 768，我们从模型获取
    hidden_size = albert_model.config.hidden_size
    classification_head = torch.nn.Linear(hidden_size, len(LABEL2ID)).to(device)
    
    # 使用 Xavier 初始化
    torch.nn.init.xavier_uniform_(classification_head.weight)
    torch.nn.init.zeros_(classification_head.bias)
    
    logger.info(f"[训练] 分类头初始化: Linear({hidden_size} → {len(LABEL2ID)})")
    
    # 4. 编码所有文本（ALBERT 冻结，可以预先编码）
    logger.info(f"[训练] 预编码文本...")
    encoded_inputs = tokenizer(
        texts,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    
    with torch.no_grad():
        outputs = albert_model(
            input_ids=encoded_inputs["input_ids"].to(device),
            attention_mask=encoded_inputs["attention_mask"].to(device),
        )
        pooler_outputs = outputs.pooler_output
    
    logger.info(f"[训练] 预编码完成: {pooler_outputs.shape}")
    
    # 创建数据集和数据加载器
    dataset = VoiceIntentDataset(
        texts=[None] * len(texts),  # 不需要文本，使用预编码的特征
        labels=labels,
    )
    
    # 直接使用 tensor 数据集
    from torch.utils.data import TensorDataset
    tensor_dataset = TensorDataset(pooler_outputs, torch.tensor(labels))
    data_loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)
    
    # 5. 训练分类头
    optimizer = torch.optim.Adam(classification_head.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    logger.info(f"[训练] 开始训练...")
    classification_head.train()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_pooler, batch_labels in tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_pooler = batch_pooler.to(device)
            batch_labels = batch_labels.to(device)
            
            # 前向传播
            logits = classification_head(batch_pooler)
            loss = criterion(logits, batch_labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item() * batch_pooler.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == batch_labels).sum().item()
            total += batch_pooler.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        logger.info(
            f"[训练] Epoch {epoch+1}/{epochs} | "
            f"Loss: {avg_loss:.4f} | Acc: {accuracy:.4f} ({correct}/{total})"
        )
    
    # 6. 评估
    logger.info(f"[训练] 训练完成，评估模型...")
    classification_head.eval()
    
    with torch.no_grad():
        all_logits = classification_head(pooler_outputs.to(device))
        all_preds = torch.argmax(all_logits, dim=-1)
        all_probs = F.softmax(all_logits, dim=-1)
        
        accuracy = (all_preds.cpu() == torch.tensor(labels)).sum().item() / len(labels)
        logger.info(f"[训练] 训练集准确率: {accuracy:.4f}")
        
        # 混淆矩阵
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(labels, all_preds.cpu().numpy())
        logger.info(f"[训练] 混淆矩阵:")
        for i, row in enumerate(cm):
            logger.info(f"  {ID2LABEL[i]}: {row.tolist()}")
        
        # 分类报告
        report = classification_report(
            labels, all_preds.cpu().numpy(),
            target_names=[ID2LABEL[i] for i in range(len(ID2LABEL))],
            zero_division=0,
        )
        logger.info(f"[训练] 分类报告:\n{report}")
    
    # 7. 保存分类头权重
    head_weights = {
        "weight": classification_head.weight.cpu(),
        "bias": classification_head.bias.cpu(),
    }
    
    head_path = output_path / "voice_intent_head.pt"
    torch.save(head_weights, str(head_path))
    logger.info(f"[训练] 分类头权重已保存: {head_path}")
    
    # 8. 保存训练元数据
    meta = {
        "model": "albert-tiny-chinese",
        "hidden_size": hidden_size,
        "num_labels": len(LABEL2ID),
        "labels": list(LABEL2ID.keys()),
        "label2id": LABEL2ID,
        "training_samples": len(TRAINING_DATA),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "final_accuracy": accuracy,
        "final_loss": avg_loss,
        "device": device,
    }
    
    meta_path = output_path / "voice_intent_training_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info(f"[训练] 训练元数据已保存: {meta_path}")
    
    logger.info(f"[训练] ✅ 训练完成！")
    logger.info(f"[训练] 分类头: {head_path}")
    logger.info(f"[训练] 元数据: {meta_path}")
    
    return accuracy


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="训练语音意图分类器")
    parser.add_argument("--model-path", type=str, default="./models/albert-tiny-chinese",
                        help="ALBERT 基础模型路径")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录（默认与 model-path 相同）")
    parser.add_argument("--batch-size", type=int, default=16, help="批次大小")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="训练设备")
    
    args = parser.parse_args()
    
    accuracy = train_voice_intent_classifier(
        model_path=args.model_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device,
    )
    
    if accuracy < 0.8:
        logger.warning(f"⚠️  训练准确率较低 ({accuracy:.4f})，可能需要更多训练数据或调整超参数")
    else:
        logger.info(f"✅ 训练准确率良好 ({accuracy:.4f})")
