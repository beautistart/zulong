# 增强型三级记忆架构 - 生产环境测试脚本（通过控制台）

"""
通过在运行中的控制台输入测试指令来测试增强型三级记忆架构

测试步骤:
1. 复制以下测试指令
2. 粘贴到运行中的调试控制台 (Terminal#14)
3. 观察输出结果
"""

test_commands = """
# ========== 测试 1: 动态容量管理 ==========
from zulong.memory.episodic_memory import EpisodicMemory
em = EpisodicMemory()
await em._calculate_dynamic_capacity()
print(f"✓ 最大记忆轮次：{em.max_episodes}")
print(f"✓ Token 预算：{em.max_tokens_reserved}")
print(f"✓ 估算每轮 tokens: {em.estimated_average_turn_tokens}")

# ========== 测试 2: 对话存储和摘要生成 ==========
import time
test_dialogues = [
    ("AI MAX 395 是什么？", "AI MAX 395 是一款高性能处理器，采用 5nm 工艺，集成 128 核 GPU，支持 DDR5 内存。"),
    ("如何安装 CPU 散热器？", "安装 CPU 散热器的步骤：1. 清洁表面 2. 涂抹硅脂 3. 安装散热器 4. 连接风扇电源"),
    ("AI MAX 395 多少钱？", "AI MAX 395 的价格约为 2999 元，不同渠道可能有所差异。"),
    ("DDR5 内存有什么优势？", "DDR5 内存相比 DDR4 有以下优势：1. 更高频率 2. 更低功耗 3. 更大带宽"),
    ("量子计算是什么？", "量子计算是基于量子力学原理的计算范式，利用量子比特的叠加和纠缠特性进行计算。"),
]

start_time = time.time()
for i, (user_input, ai_response) in enumerate(test_dialogues, 1):
    result = await em.store_episode(user_input, ai_response)
    print(f"{i}. 存储对话：episode_id={result['episode_id']}, 摘要={result['summary'][:50]}...")

elapsed = time.time() - start_time
print(f"✓ 存储 {len(test_dialogues)} 轮对话，总耗时：{elapsed*1000:.2f}ms，平均：{elapsed/len(test_dialogues)*1000:.2f}ms/轮")

# ========== 测试 3: 等待异步复盘 ==========
print("等待异步复盘完成（3 秒）...")
import asyncio
await asyncio.sleep(3)

# 检查摘要更新
semantic_count = sum(1 for m in em._episode_index.values() if m.get('summary_type') == 'semantic')
print(f"✓ 已更新 {semantic_count}/{len(em._episode_index)} 条语义摘要")

# ========== 测试 4: 记忆检索 ==========
query = "处理器相关信息"
results = await em.search_by_summary(query, top_k=3, time_window=7200)
print(f"检索查询：'{query}'")
print(f"✓ 检索到 {len(results)} 条结果")
for i, result in enumerate(results, 1):
    print(f"{i}. Episode {result['episode_id']}: {result['summary'][:60]}... (相似度：{result['similarity']:.3f})")

# ========== 测试 5: 分级读取 ==========
if results:
    episode_id = results[0]['episode_id']
    print(f"读取 Episode {episode_id} 的详情...")
    full_dialogue = await em.get_full_dialogue(episode_id)
    if full_dialogue:
        print(f"✓ 用户问：{full_dialogue['user']}")
        print(f"✓ AI 答：{full_dialogue['ai'][:100]}...")
        print("✅ 分级读取成功")
    else:
        print("⚠️ 无法读取详情")

# ========== 测试汇总 ==========
print("\\n" + "="*60)
print("  增强型三级记忆架构 - 生产环境测试报告")
print("="*60)
print(f"✓ 动态容量管理：max_episodes={em.max_episodes}, max_tokens_reserved={em.max_tokens_reserved}")
print(f"✓ 对话存储：存储 {len(test_dialogues)} 轮，平均 {elapsed/len(test_dialogues)*1000:.2f}ms/轮")
print(f"✓ 异步复盘：已更新 {semantic_count}/{len(em._episode_index)} 条")
print(f"✓ 记忆检索：检索到 {len(results)} 条结果")
print(f"✓ 分级读取：{'成功' if results and full_dialogue else '部分功能待完善'}")
print("="*60)
"""

print(__doc__)
print("\n" + "="*80)
print("请按以下步骤操作:")
print("="*80)
print("1. 打开运行中的调试控制台 (Terminal#14)")
print("2. 复制下方的测试指令代码块")
print("3. 粘贴到控制台中并回车执行")
print("4. 观察输出结果")
print("="*80)
print("\n测试指令代码:")
print("="*80)
print(test_commands)
print("="*80)
