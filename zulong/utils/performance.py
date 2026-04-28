# File: zulong/utils/performance.py
# 性能分析工具

import time


class PerformanceTimer:
    """性能计时器"""
    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.steps = {}
    
    def start(self):
        self.start_time = time.time()
        print(f"\n{'='*80}")
        print(f"[Timer] {self.name} 开始")
        print(f"{'='*80}")
        return self
    
    def step(self, step_name):
        current = time.time()
        if self.start_time:
            total = current - self.start_time
            self.steps[step_name] = total
            print(f"[Timer] 步骤：{step_name} = {total*1000:.1f}ms = {total:.2f}秒")
        return self
    
    def end(self):
        if self.start_time:
            total = time.time() - self.start_time
            print(f"\n{'='*80}")
            print(f"[Timer] {self.name} 完成，总耗时：{total*1000:.1f}ms = {total:.2f}秒")
            print(f"{'='*80}")
            if self.steps:
                print(f"\n详细步骤:")
                for name, t in self.steps.items():
                    print(f"  {name}: {t*1000:.1f}ms ({t:.2f}秒)")
        return self
