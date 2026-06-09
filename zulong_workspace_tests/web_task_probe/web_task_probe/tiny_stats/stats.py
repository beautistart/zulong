"""tiny_stats — 轻量统计工具，仅使用 Python 标准库。"""

from collections import Counter
from typing import List, Optional, Union


def safe_parse_numbers(text: str) -> List[float]:
    """从文本中安全解析数字列表。

    按空白/逗号/分号/换行分割文本，尝试将每个 token 转为 float。
    无法转换的 token 静默跳过。

    Args:
        text: 包含数字的文本字符串。

    Returns:
        解析成功的浮点数列表；若没有合法数字则返回空列表。
    """
    if not text or not text.strip():
        return []

    # 统一分隔符：空白、逗号、分号、换行
    import re
    tokens = re.split(r'[\s,;]+', text.strip())
    numbers: List[float] = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        try:
            numbers.append(float(token))
        except (ValueError, OverflowError):
            # 非法数字静默跳过
            continue
    return numbers


def mean(values: List[Union[int, float]]) -> Optional[float]:
    """计算算术平均值。

    Args:
        values: 数字列表。

    Returns:
        平均值；空列表返回 None。
    """
    if not values:
        return None
    return sum(values) / len(values)


def median(values: List[Union[int, float]]) -> Optional[float]:
    """计算中位数。

    对偶数长度列表返回中间两个数的平均值。

    Args:
        values: 数字列表。

    Returns:
        中位数；空列表返回 None。
    """
    if not values:
        return None

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2

    if n % 2 == 1:
        # 奇数：返回正中间
        return sorted_vals[mid]
    else:
        # 偶数：返回中间两个的平均值
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0


def mode(values: List[Union[int, float]]) -> Optional[float]:
    """计算众数。

    出现次数最多的值。若多个值并列最高频次，返回其中数值最小的。

    Args:
        values: 数字列表。

    Returns:
        众数；空列表返回 None。
    """
    if not values:
        return None

    counter = Counter(values)
    # 按 (-频次, 数值) 排序，取第一个即为频次最高且数值最小
    # most_common 在频次相同时不保证顺序，所以手动排序
    mode_val = min(counter.items(), key=lambda item: (-item[1], item[0]))
    return mode_val[0]
