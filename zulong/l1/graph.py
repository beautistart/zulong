# File: zulong/l1/graph.py
# 组装 L1 上半部分子图

from langgraph.graph import StateGraph
from zulong.state import ZulongState
from zulong.l1.sensors import MockSensorSource
from zulong.l1.nodes import visual_analyze_node, reflex_logic_node, scheduler_logic_node

# 创建传感器源实例
sensor_source = MockSensorSource()


def ingest_sensors(state: ZulongState) -> dict:
    """传感器数据摄入节点
    
    从模拟传感器源获取最新数据并更新状态
    
    Args:
        state: 系统状态
    
    Returns:
        包含更新后传感器数据的字典
    """
    # 获取最新传感器数据
    sensor_data = sensor_source.get_latest_data()
    print(f"[Ingest Sensors] IMU: {sensor_data['imu_status']}, Frame: {sensor_data['camera_frame_ref']}")
    
    return {'sensors': sensor_data}


# 构建 L1 上半部分子图
def build_l1_upper_graph():
    """构建 L1 上半部分子图
    
    Returns:
        StateGraph: L1 上半部分子图
    """
    # 创建状态图
    graph = StateGraph(ZulongState)
    
    # 添加节点
    graph.add_node('ingest_sensors', ingest_sensors)
    graph.add_node('visual_analyze', visual_analyze_node)
    graph.add_node('reflex_logic', reflex_logic_node)
    graph.add_node('scheduler', scheduler_logic_node)
    
    # 添加边（线性连接）
    graph.add_edge('ingest_sensors', 'visual_analyze')
    graph.add_edge('visual_analyze', 'reflex_logic')
    graph.add_edge('reflex_logic', 'scheduler')
    graph.add_edge('scheduler', '__end__')
    
    # 设置入口点
    graph.set_entry_point('ingest_sensors')
    
    return graph


# 导出 L1 上半部分子图
l1_upper_graph = build_l1_upper_graph()
