# zulong/events - 事件持久化模块

from .event_store import EventStore, get_event_store

__all__ = ["EventStore", "get_event_store"]
