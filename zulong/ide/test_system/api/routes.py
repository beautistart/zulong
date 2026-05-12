from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from test_system.definition.enums import TestCaseType
from test_system.engine.executor import TestExecutor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/test", tags=["test"])

_executor: Optional[TestExecutor] = None


def set_executor(executor: TestExecutor) -> None:
    global _executor
    _executor = executor


@router.post("/execute")
async def execute_test(test_case_id: str, config_overrides: Optional[dict] = None):
    if not _executor:
        raise HTTPException(status_code=503, detail="测试系统未初始化")
    store = _executor._store
    tc = await store.get_test_case(test_case_id)
    if not tc:
        raise HTTPException(status_code=404, detail=f"测试用例不存在: {test_case_id}")
    run = await _executor.start_test(tc, config_overrides)
    return run.to_dict()


@router.post("/execute/{execution_id}/stop")
async def stop_test(execution_id: str):
    if not _executor:
        raise HTTPException(status_code=503, detail="测试系统未初始化")
    interrupt = await _executor.stop_test(execution_id)
    if not interrupt:
        raise HTTPException(status_code=404, detail=f"未找到运行中的测试: {execution_id}")
    return {"execution_id": execution_id, "interrupt_point": interrupt.to_dict()}


@router.post("/execute/{execution_id}/resume")
async def resume_test(execution_id: str, snapshot_id: Optional[str] = None):
    if not _executor:
        raise HTTPException(status_code=503, detail="测试系统未初始化")
    try:
        run = await _executor.resume_test(execution_id, snapshot_id)
        return run.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/cases")
async def list_cases(
    type: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
):
    if not _executor:
        raise HTTPException(status_code=503, detail="测试系统未初始化")
    cases = await _executor._store.list_test_cases(type_filter=type, page=page, size=size)
    return {"items": cases, "page": page, "size": size}


@router.get("/execute/{execution_id}")
async def get_execution(execution_id: str):
    if not _executor:
        raise HTTPException(status_code=503, detail="测试系统未初始化")
    run = _executor.get_run(execution_id)
    if not run:
        run_data = await _executor._store.get_execution(execution_id)
        if not run_data:
            raise HTTPException(status_code=404, detail=f"执行记录不存在: {execution_id}")
        return run_data
    steps = await _executor._store.get_step_results(execution_id)
    result = run.to_dict()
    result["stored_steps"] = steps
    return result


@router.get("/history")
async def get_history(
    days: int = Query(30, ge=1, le=365),
    test_case_id: Optional[str] = Query(None),
):
    if not _executor:
        raise HTTPException(status_code=503, detail="测试系统未初始化")
    history = await _executor._store.get_history(days=days, test_case_id=test_case_id)
    return {"items": history, "days": days}


@router.get("/logs")
async def get_logs(
    execution_id: Optional[str] = Query(None),
    level: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    size: int = Query(200, ge=1, le=1000),
):
    if not _executor:
        raise HTTPException(status_code=503, detail="测试系统未初始化")
    logs = await _executor._log.query(execution_id=execution_id, level=level, page=page, size=size)
    return {"items": logs, "page": page, "size": size}
