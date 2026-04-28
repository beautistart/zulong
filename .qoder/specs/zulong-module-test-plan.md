# Zulong System Module Test Plan

## Context

Previous sessions implemented P0~P4 code bridges connecting MemoryGraph, AttentionWindow, Thought Navigation, and Task Orchestration. All code paths were verified working via end-to-end WebSocket tests (BFS valid=2/2, Hebbian learning confirmed, TaskGraph sync 2->9 nodes). 

The user now requests: **systematic testing of each module independently first, then cross-module integration tests, ordered by relevance**. This ensures each module's core logic is correct in isolation before testing inter-module collaboration.

## Test Architecture

```
tests/
  conftest.py                          # Shared fixtures (singleton reset, temp dirs)
  test_layer1_memory_graph.py          # L1: MemoryGraph CRUD + labels + BFS + Hebbian
  test_layer1_task_graph.py            # L1: TaskGraph operations + serialization
  test_layer1_circuit_breaker.py       # L1: CircuitBreaker 6-signal evaluation
  test_layer1_attention_window.py      # L1: AttentionWindow modes + scoring
  test_layer1_task_suspension.py       # L1: TaskSuspension serialize/deserialize
  test_layer2_graph_adapters.py        # L2: TaskGraphAdapter + DialogueAdapter sync
  test_layer2_attention_memory.py      # L2: AttentionWindow + MemoryGraph boost
  test_layer3_e2e_complex_task.py      # L3: Full WebSocket end-to-end
```

## Critical Files

| Module | Source File | Key Classes/Functions |
|--------|-----------|----------------------|
| MemoryGraph | `zulong/memory/memory_graph.py` | MemoryGraph, GraphNode, NodeType, EdgeType, Importance, Temperature |
| Graph Adapters | `zulong/memory/graph_adapters.py` | TaskGraphAdapter, DialogueAdapter |
| TaskGraph | `zulong/l2/task_graph.py` | TaskGraph, TaskNode, FileRef, DependencyEdge |
| CircuitBreaker | `zulong/l2/circuit_breaker.py` | ToolCallCircuitBreaker, CircuitBreakerState, ToolCallRecord |
| AttentionWindow | `zulong/l2/attention_window.py` | AttentionWindowManager, AttentionMode, MessageEnvelope |
| TaskSuspension | `zulong/l2/task_suspension.py` | TaskSuspensionManager, SuspendableTaskState |

## Implementation Steps

### Step 0: Create conftest.py with shared fixtures

Create `tests/conftest.py` with:

- **`reset_memory_graph`** (autouse): Reset `MemoryGraph._instance = None` and `MemoryGraph._initialized` before each test
- **`reset_task_suspension`** (autouse): Reset `TaskSuspensionManager._instance = None` and `_initialized`
- **`temp_dir`**: Provide a fresh `tempfile.mkdtemp()` per test, cleaned up after
- **`temp_memory_graph`**: Create isolated MemoryGraph with temp persist_path (depends on singleton reset + temp_dir)
- **`sample_task_graph`**: Pre-built TaskGraph with req + analysis + 3 outline nodes + h_edges + d_edges

### Step 1: test_layer1_memory_graph.py (MemoryGraph independent tests)

**TestGraphNodeBasics** (no MemoryGraph instance needed):
- `test_create_node`: GraphNode creation with all fields
- `test_node_serialize_roundtrip`: to_dict() -> from_dict() preserves all fields
- `test_enum_values`: Verify NodeType(9), EdgeType(7), Importance(6), Temperature(3) enum values

**TestMemoryGraphCRUD** (uses temp_memory_graph fixture):
- `test_add_and_get_node`: add_node returns id, get_node returns correct GraphNode
- `test_has_node`: True for existing, False for non-existent
- `test_remove_node`: Remove node removes it and its edges
- `test_get_nodes_by_type`: Filter by NodeType works correctly
- `test_add_edge`: add_edge between existing nodes, verify has_edge
- `test_add_edge_missing_node`: add_edge with non-existent node returns False
- `test_edge_weight_max`: Re-adding edge uses max(old, new) weight
- `test_remove_edge`: remove_edge works

**TestMultiDimensionalLabels** (uses temp_memory_graph):
- `test_temperature_hot`: Freshly accessed node -> HOT
- `test_temperature_cold`: Node with old last_accessed -> COLD (mock time)
- `test_importance_default_normal`: New node has NORMAL importance
- `test_set_importance`: set_importance changes value
- `test_promote_importance_only_up`: promote_importance won't downgrade
- `test_is_recent`: Within 30min -> True, outside -> False

**TestBFSActivation** (uses temp_memory_graph):
- `test_seed_activation_1`: Single seed -> activation=1.0
- `test_propagation_decay`: Chain A->B->C, verify B_act = 1.0 * weight_AB * decay, C_act further decayed
- `test_max_depth_limit`: Nodes beyond max_depth not activated
- `test_min_activation_cutoff`: Very weak edges don't propagate below threshold
- `test_multi_seed`: Two seeds, both start at 1.0, overlapping neighbors accumulate

**TestHebbianLearning** (uses temp_memory_graph):
- `test_hebbian_increases_weight`: After compute_activations + hebbian_strengthen, edge weight > original
- `test_hebbian_asymptotic`: Weight approaches but never exceeds 1.0
- `test_hebbian_skips_protected`: Protected edges not modified

**TestDecayAndPrune** (uses temp_memory_graph):
- `test_trivial_edge_decays_fast`: TRIVIAL importance edge decays with 6h half-life
- `test_must_remember_never_decays`: MUST_REMEMBER edge unchanged
- `test_weak_edge_removed`: Edge below 0.05 threshold removed after decay

**TestPersistence** (uses temp_memory_graph):
- `test_save_and_reload`: save() -> reset singleton -> reload from same path -> nodes/edges preserved

### Step 2: test_layer1_task_graph.py (TaskGraph independent tests)

**TestTaskGraphBasics**:
- `test_init_with_title`: Title and auto-generated ID
- `test_init_with_custom_id`: Explicit graph_id preserved
- `test_address_format`: address == "tg:{id}"

**TestNodeOperations**:
- `test_add_node`: Add node, get_node returns it
- `test_add_node_with_files`: FileRef correctly stored
- `test_get_nodes_by_status`: Filter pending/in_progress/completed
- `test_get_nodes_by_type`: Filter by type string
- `test_update_node_status`: Status change + result update
- `test_remove_node_cascades`: Removing parent removes all descendants
- `test_remove_req_blocked`: Cannot remove "req" root node

**TestEdgeOperations**:
- `test_add_h_edge`: Hierarchy edge stored correctly
- `test_add_d_edge`: Dependency edge with via/cross fields
- `test_get_dependencies`: Returns correct dependent node IDs
- `test_get_dependents`: Returns nodes that depend on target

**TestTreeNavigation**:
- `test_get_children`: Returns correct child nodes by h_edge
- `test_get_parent`: Returns parent node ID
- `test_get_ancestor_chain`: Full chain from node to root
- `test_get_all_descendants`: BFS all descendants
- `test_get_leaf_nodes`: Leaf nodes (no children, not "req")
- `test_depth_to_type`: 0=requirement, 1=analysis, 2=outline, 3=task, 4+=subtask

**TestSerialization**:
- `test_serialize_deserialize_roundtrip`: Full graph -> serialize -> deserialize -> compare
- `test_deserialize_handles_bad_edges`: Edges referencing missing nodes skipped
- `test_parallel_groups_preserved`: parallel_groups survive roundtrip

### Step 3: test_layer1_circuit_breaker.py (CircuitBreaker independent tests)

**TestCircuitBreakerInit**:
- `test_default_config`: Verify all default parameter values
- `test_custom_config`: Override specific params, verify applied
- `test_disabled_mode`: enabled=False -> safety_hard_cap=10

**TestSignalRepetition** (signal 1):
- `test_3_identical_calls_red`: 3 identical (name+params_hash) -> RED
- `test_2_identical_calls_yellow`: 2 identical -> YELLOW
- `test_different_calls_green`: Different calls -> GREEN

**TestSignalPatternLoop** (signal 2):
- `test_same_tool_exceeds_red_count`: 7+ calls of same tool -> RED
- `test_same_tool_exceeds_yellow_count`: 5+ calls -> YELLOW
- `test_planning_tool_exempt`: Planning tools not flagged in planning mode
- `test_similar_search_queries_red`: Search queries with Jaccard > 0.7 -> RED

**TestSignalInfoGain** (signal 3):
- `test_identical_results_red`: 3 consecutive identical result_hash -> RED
- `test_empty_results_yellow`: 3 consecutive empty/short results -> YELLOW
- `test_diverse_results_green`: Varying results -> GREEN

**TestSignalContextPressure** (signal 4):
- `test_red_at_90_percent`: Token ratio >= 0.90 -> RED
- `test_yellow_at_75_percent`: Token ratio >= 0.75 -> YELLOW
- `test_green_below_75`: Below threshold -> GREEN

**TestSignalNoProgress** (signal 6):
- `test_consecutive_info_retrieval_red`: 6+ consecutive info tools without action -> RED
- `test_action_tool_resets_counter`: Action tool in between resets count

**TestEscalation**:
- `test_yellow_to_red_upgrade`: max_yellow_before_red consecutive YELLOWs -> RED
- `test_reset_clears_state`: reset() clears history and counters
- `test_escalate_for_planning`: Verify threshold changes
- `test_reset_to_default`: Verify restore after planning mode

**TestSerializationCB**:
- `test_serialize_deserialize`: State survives roundtrip

### Step 4: test_layer1_attention_window.py (AttentionWindow independent tests)

**TestAttentionModeEnum**:
- `test_mode_values`: GLOBAL="global", FOCUS="focus", SINGLE_CHAIN="single_chain"

**TestTokenEstimation**:
- `test_estimate_tokens_chinese`: Chinese chars * 1.5
- `test_estimate_tokens_english`: English words * 0.75
- `test_estimate_tokens_mixed`: Mixed content
- `test_estimate_message_tokens`: Full message dict with tool_calls

**TestMessageRegistration**:
- `test_register_basic_message`: Envelope created with correct seq/turn
- `test_register_pinned_message`: is_pinned=True preserved
- `test_register_tool_message`: tool_name and node_id stored
- `test_new_tool_group`: group_id increments

**TestModeTransitions**:
- `test_observe_focus_trigger`: recall_memory -> GLOBAL to FOCUS
- `test_observe_single_chain_trigger`: exec_write_file -> FOCUS to SINGLE_CHAIN
- `test_observe_global_force`: submit_final_answer -> any to GLOBAL
- `test_navigate_deeper`: deeper: GLOBAL->FOCUS->SINGLE_CHAIN
- `test_navigate_broader`: broader: SINGLE_CHAIN->FOCUS->GLOBAL
- `test_navigate_jump`: Jump to node with auto-mode selection

**TestScoring**:
- `test_time_decay`: Older messages score lower (0.95^age)
- `test_mode_multiplier_global`: Overview tools score higher
- `test_mode_multiplier_focus`: Current node scores 3.0x
- `test_mode_multiplier_single_chain`: Current node 5.0x, unrelated 0.2x
- `test_memory_boost`: With MemoryGraph node activation -> score boosted

**TestApplyWindow**:
- `test_budget_respected`: Output messages fit within budget
- `test_pinned_always_included`: Pinned messages never evicted
- `test_group_atomicity`: Tool call groups kept or evicted together
- `test_eviction_summary_generated`: Summary message for evicted content

### Step 5: test_layer1_task_suspension.py (TaskSuspension independent tests)

**TestSuspendableTaskState**:
- `test_create_state`: All fields populated correctly
- `test_to_dict_excludes_task_graph`: task_graph runtime object removed
- `test_from_dict_roundtrip`: to_dict -> from_dict preserves all fields
- `test_from_dict_deserializes_task_graph`: task_graph_serialized -> TaskGraph instance

**TestTaskSuspensionManager** (async, temp_dir fixture):
- `test_suspend_and_list`: suspend_task creates file, list returns it
- `test_resume_consume`: resume with consume=True deletes file
- `test_resume_no_consume`: resume with consume=False keeps file
- `test_cancel_task`: cancel removes file
- `test_max_suspended_tasks`: Exceeding limit auto-cleans oldest
- `test_cleanup_expired`: Tasks beyond max_age_hours removed
- `test_find_by_description_exact`: Exact substring match
- `test_find_by_description_bigram`: Bigram overlap matching

### Step 6: test_layer2_graph_adapters.py (Cross-module integration)

**TestTaskGraphAdapter** (uses temp_memory_graph + sample_task_graph):
- `test_sync_creates_nodes`: sync() projects TaskGraph nodes to MemoryGraph with "task:{graph_id}/{node_id}" format
- `test_sync_returns_count`: Returns number of synced nodes
- `test_sync_creates_hierarchy_edges`: HIERARCHY edges (protected) created between parent-child
- `test_sync_creates_dependency_edges`: DEPENDENCY edges created
- `test_sync_idempotent`: Second sync() doesn't duplicate, updates existing
- `test_sync_file_refs`: FileRef -> FILE type GraphNode + REFERENCE edge
- `test_incremental_sync_node_add`: Event-driven sync for new node

**TestDialogueAdapter** (uses temp_memory_graph):
- `test_add_round_creates_node`: Creates DIALOGUE node with sub_type="round"
- `test_add_round_temporal_edge`: TEMPORAL edge between consecutive rounds
- `test_detect_importance_identity`: "我叫张三" -> IDENTITY
- `test_detect_importance_must_remember`: "帮我记住" -> MUST_REMEMBER
- `test_detect_importance_trivial`: Short greeting -> TRIVIAL
- `test_detect_importance_normal`: Regular text -> NORMAL

### Step 7: test_layer2_attention_memory.py (AttentionWindow + MemoryGraph)

**TestAttentionMemoryBoost** (uses temp_memory_graph):
- `test_memory_boost_factor`: Node with activation=0.8 -> score *= (1 + 0.5*0.8) = 1.4x
- `test_no_memory_graph_no_boost`: Without MemoryGraph -> boost = 1.0
- `test_mode_plus_memory_combined`: Focus mode multiplier * memory boost combined correctly

### Step 8: test_layer3_e2e_complex_task.py (End-to-end, requires running system)

Re-use existing `test_3d_shooter_task.py` pattern with additional assertions:
- `test_complex_task_produces_task_graph`: TASK_GRAPH_UPDATE events received
- `test_complex_task_bfs_activation`: Logs contain "BFS 激活扩散完成" with valid>0
- `test_complex_task_hebbian_learning`: Logs contain "Hebbian 学习完成"
- `test_complex_task_response_quality`: Response length > 500 chars, contains code blocks

## Execution Order

Run in this exact order (each layer depends on previous):

```bash
# Layer 1: Independent modules (no cross-dependencies)
pytest tests/test_layer1_memory_graph.py -v
pytest tests/test_layer1_task_graph.py -v
pytest tests/test_layer1_circuit_breaker.py -v
pytest tests/test_layer1_attention_window.py -v
pytest tests/test_layer1_task_suspension.py -v

# Layer 2: Cross-module integration
pytest tests/test_layer2_graph_adapters.py -v
pytest tests/test_layer2_attention_memory.py -v

# Layer 3: End-to-end (requires running Zulong system)
pytest tests/test_layer3_e2e_complex_task.py -v --timeout=300
```

Or all at once:
```bash
pytest tests/test_layer1_*.py tests/test_layer2_*.py -v
```

## Verification

1. **All Layer 1 tests pass** with 0 failures - confirms each module works correctly in isolation
2. **All Layer 2 tests pass** - confirms modules collaborate correctly through adapters
3. **Layer 3 tests pass** with Zulong system running - confirms full pipeline works end-to-end
4. **No singleton leakage** - each test gets fresh instances via conftest fixtures
5. **Coverage**: Key methods of all 6 modules have at least one test covering happy path + edge case

## Key Technical Constraints

- **Singleton reset**: MemoryGraph, TaskSuspensionManager, InferenceEngine all use singleton pattern. Must reset `_instance` and `_initialized` between tests
- **Async tests**: TaskSuspensionManager methods are async, use `pytest-asyncio` with `@pytest.mark.asyncio`
- **Temp directories**: MemoryGraph and TaskSuspension write to disk; use tempfile.mkdtemp() to isolate
- **No LLM dependency**: Layer 1-2 tests must NOT call any LLM API. MemoryGraph's retrieve_context() is async and uses embedding - mock or skip in unit tests
- **Import safety**: Some modules do lazy imports (e.g., `from zulong.memory.memory_graph import get_memory_graph` inside functions). Tests should handle ImportError gracefully
