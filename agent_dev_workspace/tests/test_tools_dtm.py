import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools

def test_build_dtm_tool_fixture():
    # 1. 建立獨立的 Session 實例
    session = SessionState()

    # 2. 載入 premade load_dataset_tool 並讀取 sample_fixture.csv
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 3. 取得並呼叫學生的 build_dtm_tool
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    
    result = build_dtm_tool.invoke({})

    # 4. 對照 TEST_FIXTURE.md 中的 Known Answer 4 進行斷言檢查
    assert result["vocabulary_size"] == 5
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert result["sparsity_pct"] == pytest.approx(47.5, abs=1e-3)

    # 5. 檢查 Session 狀態與 Artifacts
    assert session.feature_matrix is not None
    assert session.feature_matrix.shape == (8, 5)
    assert "count_vectorizer" in session.artifacts
    assert len(session.categories) == 2
    assert set(session.categories) == {"catA", "catB"}

def test_term_frequency_tool_fixture():
    # 1. 建立獨立的 Session 實例
    session = SessionState()

    # 2. 載入資料並建立 DTM
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")

    build_dtm_tool.invoke({})

    # 3. 測試 term_frequency_tool (預設呼叫)
    result = term_frequency_tool.invoke({})

    assert result["total_terms"] == 5
    assert result["returned_terms"] == 5

    tf_dict = result["term_frequencies"]
    
    # 對照 TEST_FIXTURE.md Known Answer 14 的精確詞頻數據
    assert tf_dict["always"] == 7
    assert tf_dict["gamma"] == 7
    assert tf_dict["alpha"] == 6
    assert tf_dict["delta"] == 4
    assert tf_dict["beta"] == 3

    # 檢查排序是否依照詞頻由大到小降序排列
    freq_values = list(tf_dict.values())
    assert freq_values == sorted(freq_values, reverse=True)

    # 4. 測試 top_n 切片功能 (例如 top_n=2)
    result_top2 = term_frequency_tool.invoke({"top_n": 2})
    assert result_top2["total_terms"] == 5
    assert result_top2["returned_terms"] == 2
    assert len(result_top2["term_frequencies"]) == 2

def test_dtm_heatmap_tool_fixture():
    # 1. 建立獨立的 Session 實例
    session = SessionState()

    # 2. 載入資料並建立 DTM
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    build_dtm_tool.invoke({})

    # 3. (1) 預設呼叫：對照 Known Answer 21 檢查完整 8x5 矩陣內容
    result_default = dtm_heatmap_tool.invoke({})

    assert result_default["n_documents"] == 8
    assert result_default["n_terms"] == 5
    assert result_default["terms"] == ["alpha", "always", "beta", "delta", "gamma"]

    expected_full_matrix = [
        [3, 1, 1, 0, 0],  # Doc 0
        [2, 1, 1, 0, 0],  # Doc 1
        [1, 1, 1, 0, 0],  # Doc 2
        [0, 1, 0, 1, 3],  # Doc 3
        [0, 1, 0, 1, 2],  # Doc 4
        [0, 1, 0, 1, 1],  # Doc 5
        [0, 1, 0, 1, 1],  # Doc 6
        [0, 0, 0, 0, 0],  # Doc 7 (empty text)
    ]
    assert result_default["matrix"] == expected_full_matrix

    # (3) 斷言 pending_figure 被正確設定
    assert session.pending_figure is not None

    # 4. (2) 指定 n_terms=2, n_documents=3：檢查切出左上角 3x2 的區域
    result_slice = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})

    assert result_slice["n_documents"] == 3
    assert result_slice["n_terms"] == 2
    assert result_slice["terms"] == ["alpha", "always"]

    expected_corner_matrix = [
        [3, 1],  # Doc 0
        [2, 1],  # Doc 1
        [1, 1],  # Doc 2
    ]
    assert result_slice["matrix"] == expected_corner_matrix
