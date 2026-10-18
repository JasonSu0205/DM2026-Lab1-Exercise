import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools

def test_load_dataset_tool_fixture():
    # 1. 初始化 SessionState
    session = SessionState()
    
    # 2. 從 premade_tools 載入 load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # 3. 呼叫工具載入 agent_dev/sample_fixture.csv
    result = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # 4. 斷言檢查：對照 TEST_FIXTURE.md 中的 Known Answer 1
    # 總文件數應為 8
    assert result["n_documents"] == 8
    
    # 分類數量應為 catA: 4, catB: 4
    assert result["counts_per_category"] == {"catA": 4, "catB": 4}
    
    # 檢查 session.dataframe 中的資料結構與筆數
    assert session.dataframe is not None
    assert len(session.dataframe) == 8
    assert "text" in session.dataframe.columns
    assert "category_name" in session.dataframe.columns
    assert "category" in session.dataframe.columns
