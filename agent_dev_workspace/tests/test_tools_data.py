import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools

def test_check_missing_tool_fixture():
    # 1. 初始化獨立的 SessionState
    session = SessionState()

    # 2. 載入 premade load_dataset_tool 並讀取 sample_fixture.csv
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 3. 取得並呼叫學生的 check_missing_tool
    data_tools = make_tools(session)
    check_missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")

    result = check_missing_tool.invoke({})

    # 4. 對照 TEST_FIXTURE.md 中的 Known Answer 2 進行斷言檢查
    # 總筆數 8，精確有 1 筆缺失文字（索引 7）
    assert result["total_rows"] == 8
    assert result["missing_count"] == 1
    assert result["has_missing"] is True
    assert result["missing_indices"] == [7]
    assert result["missing_pct"] == pytest.approx(12.5, abs=1e-3)

def test_check_duplicates_tool_fixture():
    # 1. 初始化獨立的 SessionState
    session = SessionState()

    # 2. 載入 premade load_dataset_tool 並讀取 sample_fixture.csv
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    check_duplicates_tool = next(t for t in data_tools if t.name == "check_duplicates_tool")

    # 3. 測試僅檢查不刪除 (drop=False)
    result_check = check_duplicates_tool.invoke({"drop": False})
    assert result_check["original_rows"] == 8
    assert result_check["duplicate_count"] == 1
    assert result_check["dropped"] is False
    assert result_check["remaining_rows"] == 8
    assert len(session.dataframe) == 8

    # 4. 測試執行刪除 (drop=True)：對照 Known Answer 3 (keep=False 刪除所有副本後剩 6 筆)
    result_drop = check_duplicates_tool.invoke({"drop": True})
    assert result_drop["original_rows"] == 8
    assert result_drop["duplicate_count"] == 1
    assert result_drop["dropped"] is True
    assert result_drop["remaining_rows"] == 6
    assert len(session.dataframe) == 6

def test_tokenize_tool_fixture():
    # 1. 初始化獨立的 SessionState
    session = SessionState()

    # 2. 載入資料
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    # 3. 呼叫 tokenize_tool
    result = tokenize_tool.invoke({})

    assert result["n_documents"] == 8
    assert result["total_tokens"] == 27  # (5+4+3) + (5+4+3+3) + 0 = 27

    # 4. 精確對照 Known Answer 10 檢查 8 筆文件的分詞內容
    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        []
    ]

    assert "unigrams" in session.dataframe.columns
    assert session.dataframe["unigrams"].tolist() == expected_tokens
    assert session.artifacts["tokens"] == expected_tokens

def test_list_files_tool_fixture():
    # 1. 初始化獨立的 SessionState
    session = SessionState()

    # 2. 取得學生的 list_files_tool
    data_tools = make_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")

    # 3. 測試 Known Answer #11: subdirectory="newdataset"
    res_newdataset = list_files_tool.invoke({"subdirectory": "newdataset"})
    assert res_newdataset["directories"] == []
    assert res_newdataset["files"] == ["Reddit-stock-sentiment.csv"]

    # 4. 測試路徑越界防護: 傳入 "../" 應回傳錯誤
    res_outside = list_files_tool.invoke({"subdirectory": "../"})
    assert "error" in res_outside
    assert "Access denied" in res_outside["error"]

def test_sample_data_tool_fixture():
    # 1. 初始化獨立的 SessionState
    session = SessionState()

    # 2. 載入資料 sample_fixture.csv (共 8 列)
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    sample_data_tool = next(t for t in data_tools if t.name == "sample_data_tool")

    # 3. 測試 Known Answer #13: n=4, random_state=42
    result_valid = sample_data_tool.invoke({"n": 4, "random_state": 42})
    assert result_valid["original_rows"] == 8
    assert result_valid["sampled_rows"] == 4
    assert result_valid["sampled_indices"] == [1, 5, 0, 7]
    assert len(session.dataframe) == 4

    # 4. 測試異常情況 1: n 與 frac 同時傳入 -> 應回傳 error
    result_both = sample_data_tool.invoke({"n": 2, "frac": 0.5})
    assert "error" in result_both

    # 5. 測試異常情況 2: n > 總列數 (n=10, 只有 4 列, replace=False) -> 應回傳 error
    result_oversized = sample_data_tool.invoke({"n": 10, "replace": False})
    assert "error" in result_oversized

def test_describe_data_tool_fixture():
    # 1. 初始化獨立的 SessionState
    session = SessionState()

    # 2. 載入資料 sample_fixture.csv
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 3. 呼叫 describe_data_tool
    data_tools = make_tools(session)
    describe_data_tool = next(t for t in data_tools if t.name == "describe_data_tool")

    result = describe_data_tool.invoke({})

    # 4. 對照 Known Answer #18 斷言檢查
    # 每篇字數: [29, 23, 17, 30, 24, 18, 18, 0]
    expected_lengths = [29, 23, 17, 30, 24, 18, 18, 0]
    assert "text_length" in session.dataframe.columns
    assert session.dataframe["text_length"].tolist() == expected_lengths

    # 全體 overall: count=8, mean=19.875, max=30
    overall = result["overall"]
    assert overall["count"] == 8
    assert overall["mean"] == pytest.approx(19.875, abs=1e-3)
    assert overall["max"] == pytest.approx(30, abs=1e-3)

    # 分類統計 (catA / catB)
    by_cat = result["by_category"]
    assert "catA" in by_cat
    assert "catB" in by_cat

    # catA mean=17.25, catB mean=22.5
    assert by_cat["catA"]["mean"] == pytest.approx(17.25, abs=1e-3)
    assert by_cat["catB"]["mean"] == pytest.approx(22.5, abs=1e-3)

    # 驗證 session.pending_figure 非 None
    assert session.pending_figure is not None

def test_inspect_data_tool_fixture():
    # 1. 初始化獨立的 SessionState
    session = SessionState()

    # 2. 載入資料 sample_fixture.csv
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    # 3. 測試 n=3: total_rows=8, rows_inspected=3, 第一列內容與 csv 一致
    res_n3 = inspect_data_tool.invoke({"n": 3})
    assert res_n3["total_rows"] == 8
    assert res_n3["rows_inspected"] == 3
    assert len(res_n3["data"]) == 3
    assert res_n3["data"][0]["text"] == "always alpha alpha alpha beta"

    # 4. 測試 n=0: 應回傳 error
    res_n0 = inspect_data_tool.invoke({"n": 0})
    assert "error" in res_n0

    # 5. 測試 n=100 (大於總列數 8): 應回傳全部 8 列
    res_n100 = inspect_data_tool.invoke({"n": 100})
    assert res_n100["total_rows"] == 8
    assert res_n100["rows_inspected"] == 8
    assert len(res_n100["data"]) == 8
