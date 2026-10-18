import pytest
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools

def test_variance_filter_tool_fixture():
    # 1. 初始化 SessionState
    session = SessionState()

    # 2. 載入 premade load_dataset_tool 並讀取 sample_fixture.csv
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 3. 呼叫 build_dtm_tool 建置 DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 4. 取得並執行 variance_filter_tool
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # 測試預設情況 (threshold=0.0)
    res_default = variance_filter_tool.invoke({"threshold": 0.0})
    assert res_default["total_terms"] == 5
    assert res_default["kept_terms_count"] == 5
    assert res_default["removed_terms_count"] == 0

    # 測試 Known Answer #5 (threshold=0.15)
    res_filtered = variance_filter_tool.invoke({"threshold": 0.15})
    assert res_filtered["total_terms"] == 5
    assert res_filtered["kept_terms_count"] == 4
    assert res_filtered["removed_terms_count"] == 1
    assert set(res_filtered["kept_terms"]) == {"alpha", "beta", "delta", "gamma"}
    assert res_filtered["removed_terms"] == ["always"]

    # 5. 驗證 full_report 內容與數值 (對照 Known Answer #5 變異數)
    result_id = res_filtered["result_id"]
    stored_entry = session.results_store[result_id]
    report_df = stored_entry["full_report"]

    assert isinstance(report_df, pd.DataFrame)
    assert list(report_df.columns) == ["term", "variance"]

    # 驗證 full_report 是否依 variance 降序排序
    variances_list = report_df["variance"].tolist()
    assert variances_list == sorted(variances_list, reverse=True)

    # 比對精確數值 (Known Answer #5)
    var_dict = dict(zip(report_df["term"], report_df["variance"]))
    assert var_dict["alpha"] == pytest.approx(1.1875, abs=1e-3)
    assert var_dict["gamma"] == pytest.approx(1.109375, abs=1e-3)
    assert var_dict["delta"] == pytest.approx(0.25, abs=1e-3)
    assert var_dict["beta"] == pytest.approx(0.234375, abs=1e-3)
    assert var_dict["always"] == pytest.approx(0.109375, abs=1e-3)

def test_pearson_filter_tool_fixture():
    # 1. 初始化獨立 SessionState
    session = SessionState()

    # 2. 載入 premade load_dataset_tool 並讀取 sample_fixture.csv
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 3. 建置 DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 4. 取得 pearson_filter_tool
    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    # 測試不存在的類別 (例如 "non_existent_cat") -> 應回傳 error
    res_invalid_cat = pearson_filter_tool.invoke({"target_class": "non_existent_cat"})
    assert "error" in res_invalid_cat

    # 5. 測試標靶類別 target_class="catB" ( Known Answer #6 )
    res_catB = pearson_filter_tool.invoke({"target_class": "catB"})
    assert res_catB["target_class"] == "catB"
    assert res_catB["total_terms"] == 5

    # 6. 檢查 full_report 內容與數值
    result_id = res_catB["result_id"]
    stored_entry = session.results_store[result_id]
    report_df = stored_entry["full_report"]

    assert isinstance(report_df, pd.DataFrame)
    assert list(report_df.columns) == ["term", "pearson_r"]

    # 斷言報表第一列為絕對值最大的 "delta" (r = 1.0)
    assert report_df.iloc[0]["term"] == "delta"

    # 7. 對照 Known Answer #6 比對 5 個詞的 Pearson correlation r
    # delta: 1.0, gamma: 0.8307, beta: -0.7746, alpha: -0.6882, always: 0.3780
    r_dict = dict(zip(report_df["term"], report_df["pearson_r"]))
    assert r_dict["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert r_dict["gamma"] == pytest.approx(0.8307, abs=1e-3)
    assert r_dict["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert r_dict["alpha"] == pytest.approx(-0.6882, abs=1e-3)
    assert r_dict["always"] == pytest.approx(0.3780, abs=1e-3)

def test_spearman_filter_tool_fixture():
    # 1. 初始化獨立 SessionState
    session = SessionState()

    # 2. 載入 premade load_dataset_tool 並讀取 sample_fixture.csv
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 3. 建置 DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 4. 取得 spearman_filter_tool
    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # 測試不存在的類別 (例如 "non_existent_cat") -> 應回傳 error
    res_invalid_cat = spearman_filter_tool.invoke({"target_class": "non_existent_cat"})
    assert "error" in res_invalid_cat

    # 5. 測試標靶類別 target_class="catB" ( Known Answer #6 )
    res_catB = spearman_filter_tool.invoke({"target_class": "catB"})
    assert res_catB["target_class"] == "catB"
    assert res_catB["total_terms"] == 5

    # 6. 檢查 full_report 內容與數值
    result_id = res_catB["result_id"]
    stored_entry = session.results_store[result_id]
    report_df = stored_entry["full_report"]

    assert isinstance(report_df, pd.DataFrame)
    assert list(report_df.columns) == ["term", "spearman_r"]

    # 斷言報表第一列為絕對值最大的 "delta" (spearman_r = 1.0)
    assert report_df.iloc[0]["term"] == "delta"

    # 7. 對照 Known Answer #6 比對 5 個詞的 Spearman correlation r
    # delta: 1.0, gamma: 0.9363, beta: -0.7746, alpha: -0.75, always: 0.3780
    r_dict = dict(zip(report_df["term"], report_df["spearman_r"]))
    assert r_dict["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert r_dict["gamma"] == pytest.approx(0.9363, abs=1e-3)
    assert r_dict["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert r_dict["alpha"] == pytest.approx(-0.7500, abs=1e-3)
    assert r_dict["always"] == pytest.approx(0.3780, abs=1e-3)
