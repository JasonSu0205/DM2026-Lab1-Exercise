import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools

def test_cosine_similarity_tool_fixture():
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

    # 4. 取得 cosine_similarity_tool
    exploration_tools = make_exploration_tools(session)
    cosine_similarity_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # 5. 測試範圍外編號 -> 應回傳 error
    res_out_of_bounds = cosine_similarity_tool.invoke({"doc_idx1": 0, "doc_idx2": 99})
    assert "error" in res_out_of_bounds

    # 6. 對照 Known Answer #16 比對 4 組文件編號的 Cosine Similarity
    # 0 vs 1: 0.9847
    res_0_1 = cosine_similarity_tool.invoke({"doc_idx1": 0, "doc_idx2": 1})
    assert res_0_1["cosine_similarity"] == pytest.approx(0.9847, abs=1e-3)

    # 0 vs 3: 0.0909
    res_0_3 = cosine_similarity_tool.invoke({"doc_idx1": 0, "doc_idx2": 3})
    assert res_0_3["cosine_similarity"] == pytest.approx(0.0909, abs=1e-3)

    # 3 vs 4: 0.9847
    res_3_4 = cosine_similarity_tool.invoke({"doc_idx1": 3, "doc_idx2": 4})
    assert res_3_4["cosine_similarity"] == pytest.approx(0.9847, abs=1e-3)

    # 0 vs 7 (空白文件): 0.0
    res_0_7 = cosine_similarity_tool.invoke({"doc_idx1": 0, "doc_idx2": 7})
    assert res_0_7["cosine_similarity"] == pytest.approx(0.0, abs=1e-3)

    # 7. 測試輸入自訂 text1 (使用 session.artifacts["count_vectorizer"])
    # 傳入 text1 = "always alpha alpha alpha beta" 與 doc_idx2 = 1 進行比較 -> 應為 0.9847
    res_custom_text = cosine_similarity_tool.invoke({
        "doc_idx1": 0,
        "doc_idx2": 1,
        "text1": "always alpha alpha alpha beta"
    })
    assert res_custom_text["cosine_similarity"] == pytest.approx(0.9847, abs=1e-3)


def test_feature_correlation_matrix_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    exp_tools = make_exploration_tools(session)
    feat_corr_tool = next(t for t in exp_tools if t.name == "feature_correlation_matrix_tool")

    # Known answer 19
    res = feat_corr_tool.invoke({"top_n": 20})

    # 驗證欄位 key 與字詞順序 (top_features)
    expected_order = ["alpha", "gamma", "delta", "beta", "always"]
    assert res["top_features"] == expected_order

    corr = np.array(res["correlation_matrix"])

    # 對角線全為 1.0
    assert np.allclose(np.diag(corr), 1.0, atol=1e-3)

    # alpha (idx 0) 對 beta (idx 3) 為 0.8885
    assert corr[0, 3] == pytest.approx(0.8885, abs=1e-3)
    assert corr[3, 0] == pytest.approx(0.8885, abs=1e-3)

    # gamma (idx 1) 對 delta (idx 2) 為 0.8307
    assert corr[1, 2] == pytest.approx(0.8307, abs=1e-3)
    assert corr[2, 1] == pytest.approx(0.8307, abs=1e-3)

    # 驗證 pending_figure 有正確被設定
    assert session.pending_figure is not None
