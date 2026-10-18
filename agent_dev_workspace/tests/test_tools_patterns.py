import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools as make_patterns_tools

def test_mine_patterns_tool_fixture():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    # 1. 測試錯誤處理：漏給 min_sup, 漏給 k, 輸入不存在的類別
    res_no_minsup = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth"
    })
    assert "error" in res_no_minsup

    res_no_k = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk"
    })
    assert "error" in res_no_k

    res_invalid_cat = mine_patterns_tool.invoke({
        "category_name": "non_existent_cat",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert "error" in res_invalid_cat

    # 2. 測試 Known Answer #17: variance 與 term_frequency 篩選時兩個類別都留下 0 個字，且回傳 note 而不報錯
    for f_method in ["variance", "term_frequency"]:
        for cat in ["catA", "catB"]:
            res_no_terms = mine_patterns_tool.invoke({
                "category_name": cat,
                "filtering_method": f_method,
                "algorithm": "fpgrowth",
                "min_sup": 1
            })
            assert res_no_terms["pattern_count"] == 0
            assert "note" in res_no_terms
            assert res_no_terms["note"] == "no terms survived filtering"

    # 3. 測試 Known Answer #17: tfidf + fpgrowth
    # catA, min_sup=1 -> alpha (support 3)
    res_catA_fpg = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_catA_fpg["pattern_count"] == 1
    assert res_catA_fpg["patterns"][0]["pattern"].strip() == "alpha"
    assert res_catA_fpg["patterns"][0]["support"] == 3

    # 即刻檢查 artifacts 內有正確儲存結果 (避免被隨後的 min_sup=4 覆蓋)
    assert "patterns_catA_tfidf_fpgrowth" in session.artifacts
    assert len(session.artifacts["patterns_catA_tfidf_fpgrowth"]) == 1
    assert session.artifacts["patterns_catA_tfidf_fpgrowth"][0]["pattern"].strip() == "alpha"

    # catA, min_sup=4 -> 0 patterns
    res_catA_fpg_4 = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4
    })
    assert res_catA_fpg_4["pattern_count"] == 0

    # catB, min_sup=1 -> gamma (support 4)
    res_catB_fpg = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_catB_fpg["pattern_count"] == 1
    assert res_catB_fpg["patterns"][0]["pattern"].strip() == "gamma"
    assert res_catB_fpg["patterns"][0]["support"] == 4

    # 4. 測試 Known Answer #20: topk (k=1, 2, 3 時各自得出跟上面一樣的那一個結果)
    for k_val in [1, 2, 3]:
        res_catA_topk = mine_patterns_tool.invoke({
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "topk",
            "k": k_val
        })
        assert res_catA_topk["pattern_count"] == 1
        assert res_catA_topk["patterns"][0]["pattern"].strip() == "alpha"
        assert res_catA_topk["patterns"][0]["support"] == 3

        res_catB_topk = mine_patterns_tool.invoke({
            "category_name": "catB",
            "filtering_method": "tfidf",
            "algorithm": "topk",
            "k": k_val
        })
        assert res_catB_topk["pattern_count"] == 1
        assert res_catB_topk["patterns"][0]["pattern"].strip() == "gamma"
        assert res_catB_topk["patterns"][0]["support"] == 4

    # 5. 測試 Known Answer #20: maxfpgrowth (min_sup=1, 4 時都是 0 個結果，且不報錯)
    for min_s in [1, 4]:
        for cat in ["catA", "catB"]:
            res_maxfpg = mine_patterns_tool.invoke({
                "category_name": cat,
                "filtering_method": "tfidf",
                "algorithm": "maxfpgrowth",
                "min_sup": min_s
            })
            assert res_maxfpg["pattern_count"] == 0
            assert "note" in res_maxfpg
