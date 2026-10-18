import pytest
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools

def test_binarize_labels_tool_fixture():
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

    # 3. 取得並執行 binarize_labels_tool
    reduction_tools = make_reduction_tools(session)
    binarize_labels_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    result = binarize_labels_tool.invoke({})

    # 4. 斷言類別與表格形狀 (8 列, 2 欄)
    assert result["classes"] == ["catA", "catB"]
    assert result["shape"] == [8, 2]

    # 5. 斷言第 0 列 (catA) 與第 3 列 (catB) 的 One-Hot 數值
    samples = result["sample"]
    assert samples[0] == {"catA": 1, "catB": 0}
    assert samples[3] == {"catA": 0, "catB": 1}

    # 6. 驗證結果有正確存入 session.artifacts
    assert "binarized_labels" in session.artifacts
    binarized_df = session.artifacts["binarized_labels"]
    assert isinstance(binarized_df, pd.DataFrame)
    assert list(binarized_df.columns) == ["catA", "catB"]


def test_reduce_dimensions_tool():
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

    reduction_tools = make_reduction_tools(session)
    reduce_dims_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # 1. PCA (Known Answer #7) - 維持原樣比對數值與 pending_figure
    res_pca = reduce_dims_tool.invoke({"method": "pca"})
    coords_pca = res_pca["coordinates"]
    assert coords_pca[0][0] == pytest.approx(2.3669, abs=1e-3)
    assert coords_pca[0][1] == pytest.approx(0.8871, abs=1e-3)

    evr = res_pca["explained_variance_ratio"]
    assert evr[0] == pytest.approx(0.7532, abs=1e-3)
    assert evr[1] == pytest.approx(0.1965, abs=1e-3)
    assert session.pending_figure is not None

    # 2. t-SNE - 彈性測試：不比對固定座標，驗證維度、重複樣本一致性、種子可重現性與 pending_figure
    session.pending_figure = None
    res_tsne1 = reduce_dims_tool.invoke({
        "method": "tsne",
        "perplexity": 2,
        "random_state": 42
    })
    coords_tsne1 = res_tsne1["coordinates"]
    
    # 驗證維度為 8 篇 x 2 維
    assert len(coords_tsne1) == 8
    assert len(coords_tsne1[0]) == 2

    # 驗證內容完全重複的第 5 篇與第 6 篇座標近似相同
    assert coords_tsne1[5][0] == pytest.approx(coords_tsne1[6][0], abs=1e-3)
    assert coords_tsne1[5][1] == pytest.approx(coords_tsne1[6][1], abs=1e-3)

    # 驗證固定 random_state 重複執行結果一致
    res_tsne2 = reduce_dims_tool.invoke({
        "method": "tsne",
        "perplexity": 2,
        "random_state": 42
    })
    coords_tsne2 = res_tsne2["coordinates"]
    assert coords_tsne1[0][0] == pytest.approx(coords_tsne2[0][0], abs=1e-3)
    assert coords_tsne1[0][1] == pytest.approx(coords_tsne2[0][1], abs=1e-3)

    assert session.pending_figure is not None

    # 3. UMAP (Known Answer #9) - 維持原樣比對數值與 pending_figure
    session.pending_figure = None
    res_umap = reduce_dims_tool.invoke({
        "method": "umap",
        "n_neighbors": 3,
        "random_state": 42
    })
    coords_umap = res_umap["coordinates"]
    assert coords_umap[0][0] == pytest.approx(6.6811, abs=1e-3)
    assert coords_umap[0][1] == pytest.approx(-6.8218, abs=1e-3)
    assert session.pending_figure is not None
