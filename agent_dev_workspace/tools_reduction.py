from typing import Optional, Literal, List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def binarize_labels_tool(
        category_column: str = "category_name",
        max_rows_to_show: Optional[int] = 50
    ) -> dict:
        """One-hot encodes category labels into binary columns (0 and 1) using LabelBinarizer.

        For 2 classes, forces 2-column output format (one column per class) rather than
        scikit-learn's single-column default. Stores binarized labels in session.artifacts['binarized_labels'].

        Args:
            category_column: Name of the column containing category strings (default "category_name").
            max_rows_to_show: Maximum number of sample rows to display in summary (default 50).

        Returns:
            A summary dictionary containing class names, shape, sample records, and result_id.
        """
        if session.dataframe is None:
            return {"error": "No dataframe found in session. Load a dataset first."}

        # 欄位備選邏輯
        target_col = category_column if category_column in session.dataframe.columns else ("label" if "label" in session.dataframe.columns else None)

        if target_col is None:
            return {"error": f"Category column '{category_column}' (or 'label') not found in dataframe."}

        labels = session.dataframe[target_col].values

        lb = LabelBinarizer()
        binary_matrix = lb.fit_transform(labels)
        classes = lb.classes_.tolist()

        # 特殊邊界處理：若只有 2 個類別，LabelBinarizer 預設只產生 1 欄 (0/1)。
        # 依據需求，強制補齊成 2 欄的二元格式 [col_0, col_1]
        if len(classes) == 2 and binary_matrix.shape[1] == 1:
            binary_matrix = np.hstack([1 - binary_matrix, binary_matrix])

        # 轉成 DataFrame
        binarized_df = pd.DataFrame(binary_matrix, columns=[str(c) for c in classes])

        # 存回 session 供後續工具使用
        session.artifacts["binarized_labels"] = binarized_df
        session.artifacts["binarizer"] = lb

        # 限制顯示的範例列數
        if max_rows_to_show is not None and max_rows_to_show > 0:
            sample_records = binarized_df.head(max_rows_to_show).to_dict(orient="records")
        else:
            sample_records = binarized_df.to_dict(orient="records")

        result_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": result_id,
            "classes": classes,
            "shape": list(binarized_df.shape),
            "sample": sample_records
        }

        session.store_result(
            "binarize_labels_tool",
            {"category_column": category_column, "max_rows_to_show": max_rows_to_show},
            summary
        )
        return summary

    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: Optional[int] = None,
        max_coords_to_return: int = 50
    ) -> dict:
        """Reduces feature_matrix dimensions using PCA, t-SNE, or UMAP, and generates a scatter plot.

        Args:
            method: Dimensionality reduction algorithm ("pca", "tsne", or "umap").
            n_components: Number of target dimensions (default 2).
            perplexity: Perplexity parameter for t-SNE (default 30.0). Auto-adjusted if sample size is small.
            n_neighbors: Number of neighbors for UMAP (default 15). Auto-adjusted if sample size is small.
            random_state: Seed for reproducibility.
            max_coords_to_return: Cap on returned coordinates list to avoid oversized output (default 50).

        Returns:
            Dictionary with method, reduced coordinates (capped), optional explained variance ratio, and result_id.
        """
        if session.feature_matrix is None:
            return {"error": "No feature_matrix found in session. Run build_dtm_tool first."}

        X = session.feature_matrix
        n_samples = X.shape[0]

        if n_samples == 0:
            return {"error": "feature_matrix is empty."}

        # 1. 降維邏輯與參數保護
        explained_variance_ratio = None
        method_lower = method.lower()

        if method_lower == "pca":
            pca = PCA(n_components=n_components, random_state=random_state)
            # dense conversion for PCA/t-SNE/UMAP consistency
            coords = pca.fit_transform(X.toarray())
            explained_variance_ratio = pca.explained_variance_ratio_.tolist()

        elif method_lower == "tsne":
            # t-SNE 防護：perplexity 必須小於 n_samples
            actual_perplexity = perplexity
            if actual_perplexity >= n_samples:
                actual_perplexity = max(1.0, float(n_samples - 1))

            tsne_model = TSNE(
                n_components=n_components,
                perplexity=actual_perplexity,
                random_state=random_state
            )
            coords = tsne_model.fit_transform(X.toarray())

        elif method_lower == "umap":
            # UMAP 防護：n_neighbors 必須小於 n_samples 且 >= 2
            actual_n_neighbors = n_neighbors
            if actual_n_neighbors >= n_samples:
                actual_n_neighbors = max(2, n_samples - 1)

            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=actual_n_neighbors,
                random_state=random_state
            )
            coords = reducer.fit_transform(X.toarray())

        else:
            return {"error": f"Unsupported dimensionality reduction method: {method}"}

        # 保存降維座標至 session.artifacts
        session.artifacts[f"reduced_{method_lower}"] = coords

        # 2. 類別標籤準備 (用於繪圖)
        category_labels = None
        if session.dataframe is not None:
            if "category_name" in session.dataframe.columns:
                category_labels = session.dataframe["category_name"].values
            elif "label" in session.dataframe.columns:
                category_labels = session.dataframe["label"].values

        if category_labels is None and session.labels is not None:
            category_labels = session.labels

        # 3. 繪製依類別上色的 2D 散佈圖 (必備需求)
        fig, ax = plt.subplots(figsize=(8, 6))
        if category_labels is not None and len(category_labels) == n_samples:
            sns.scatterplot(
                x=coords[:, 0],
                y=coords[:, 1],
                hue=category_labels,
                palette="tab10",
                alpha=0.8,
                ax=ax
            )
            ax.legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            ax.scatter(coords[:, 0], coords[:, 1], alpha=0.8)

        ax.set_title(f"Dimensionality Reduction: {method_lower.upper()}")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        plt.tight_layout()

        # 設定 session.pending_figure 並關閉 plt
        session.pending_figure = fig
        plt.close(fig)

        # 4. 座標上限截斷，避免傳回太大 JSON
        coords_list = coords.tolist()
        capped_coords = coords_list[:max_coords_to_return]

        result_id = session.next_result_id(f"reduce_{method_lower}")
        summary = {
            "result_id": result_id,
            "method": method_lower,
            "n_components": n_components,
            "total_documents": n_samples,
            "returned_coords_count": len(capped_coords),
            "coordinates": capped_coords
        }

        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        session.store_result(
            "reduce_dimensions_tool",
            {
                "method": method,
                "n_components": n_components,
                "perplexity": perplexity,
                "n_neighbors": n_neighbors,
                "random_state": random_state
            },
            summary
        )
        return summary

    return [binarize_labels_tool, reduce_dimensions_tool]
