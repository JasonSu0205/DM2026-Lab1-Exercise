from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def cosine_similarity_tool(
        doc_idx1: int,
        doc_idx2: int,
        text1: Optional[str] = None,
        text2: Optional[str] = None
    ) -> dict:
        """Computes cosine similarity between two documents in session.feature_matrix or custom texts.

        Args:
            doc_idx1: Index of the first document in session.dataframe / session.feature_matrix.
            doc_idx2: Index of the second document in session.dataframe / session.feature_matrix.
            text1: Optional raw text for document 1 (vectorized using session.artifacts['count_vectorizer']).
            text2: Optional raw text for document 2 (vectorized using session.artifacts['count_vectorizer']).

        Returns:
            A summary dictionary containing doc indices, raw similarity score, and result_id.
        """
        if session.feature_matrix is None:
            return {"error": "No DTM feature matrix found in session. Run build_dtm_tool first."}

        total_docs = session.feature_matrix.shape[0]

        # 1. 處理 custom text 的情況 (使用 session.artifacts['count_vectorizer'])
        vectorizer = session.artifacts.get("count_vectorizer")

        if text1 is not None:
            if vectorizer is None:
                return {"error": "count_vectorizer not found in session.artifacts to vectorize custom text1."}
            vec1 = vectorizer.transform([text1])
        else:
            if doc_idx1 < 0 or doc_idx1 >= total_docs:
                return {"error": f"doc_idx1 ({doc_idx1}) out of range [0, {total_docs - 1}]."}
            vec1 = session.feature_matrix[doc_idx1]

        if text2 is not None:
            if vectorizer is None:
                return {"error": "count_vectorizer not found in session.artifacts to vectorize custom text2."}
            vec2 = vectorizer.transform([text2])
        else:
            if doc_idx2 < 0 or doc_idx2 >= total_docs:
                return {"error": f"doc_idx2 ({doc_idx2}) out of range [0, {total_docs - 1}]."}
            vec2 = session.feature_matrix[doc_idx2]

        # 2. 計算 Cosine Similarity 並邊界防護 (對零向量回傳 0.0)
        norm1 = np.linalg.norm(vec1.toarray())
        norm2 = np.linalg.norm(vec2.toarray())

        if norm1 == 0 or norm2 == 0:
            sim = 0.0
        else:
            raw_sim = cosine_similarity(vec1, vec2)[0, 0]
            if np.isnan(raw_sim):
                sim = 0.0
            else:
                sim = float(raw_sim)

        result_id = session.next_result_id("cosine_sim")
        summary = {
            "result_id": result_id,
            "doc_idx1": doc_idx1,
            "doc_idx2": doc_idx2,
            "cosine_similarity": round(sim, 4),
            "raw_similarity": sim
        }

        session.store_result(
            "cosine_similarity_tool",
            {"doc_idx1": doc_idx1, "doc_idx2": doc_idx2},
            summary
        )
        return summary

    @tool
    def feature_correlation_matrix_tool(top_n: int = 20) -> dict:
        """Computes Pearson correlation matrix over top variance features and sets heatmap figure on session.pending_figure.

        Args:
            top_n: Number of top variance terms to evaluate (default 20).

        Returns:
            Dictionary with correlation matrix, top feature names, and result_id.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No feature_matrix or feature_names in session. Run build_dtm_tool first."}

        # 1. 計算各詞欄位的變異數
        # 使用 sparse calculation 避免將整個矩陣轉成 dense
        X = session.feature_matrix
        feature_names = np.array(session.feature_names)
        
        # Mean per column
        mean = np.array(X.mean(axis=0)).ravel()
        # Mean of squares per column
        mean_sq = np.array(X.power(2).mean(axis=0)).ravel()
        variances = mean_sq - (mean ** 2)

        # 2. 挑選變異數最大的前 top_n 個詞 (若少於 top_n 則全選)
        n_features = len(feature_names)
        actual_top_n = min(top_n, n_features)

        if actual_top_n == 0:
            return {"error": "No features available to compute correlation matrix."}

        # 由大到小排序 indices
        top_indices = np.argsort(variances)[::-1][:actual_top_n]
        top_names = feature_names[top_indices].tolist()

        # 3. 只將這前 top_n 欄轉為 dense numpy array 進行 np.corrcoef
        top_dense = X[:, top_indices].toarray()  # (n_samples, actual_top_n)

        # np.corrcoef 傳入 rowvar=False 表示每一欄為一個 variable
        corr_matrix = np.corrcoef(top_dense, rowvar=False)

        # 處理單一欄位或 2D 矩陣 shape 規格化
        if actual_top_n == 1:
            corr_matrix = np.array([[1.0]])

        # 處理完全沒變化的詞導致 0/0 計算出的 NaN (設為 0.0，對角線仍為 1.0)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        np.fill_diagonal(corr_matrix, 1.0)

        # 4. 繪製並標有數字的熱力圖
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".4f",
            xticklabels=top_names,
            yticklabels=top_names,
            cmap="vlag",
            ax=ax,
            vmin=-1.0,
            vmax=1.0
        )
        ax.set_title(f"Feature Correlation Matrix (Top {actual_top_n} Variance Terms)")
        plt.tight_layout()

        # 設定 session.pending_figure 並關閉 plt 避免記憶體洩漏
        session.pending_figure = fig
        plt.close(fig)

        # 5. 回傳結果 dict
        result_id = session.next_result_id("feat_corr")
        corr_list = corr_matrix.tolist()

        summary = {
            "result_id": result_id,
            "top_features": top_names,
            "correlation_matrix": corr_list
        }

        session.store_result(
            "feature_correlation_matrix_tool",
            {"top_n": top_n},
            summary
        )
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
