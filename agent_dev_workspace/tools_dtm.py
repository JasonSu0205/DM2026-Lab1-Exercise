from typing import List, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer

def make_tools(session):
    @tool
    def build_dtm_tool(
        max_features: Optional[int] = None,
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
        ngram_range: List[int] = [1, 1]
    ) -> dict:
        """Builds a Document-Term Matrix (DTM) from document text in session.dataframe.

        Args:
            max_features: Maximum number of terms to keep.
            min_df: Minimum document frequency threshold (int for count, float for proportion).
            max_df: Maximum document frequency threshold (int for count, float for proportion).
            ngram_range: Range of n-gram sizes, e.g. [1, 1] for unigrams.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"error": "No dataframe or text column found in session."}

        # 1. 處理 None / 缺失文字列（填補為空字串以避免 vectorizer 報錯）
        texts = session.dataframe["text"].fillna("").tolist()

        vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=tuple(ngram_range)
        )
        
        feature_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out().tolist()

        # 2. 更新 session 矩陣與詞彙表狀態
        session.feature_matrix = feature_matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        # 3. 同步設定 session.labels 與 session.categories
        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].to_numpy())

        # 4. 計算 Sparsity
        n_rows, n_cols = feature_matrix.shape
        total_elements = n_rows * n_cols
        non_zero = int(feature_matrix.nnz)
        sparsity_pct = float(100.0 * (1.0 - non_zero / total_elements)) if total_elements > 0 else 0.0

        summary = {
            "result_id": session.next_result_id("dtm"),
            "n_documents": n_rows,
            "vocabulary_size": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4)
        }
        session.store_result(
            "build_dtm_tool",
            {
                "max_features": max_features,
                "min_df": min_df,
                "max_df": max_df,
                "ngram_range": ngram_range
            },
            summary
        )
        return summary

    @tool
    def term_frequency_tool(top_n: Optional[int] = 50) -> dict:
        """Aggregates term frequencies across all documents using session.feature_matrix.

        Args:
            top_n: Maximum number of top frequent terms to return. Default is 50. Pass None or -1 to return all terms.

        Returns:
            A dictionary containing top term frequencies sorted in descending order.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No DTM found in session. Run build_dtm_tool first."}

        # 對 DTM 進行欄加總 (axis=0)
        col_sums = np.asarray(session.feature_matrix.sum(axis=0)).ravel()
        
        # 配對詞名與詞頻，並按頻率降序排列
        term_counts = list(zip(session.feature_names, col_sums.tolist()))
        term_counts_sorted = sorted(term_counts, key=lambda x: x[1], reverse=True)
        
        total_vocabulary_size = len(term_counts_sorted)

        # 處理 top_n 切片邏輯（預設限制前 50 個，防止大型數據集 Token 爆表）
        if top_n is not None and top_n > 0:
            term_counts_sorted = term_counts_sorted[:top_n]

        frequencies_dict = {term: int(count) for term, count in term_counts_sorted}

        summary = {
            "result_id": session.next_result_id("tf"),
            "total_terms": total_vocabulary_size,
            "returned_terms": len(frequencies_dict),
            "term_frequencies": frequencies_dict
        }
        session.store_result("term_frequency_tool", {"top_n": top_n}, summary)
        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> dict:
        """Renders a heatmap of a raw positional slice of the document-term matrix.

        Args:
            n_terms: Number of terms (columns) to include from the DTM (default 20).
            n_documents: Number of documents (rows) to include from the DTM (default 20).

        Returns:
            A dictionary containing the sliced matrix, term names, and result ID.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No DTM found in session. Run build_dtm_tool first."}

        n_rows, n_cols = session.feature_matrix.shape

        # 防禦性邊界保護：若傳入 <=0 則自動取全量，否則取指定的範圍（不超過實際矩陣大小）
        actual_docs = min(n_documents, n_rows) if n_documents > 0 else n_rows
        actual_terms = min(n_terms, n_cols) if n_terms > 0 else n_cols

        # 取得密集切片 (Dense slice)
        sliced_matrix = session.feature_matrix[0:actual_docs, 0:actual_terms].toarray()
        sliced_terms = session.feature_names[0:actual_terms]
        doc_labels = [f"Doc_{i}" for i in range(actual_docs)]

        # 繪製 Heatmap（依照範例習慣，不使用 matplotlib.use("Agg")，並在繪圖結束後配合 plt.close(fig) 清理記憶體）
        fig, ax = plt.subplots(figsize=(max(6, actual_terms * 0.8), max(4, actual_docs * 0.5)))
        sns.heatmap(
            sliced_matrix,
            xticklabels=sliced_terms,
            yticklabels=doc_labels,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            ax=ax
        )
        ax.set_title(f"DTM Heatmap ({actual_docs} Docs x {actual_terms} Terms)")
        plt.tight_layout()

        # 設定 session.pending_figure 供系統自動儲存與渲染
        session.pending_figure = fig
        plt.close(fig)

        summary = {
            "result_id": session.next_result_id("dtm_heatmap"),
            "n_documents": actual_docs,
            "n_terms": actual_terms,
            "terms": sliced_terms,
            "matrix": sliced_matrix.tolist()
        }
        session.store_result("dtm_heatmap_tool", {"n_terms": n_terms, "n_documents": n_documents}, summary)
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
