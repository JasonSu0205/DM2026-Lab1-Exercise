from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def variance_filter_tool(
        threshold: float = 0.0,
        max_terms_to_show: Optional[int] = 50
    ) -> dict:
        """Flags near-constant terms by feature variance for removal from session.feature_matrix.

        Args:
            threshold: Variance threshold strictly above which terms are kept (default 0.0).
            max_terms_to_show: Maximum number of kept/removed term names to display in summary (default 50).

        Returns:
            A summary dictionary with term counts, filtered term lists, and result_id.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No DTM feature matrix found in session. Run build_dtm_tool first."}

        X = session.feature_matrix

        # 記憶體優化：使用稀疏矩陣計算 variance: E[X^2] - (E[X])^2
        mean = np.array(X.mean(axis=0)).ravel()
        squared_X = X.copy()
        squared_X.data = squared_X.data ** 2
        mean_of_squared = np.array(squared_X.mean(axis=0)).ravel()
        variances = mean_of_squared - mean ** 2

        feature_names = session.feature_names
        total_terms = len(feature_names)

        # 建立完整的變異數 DataFrame (供 full_report 畫圖使用，第1欄: term, 第2欄: variance)
        report_df = pd.DataFrame({
            "term": feature_names,
            "variance": variances
        }).sort_values(by="variance", ascending=False).reset_index(drop=True)

        # 嚴格大於門檻: kept_mask = variances > threshold
        kept_mask = variances > threshold
        all_kept_terms = [name for name, keep in zip(feature_names, kept_mask) if keep]
        all_removed_terms = [name for name, keep in zip(feature_names, kept_mask) if not keep]

        # 限制顯示的清單長度，防止 LLM token 溢出
        if max_terms_to_show is not None and max_terms_to_show > 0:
            returned_kept = all_kept_terms[:max_terms_to_show]
            returned_removed = all_removed_terms[:max_terms_to_show]
        else:
            returned_kept = all_kept_terms
            returned_removed = all_removed_terms

        result_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": result_id,
            "total_terms": total_terms,
            "kept_terms_count": len(all_kept_terms),
            "removed_terms_count": len(all_removed_terms),
            "threshold": threshold,
            "kept_terms": returned_kept,
            "removed_terms": returned_removed
        }

        # 將完整的 report_df 存入 session.store_result 供 visualize_result_tool 使用
        session.store_result(
            "variance_filter_tool",
            {"threshold": threshold, "max_terms_to_show": max_terms_to_show},
            summary,
            full_report=report_df
        )
        return summary

    @tool
    def pearson_filter_tool(
        target_class: str,
        threshold: float = 0.0,
        max_terms_to_show: Optional[int] = 50
    ) -> dict:
        """Computes Pearson correlation between each term's count and a target class label using matrix operations.

        Args:
            target_class: Target class category string for One-vs-Rest correlation check.
            threshold: Correlation magnitude threshold strictly above which terms are kept (default 0.0).
            max_terms_to_show: Maximum number of kept/removed terms to display in summary (default 50).

        Returns:
            A summary dictionary with Pearson correlation statistics and filtered terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No DTM feature matrix found in session. Run build_dtm_tool first."}

        if session.dataframe is None or ("category_name" not in session.dataframe.columns and "label" not in session.dataframe.columns):
            return {"error": "No dataframe or category label column found in session."}

        cat_col = "category_name" if "category_name" in session.dataframe.columns else "label"
        labels = session.dataframe[cat_col].values

        if target_class not in labels:
            return {"error": f"Target class '{target_class}' not found in dataframe labels."}

        # 1. 建立 One-vs-Rest 標籤向量 (1.0 或 0.0)
        y = (labels == target_class).astype(float)
        y_mean = np.mean(y)
        y_diff = y - y_mean
        y_norm = np.linalg.norm(y_diff)

        X = session.feature_matrix
        feature_names = session.feature_names
        total_terms = len(feature_names)

        if y_norm == 0:
            pearson_r_array = np.zeros(total_terms)
        else:
            # 2. 全向量化矩陣運算 (計算 Pearson correlation)
            X_mean = np.array(X.mean(axis=0)).ravel()
            cov = np.array(X.T.dot(y_diff)).ravel()

            X_sq = X.copy()
            X_sq.data = X_sq.data ** 2
            X_sq_sum = np.array(X_sq.sum(axis=0)).ravel()
            n_samples = X.shape[0]
            SS_x = X_sq_sum - n_samples * (X_mean ** 2)
            SS_x = np.maximum(SS_x, 0.0)  # 防止浮點數誤差導致負數
            X_norm = np.sqrt(SS_x)

            denom = X_norm * y_norm
            with np.errstate(divide='ignore', invalid='ignore'):
                pearson_r_array = np.where(denom > 0, cov / denom, 0.0)
                pearson_r_array = np.nan_to_num(pearson_r_array, nan=0.0)

        # 3. 建立 full_report DataFrame (按 |pearson_r| 絕對值由大到小排序)
        report_df = pd.DataFrame({
            "term": feature_names,
            "pearson_r": pearson_r_array,
            "abs_r": np.abs(pearson_r_array)
        }).sort_values(by="abs_r", ascending=False).reset_index(drop=True)

        # 4. 依據已依 |pearson_r| 排序好的順序提取 kept / removed terms
        all_kept_terms = report_df[report_df["abs_r"] > threshold]["term"].tolist()
        all_removed_terms = report_df[report_df["abs_r"] <= threshold]["term"].tolist()

        # 移除輔助欄位 abs_r，精確保留規格要求的前兩欄
        report_df_to_store = report_df[["term", "pearson_r"]]

        # 5. 取前 max_terms_to_show 個最具代表性 (絕對值最大) 的詞彙
        if max_terms_to_show is not None and max_terms_to_show > 0:
            returned_kept = all_kept_terms[:max_terms_to_show]
            returned_removed = all_removed_terms[:max_terms_to_show]
        else:
            returned_kept = all_kept_terms
            returned_removed = all_removed_terms

        result_id = session.next_result_id("pearson_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": total_terms,
            "kept_terms_count": len(all_kept_terms),
            "removed_terms_count": len(all_removed_terms),
            "kept_terms": returned_kept,
            "removed_terms": returned_removed
        }

        session.store_result(
            "pearson_filter_tool",
            {"target_class": target_class, "threshold": threshold, "max_terms_to_show": max_terms_to_show},
            summary,
            full_report=report_df_to_store
        )
        return summary

    @tool
    def spearman_filter_tool(
        target_class: str,
        threshold: float = 0.0,
        top_k_var_prefilter: int = 1500,
        max_terms_to_show: Optional[int] = 50
    ) -> dict:
        """Computes Spearman rank correlation between each term's rank and a target class label.

        Pre-filters terms by top variance (default 1500) to keep computation efficient, converts
        counts to ranks, and computes rank correlation. Constant terms receive 0.0 correlation.

        Args:
            target_class: Target class category string for One-vs-Rest correlation check.
            threshold: Spearman correlation magnitude threshold strictly above which terms are kept (default 0.0).
            top_k_var_prefilter: Pre-filter top K terms by variance before computing rank correlation (default 1500).
            max_terms_to_show: Maximum number of kept/removed terms to display in summary (default 50).

        Returns:
            A summary dictionary with Spearman correlation statistics and filtered terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No DTM feature matrix found in session. Run build_dtm_tool first."}

        if session.dataframe is None or ("category_name" not in session.dataframe.columns and "label" not in session.dataframe.columns):
            return {"error": "No dataframe or category label column found in session."}

        cat_col = "category_name" if "category_name" in session.dataframe.columns else "label"
        labels = session.dataframe[cat_col].values

        if target_class not in labels:
            return {"error": f"Target class '{target_class}' not found in dataframe labels."}

        X = session.feature_matrix
        feature_names = np.array(session.feature_names)
        total_terms = len(feature_names)

        # 1. 前置依變異數挑選 top_k_var_prefilter (預設 1500) 個詞彙
        mean = np.array(X.mean(axis=0)).ravel()
        squared_X = X.copy()
        squared_X.data = squared_X.data ** 2
        mean_of_squared = np.array(squared_X.mean(axis=0)).ravel()
        variances = mean_of_squared - mean ** 2

        if top_k_var_prefilter is not None and top_k_var_prefilter > 0 and top_k_var_prefilter < total_terms:
            top_indices = np.argsort(variances)[::-1][:top_k_var_prefilter]
        else:
            top_indices = np.arange(total_terms)

        selected_names = feature_names[top_indices]
        selected_X = X[:, top_indices].toarray()

        # 2. 建立標籤 vector 與其 rank
        y = (labels == target_class).astype(float)
        y_ranks = rankdata(y)
        y_rank_diff = y_ranks - np.mean(y_ranks)
        y_rank_norm = np.linalg.norm(y_rank_diff)

        spearman_r_list = []

        # 3. 對篩選出的每個 term 計算 rank correlation (Spearman r)
        for i in range(selected_X.shape[1]):
            x = selected_X[:, i]
            # 完全沒變化的詞 (無變異數/常數欄位) 直接給 0.0
            if np.max(x) == np.min(x):
                r = 0.0
            else:
                x_ranks = rankdata(x)
                x_rank_diff = x_ranks - np.mean(x_ranks)
                x_rank_norm = np.linalg.norm(x_rank_diff)

                denom = x_rank_norm * y_rank_norm
                if denom == 0:
                    r = 0.0
                else:
                    r = np.dot(x_rank_diff, y_rank_diff) / denom
                    if np.isnan(r):
                        r = 0.0
            spearman_r_list.append(float(r))

        spearman_r_array = np.array(spearman_r_list)

        # 4. 建立 full_report DataFrame (第1欄: term, 第2欄: spearman_r，按 |spearman_r| 絕對值由大到小排序)
        report_df = pd.DataFrame({
            "term": selected_names,
            "spearman_r": spearman_r_array,
            "abs_r": np.abs(spearman_r_array)
        }).sort_values(by="abs_r", ascending=False).reset_index(drop=True)

        # 依據 |spearman_r| 排序好後提取 kept / removed terms
        all_kept_terms = report_df[report_df["abs_r"] > threshold]["term"].tolist()
        all_removed_terms = report_df[report_df["abs_r"] <= threshold]["term"].tolist()

        # 精確保留規格要求的兩欄: term, spearman_r
        report_df_to_store = report_df[["term", "spearman_r"]]

        # 取前 max_terms_to_show 個最具代表性 (絕對值最大) 的詞彙
        if max_terms_to_show is not None and max_terms_to_show > 0:
            returned_kept = all_kept_terms[:max_terms_to_show]
            returned_removed = all_removed_terms[:max_terms_to_show]
        else:
            returned_kept = all_kept_terms
            returned_removed = all_removed_terms

        result_id = session.next_result_id("spearman_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(selected_names),
            "kept_terms_count": len(all_kept_terms),
            "removed_terms_count": len(all_removed_terms),
            "kept_terms": returned_kept,
            "removed_terms": returned_removed
        }

        session.store_result(
            "spearman_filter_tool",
            {
                "target_class": target_class,
                "threshold": threshold,
                "top_k_var_prefilter": top_k_var_prefilter,
                "max_terms_to_show": max_terms_to_show
            },
            summary,
            full_report=report_df_to_store
        )
        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
