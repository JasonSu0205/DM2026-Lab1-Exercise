from typing import Optional
import os
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def list_files_tool(subdirectory: str = ".") -> dict:
        """Lists files and directories in the target subdirectory to assist in dataset discovery.

        Args:
            subdirectory: Relative path to list files from (default ".").

        Returns:
            A dictionary containing lists of files and directories found in the specified location.
        """
        repo_root = os.path.abspath(".")
        target_path = os.path.abspath(os.path.join(repo_root, subdirectory))

        # 安全防護：檢查解析後的絕對路徑是否依然位在 repo_root 底下
        try:
            common = os.path.commonpath([repo_root, target_path])
        except ValueError:
            return {"error": "Access denied: Path is outside the repository."}

        if common != repo_root:
            return {"error": "Access denied: Cannot access paths outside the repository."}

        if not os.path.exists(target_path) or not os.path.isdir(target_path):
            return {"error": f"Directory '{subdirectory}' not found or is not a valid directory."}

        entries = os.listdir(target_path)
        directories = []
        files = []

        for entry in entries:
            full_path = os.path.join(target_path, entry)
            if os.path.isdir(full_path):
                directories.append(entry)
            else:
                files.append(entry)

        directories.sort()
        files.sort()

        summary = {
            "result_id": session.next_result_id("list_files"),
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files
        }
        session.store_result(
            "list_files_tool",
            {"subdirectory": subdirectory},
            summary
        )
        return summary

    @tool
    def check_missing_tool(
        text_column: str = "text",
        max_indices_to_show: Optional[int] = 50
    ) -> dict:
        """Checks for missing or empty text values in session.dataframe.

        Args:
            text_column: Name of the column containing document text (default "text").
            max_indices_to_show: Maximum number of missing row indices to return in summary (default 50). Pass None or -1 for all.

        Returns:
            A dictionary with counts, percentage, and index samples of missing text entries.
        """
        if session.dataframe is None:
            return {"error": "No dataframe found in session. Load a dataset first."}

        if text_column not in session.dataframe.columns:
            return {"error": f"Column '{text_column}' not found in dataframe."}

        series = session.dataframe[text_column]

        # 判定條件：NaN/None 或 去除前後空白後為空字串
        missing_mask = series.isna() | series.astype(str).str.strip().eq("")
        all_missing_indices = session.dataframe.index[missing_mask].tolist()
        missing_count = len(all_missing_indices)
        total_rows = len(session.dataframe)
        missing_pct = round(100.0 * missing_count / total_rows, 4) if total_rows > 0 else 0.0

        # 安全限制回傳的索引列表數量，避免 Token 上限溢出
        if max_indices_to_show is not None and max_indices_to_show > 0:
            returned_indices = all_missing_indices[:max_indices_to_show]
        else:
            returned_indices = all_missing_indices

        summary = {
            "result_id": session.next_result_id("missing"),
            "total_rows": total_rows,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "missing_indices": returned_indices,
            "has_missing": missing_count > 0
        }
        session.store_result(
            "check_missing_tool",
            {"text_column": text_column, "max_indices_to_show": max_indices_to_show},
            summary
        )
        return summary

    @tool
    def check_duplicates_tool(
        text_column: str = "text",
        drop: bool = False
    ) -> dict:
        """Checks for duplicate documents and optionally drops all copies of duplicate rows.

        Args:
            text_column: Column name to check for duplicates (default "text").
            drop: If True, drops EVERY copy of duplicated rows using keep=False (default False).

        Returns:
            A summary dictionary with duplicate count and row statistics.
        """
        if session.dataframe is None:
            return {"error": "No dataframe found in session. Load a dataset first."}

        if text_column not in session.dataframe.columns:
            return {"error": f"Column '{text_column}' not found in dataframe."}

        original_rows = len(session.dataframe)
        
        # 計算重複筆數（標記重複列）
        duplicate_mask = session.dataframe.duplicated(subset=[text_column], keep='first')
        duplicate_count = int(duplicate_mask.sum())

        if drop and duplicate_count > 0:
            # 依據規格：使用 keep=False 移除所有重複副本，對齊 Master 邏輯
            session.dataframe.drop_duplicates(subset=[text_column], keep=False, inplace=True)
            session.dataframe.reset_index(drop=True, inplace=True)
            remaining_rows = len(session.dataframe)
            dropped = True

            # 清空因刪除資料而失效的舊 DTM 狀態，避免維度不匹配
            session.feature_matrix = None
            session.feature_names = None
            session.labels = None
            note = "Duplicates dropped. Session DTM feature_matrix reset; please re-run build_dtm_tool."
        else:
            remaining_rows = original_rows
            dropped = False
            note = "No rows dropped."

        summary = {
            "result_id": session.next_result_id("duplicates"),
            "original_rows": original_rows,
            "duplicate_count": duplicate_count,
            "dropped": dropped,
            "remaining_rows": remaining_rows,
            "note": note
        }
        session.store_result(
            "check_duplicates_tool",
            {"text_column": text_column, "drop": drop},
            summary
        )
        return summary

    @tool
    def sample_data_tool(
        n: Optional[int] = None,
        frac: Optional[float] = None,
        random_state: int = 42,
        replace: bool = False,
        max_indices_to_show: Optional[int] = 50
    ) -> dict:
        """Subsamples the working DataFrame (session.dataframe) using pandas sample.

        Args:
            n: Absolute number of rows to sample.
            frac: Fraction of axis items to return (0.0 to 1.0). Cannot be combined with n.
            random_state: Seed for random number generator (default 42 for reproducibility).
            replace: Allow or disallow sampling of the same row more than once (default False).
            max_indices_to_show: Maximum sampled row indices to show in summary (default 50).

        Returns:
            A summary dictionary with original and sampled row counts, sampled indices, and status.
        """
        if session.dataframe is None:
            return {"error": "No dataframe found in session. Load a dataset first."}

        # 錯誤處理：不能同時指定 n 與 frac
        if n is not None and frac is not None:
            return {"error": "Cannot specify both 'n' and 'frac' at the same time."}

        total_rows = len(session.dataframe)

        # 錯誤處理：未開啟 replace 時，n 不能大於總筆數
        if n is not None and not replace and n > total_rows:
            return {"error": f"Cannot sample {n} rows without replacement when dataframe only has {total_rows} rows."}

        # 執行抽樣
        sampled_df = session.dataframe.sample(
            n=n,
            frac=frac,
            replace=replace,
            random_state=random_state
        )

        # 於 reset_index 之前記錄原始抽樣列號（對齊 Known Answer #13，如 [1, 5, 0, 7]）
        sampled_indices = sampled_df.index.tolist()

        # 限制顯示的列號清單長度，防止 token 溢出
        if max_indices_to_show is not None and max_indices_to_show > 0:
            returned_indices = sampled_indices[:max_indices_to_show]
        else:
            returned_indices = sampled_indices

        # 更新 session.dataframe 並重設 index
        session.dataframe = sampled_df.reset_index(drop=True)

        # 清空舊 DTM 與 labels 狀態，防止後續維度不符
        session.feature_matrix = None
        session.feature_names = None
        session.labels = None

        sampled_rows = len(session.dataframe)

        summary = {
            "result_id": session.next_result_id("sample"),
            "original_rows": total_rows,
            "sampled_rows": sampled_rows,
            "sampled_indices": returned_indices,
            "random_state": random_state,
            "replace": replace
        }
        session.store_result(
            "sample_data_tool",
            {
                "n": n,
                "frac": frac,
                "random_state": random_state,
                "replace": replace,
                "max_indices_to_show": max_indices_to_show
            },
            summary
        )
        return summary

    @tool
    def tokenize_tool(text_column: str = "text") -> dict:
        """Tokenizes each document's text into lowercased unigrams using nltk.word_tokenize.

        Args:
            text_column: Name of the text column to tokenize (default "text").

        Returns:
            A dictionary summary with document count and total tokens.
        """
        if session.dataframe is None:
            return {"error": "No dataframe found in session. Load a dataset first."}

        if text_column not in session.dataframe.columns:
            return {"error": f"Column '{text_column}' not found in dataframe."}

        # 確保 NLTK punkt 資料包已下載
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

        unigrams_list = []
        total_tokens = 0

        for text in session.dataframe[text_column]:
            if text is None or not isinstance(text, str) or not text.strip():
                tokens = []
            else:
                # 轉小寫後使用 nltk.word_tokenize 分詞
                tokens = word_tokenize(text.lower())
            
            unigrams_list.append(tokens)
            total_tokens += len(tokens)

        # 更新 session.dataframe 與 session.artifacts
        session.dataframe["unigrams"] = unigrams_list
        session.artifacts["tokens"] = unigrams_list

        summary = {
            "result_id": session.next_result_id("tokenize"),
            "n_documents": len(session.dataframe),
            "total_tokens": total_tokens,
            "avg_tokens_per_doc": round(total_tokens / len(session.dataframe), 4) if len(session.dataframe) > 0 else 0.0
        }
        session.store_result("tokenize_tool", {"text_column": text_column}, summary)
        return summary

    @tool
    def describe_data_tool(
        text_column: str = "text",
        category_column: str = "category_name"
    ) -> dict:
        """Computes descriptive statistics on text length and generates a boxplot by category.

        Args:
            text_column: Name of text column to compute character length from (default "text").
            category_column: Name of category column for grouped statistics and boxplot (default "category_name").

        Returns:
            A dictionary summary containing text_length summary statistics overall and per category.
        """
        if session.dataframe is None:
            return {"error": "No dataframe found in session. Load a dataset first."}

        if text_column not in session.dataframe.columns:
            return {"error": f"Column '{text_column}' not found in dataframe."}

        df = session.dataframe.copy()

        # 安全處理 missing text values (NaN / None 轉為空字串，長度為 0)
        text_series = df[text_column].fillna("")
        text_lengths = text_series.apply(len)
        df["text_length"] = text_lengths
        session.dataframe["text_length"] = text_lengths

        # 1. 計算全體 (overall) 描述性統計
        overall_stats = df["text_length"].describe().to_dict()

        # 2. 按類別計算描述性統計
        by_category = {}
        target_cat_col = category_column if category_column in df.columns else ("category" if "category" in df.columns else None)

        if target_cat_col is not None:
            for cat, group in df.groupby(target_cat_col):
                by_category[str(cat)] = group["text_length"].describe().to_dict()

        # 3. 繪製 Boxplot 並設定 session.pending_figure
        fig, ax = plt.subplots(figsize=(6, 4))
        if target_cat_col is not None:
            sns.boxplot(data=df, x=target_cat_col, y="text_length", ax=ax)
            ax.set_xlabel(target_cat_col)
        else:
            sns.boxplot(data=df, y="text_length", ax=ax)

        ax.set_title("Text Length Distribution")
        ax.set_ylabel("Text Length (characters)")

        session.pending_figure = fig
        plt.close(fig)

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "overall": overall_stats,
            "by_category": by_category
        }
        session.store_result(
            "describe_data_tool",
            {"text_column": text_column, "category_column": category_column},
            summary
        )
        return summary

    @tool
    def inspect_data_tool(
        n: int = 5,
        max_char_len: int = 100,
        max_list_len: int = 10
    ) -> dict:
        """Inspects the first n rows of session.dataframe, returning columns and truncated content.

        Args:
            n: Number of initial rows to inspect (default 5). Must be greater than 0.
            max_char_len: Maximum character length for string values before truncating (default 100).
            max_list_len: Maximum elements for list values (e.g. unigrams) before truncating (default 10).

        Returns:
            A summary dictionary with total_rows, columns, rows_inspected, and data records.
        """
        if session.dataframe is None:
            return {"error": "No dataframe found in session. Load a dataset first."}

        if n <= 0:
            return {"error": "Parameter 'n' must be a positive integer (greater than 0)."}

        total_rows = len(session.dataframe)
        head_df = session.dataframe.head(n).copy()

        def _truncate_val(val):
            if isinstance(val, str) and len(val) > max_char_len:
                return val[:max_char_len] + "..."
            elif isinstance(val, list) and len(val) > max_list_len:
                return val[:max_list_len] + [f"... (+{len(val) - max_list_len} more)"]
            return val

        # 針對字串與 list 型態欄位進行長度截斷
        for col in head_df.columns:
            head_df[col] = head_df[col].apply(_truncate_val)

        result_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": result_id,
            "total_rows": total_rows,
            "columns": head_df.columns.tolist(),
            "rows_inspected": len(head_df),
            "data": head_df.to_dict(orient="records")
        }
        session.store_result(
            "inspect_data_tool",
            {"n": n, "max_char_len": max_char_len, "max_list_len": max_list_len},
            summary
        )
        return summary

    return [
        list_files_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        tokenize_tool,
        describe_data_tool,
        inspect_data_tool
    ]
