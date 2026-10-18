import os
import tempfile
from typing import Optional, Literal, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool

# Import PAMI components
from PAMI.extras.convert import DF2DB
from PAMI.frequentPattern.basic import FPGrowth
from PAMI.frequentPattern.topk import FAE
from PAMI.frequentPattern.maximal import MaxFPGrowth

def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"] = "tfidf",
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None,
        max_patterns_to_show: int = 50
    ) -> dict:
        """Mines frequent patterns per category with vocabulary filtering using PAMI.

        Args:
            category_name: Target category string to mine.
            filtering_method: Feature selection method ("variance", "tfidf", or "term_frequency").
            algorithm: Pattern mining algorithm ("fpgrowth", "topk", or "maxfpgrowth").
            min_sup: Minimum support threshold (required for fpgrowth and maxfpgrowth).
            k: Number of top patterns to keep (required for topk).
            max_patterns_to_show: Maximum number of patterns to display in summary (default 50).

        Returns:
            Dictionary with mined patterns summary, count, and result_id.
        """
        if session.dataframe is None:
            return {"error": "No dataframe found in session. Load a dataset first."}

        # 1. 檢查 category_name 欄位與資料列是否存在
        df = session.dataframe
        cat_col = "category_name" if "category_name" in df.columns else ("label" if "label" in df.columns else None)
        if cat_col is None:
            return {"error": "Neither 'category_name' nor 'label' column found in session.dataframe."}

        cat_df = df[df[cat_col] == category_name]
        if len(cat_df) == 0:
            return {"error": f"Category '{category_name}' not found in dataset. Available: {list(df[cat_col].unique())}"}

        # 2. 驗證演算法必填參數
        if algorithm in ["fpgrowth", "maxfpgrowth"] and min_sup is None:
            return {"error": f"Parameter 'min_sup' is required for algorithm '{algorithm}'."}
        if algorithm == "topk" and k is None:
            return {"error": "Parameter 'k' is required for algorithm 'topk'."}

        # 3. 準備文字資料與 CountVectorizer (專屬此類別，使用 stop_words="english")
        text_col = "text" if "text" in cat_df.columns else cat_df.columns[0]
        texts = cat_df[text_col].fillna("").astype(str).tolist()

        try:
            vectorizer = CountVectorizer(stop_words="english")
            X_sparse = vectorizer.fit_transform(texts)
            feature_names = np.array(vectorizer.get_feature_names_out())
        except ValueError as e:
            # 處理 empty vocabulary (例如全部單字都是 stopwords 或空白)
            result_id = session.next_result_id("mine_pat")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "pattern_count": 0,
                "patterns": [],
                "note": f"Empty vocabulary after stopword removal: {str(e)}"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        if len(feature_names) == 0:
            result_id = session.next_result_id("mine_pat")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "pattern_count": 0,
                "patterns": [],
                "note": "No features survived stopword removal."
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # 轉為 Dense DataFrame 方便計算特徵指標
        dense_df = pd.DataFrame(X_sparse.toarray(), columns=feature_names)

        # 4. 套用 filtering_method 篩選字詞
        if filtering_method == "variance":
            # 使用 pandas 預設 df.var() (ddof=1)
            scores = dense_df.var()
            p_low = np.percentile(scores, 5)
            p_high = np.percentile(scores, 95)
            kept_mask = (scores >= p_low) & (scores <= p_high)

        elif filtering_method == "term_frequency":
            scores = dense_df.sum(axis=0)
            p_low = np.percentile(scores, 5)
            p_high = np.percentile(scores, 95)
            kept_mask = (scores >= p_low) & (scores <= p_high)

        elif filtering_method == "tfidf":
            transformer = TfidfTransformer()
            tfidf_mat = transformer.fit_transform(X_sparse)
            scores = pd.Series(np.array(tfidf_mat.mean(axis=0)).ravel(), index=feature_names)
            p_20 = np.percentile(scores, 20)
            kept_mask = scores >= p_20

        else:
            return {"error": f"Unsupported filtering_method: {filtering_method}"}

        filtered_df = dense_df.loc[:, kept_mask]
        kept_terms = filtered_df.columns.tolist()

        # 邊界護欄：若沒有半個字詞存活
        if len(kept_terms) == 0:
            result_id = session.next_result_id("mine_pat")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "pattern_count": 0,
                "patterns": [],
                "note": "no terms survived filtering"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # 5. 轉為 PAMI 交易資料庫並呼叫演算法探勘
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = os.path.join(tmpdir, "transactions.txt")

            # 轉為二元/計數 0/1 狀態（存在即可）
            # DF2DB: convert2TransactionalDatabase(db_file, '>=', 1)
            converter = DF2DB.DF2DB(filtered_df)
            converter.convert2TransactionalDatabase(db_file, '>=', 1)

            # 執行 PAMI 探勘
            if algorithm == "fpgrowth":
                model = FPGrowth.FPGrowth(db_file, minSup=min_sup, sep="\t")
            elif algorithm == "topk":
                model = FAE.FAE(db_file, k=k)
            elif algorithm == "maxfpgrowth":
                model = MaxFPGrowth.MaxFPGrowth(db_file, minSup=min_sup)
            else:
                return {"error": f"Unsupported algorithm: {algorithm}"}

            model.mine()
            patterns_df = model.getPatternsAsDataFrame()

        # 6. 解析探勘結果
        patterns_list = []
        if patterns_df is not None and not patterns_df.empty:
            # PAMI 回傳 DataFrame 通常包含 'Patterns' / 'pattern' 和 'Support' / 'support'
            pat_col = [c for c in patterns_df.columns if c.lower() in ["patterns", "pattern"]][0]
            sup_col = [c for c in patterns_df.columns if c.lower() in ["support", "frequency"]][0]

            for _, row in patterns_df.iterrows():
                pat_str = str(row[pat_col])
                sup_val = int(row[sup_col])
                patterns_list.append({"pattern": pat_str, "support": sup_val})

        # 7. 保存完整結果至 session.artifacts 與存入 session.results_store
        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = patterns_list

        capped_patterns = patterns_list[:max_patterns_to_show]

        result_id = session.next_result_id("mine_pat")
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "kept_terms_count": len(kept_terms),
            "pattern_count": len(patterns_list),
            "returned_patterns_count": len(capped_patterns),
            "patterns": capped_patterns
        }

        if len(patterns_list) == 0:
            summary["note"] = "no patterns found"

        session.store_result(
            "mine_patterns_tool",
            {
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "min_sup": min_sup,
                "k": k
            },
            summary
        )
        return summary

    return [mine_patterns_tool]
