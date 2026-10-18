# Agentic Pipeline Report: 20-Newsgroups (4 categories)
## 1. Data loading and checks

The session was restarted twice (once after the API quota ran out, once to fix an encoding error), so result_id numbers restart from 1 after each restart. Each restart reloaded the same data and rebuilt the same DTM, with identical results.

We loaded four categories from 20-Newsgroups: alt.atheism, soc.religion.christian, comp.graphics and sci.med, 2257 documents in total (load_dataset_1). soc.religion.christian has the most documents (599) and alt.atheism the fewest (480). There are no missing values (missing_3) and no duplicates (duplicates_4), so the data is clean and can be analyzed directly. For document length (describe_data_5), the mean (1992.96 characters) is much larger than the median (1235), which means the distribution is right-skewed with some extreme outliers. comp.graphics has the shortest median length (883), but the longest document (60713 characters) is also in this category.

![Document length by category](../plots/蘇宗瞬_agentic-pipeline_plot_describe_data_5.png)

## 2. Document-term matrix and feature filtering

We built a document-term matrix (dtm_2) of 2257 documents × 35788 terms. Its sparsity is 99.547%, which means most terms (rare words, names, number codes) appear in only a few documents, so almost every cell is 0. The most frequent terms (tf_7) are stopwords such as “the” (29825), “of” (17660), “to” and “and”, plus “edu” from email headers. These words appear in every category, so they cannot tell the topics apart and need filtering. The DTM heatmap (dtm_heatmap_3) shows that the first 20 terms alphabetically are number codes like “00” and “000”, and almost all cells are 0, so these terms are noise.

A variance filter with threshold 0.01 (variance_filter_4) removed 30434 low-variance terms, mostly rare number codes like “0000”, and kept 5354 (about 15%). This makes the matrix much smaller and less noisy. We then computed Pearson (pearson_filter_5) and Spearman (spearman_filter_6) correlations with comp.graphics. Terms like “graphics” (Spearman about 0.52), “image”, “3d” and “files” are positively correlated, while stopwords like “the”, “of”, “that” and “is” are negatively correlated. Since comp.graphics documents are the shortest, they contain fewer stopwords, so the correlation is negative. Finally, the feature correlation matrix (feat_corr_7) uses the top 20 terms by variance, not the Spearman terms. Almost the whole matrix is red: stopwords correlate with each other at about 0.75–0.94, because long documents contain more of every stopword. Only “jpeg” and “edu” have low correlations (about 0.1–0.45), since they appear only in specific documents.

![Pearson filter, comp.graphics](../plots/蘇宗瞬_agentic-pipeline_plot_turn38.png)

![Spearman filter, comp.graphics](../plots/蘇宗瞬_agentic-pipeline_plot_turn48.png)

![Feature correlation matrix](../plots/蘇宗瞬_agentic-pipeline_plot_feat_corr_7.png)

## 3. Frequent pattern mining (sci.med)

We mined frequent patterns in sci.med (mine_pat_3) using tfidf filtering, the top-k algorithm (FAE) and k = 20; 12807 terms were kept after filtering. The first run failed with a UnicodeEncodeError (Windows cp950 encoding could not write the character “í”), so we restarted Jupyter in UTF-8 mode (PYTHONUTF8=1) and it worked. All 20 patterns are email header words: “subject” (594 documents), “lines” (585), “organization” (582), “lines subject organization” (573), “edu” (416), “writes” (344) and “article” (343). “subject” appears in all 594 sci.med documents because every post has a “Subject:” line. None of the patterns are medical terms, so the mining found the fixed email format instead of the topic. To find real medical patterns, we could first remove the email headers, or compare categories and remove words that appear frequently across all of them.

## 4. Dimension reduction, labels and similarity

We reduced the DTM to two dimensions with PCA (reduce_pca_4). PC1 explains 65.9% of the variance and PC2 only 4.3% (70.3% in total). In the plot, most documents are packed near the origin and the four categories overlap, so this raw DTM cannot separate the topics well. The main variance comes from document length and stopwords, which appear in every category, not from topic words. A few points lie far from the center, and they show some direction: comp.graphics points go up, the two religion categories go to the lower right, and sci.med stays in the middle. These are probably very long documents (for example, posts with long code or data listings), because raw counts push long documents far away. The agent first said PC1 ranges from −21.83 to 44.31, but the plot shows it extends to about 830, so we corrected it. Next, we binarized the labels (binarize_labels_5) into a 2257 × 4 one-hot table, where each document has a 1 only in its own category column. This turns category names into numbers that can be used for classification or for correlating terms with a category. Finally, the cosine similarity between documents 0 and 1 (cosine_sim_6), both from comp.graphics, is 0.3532, which is fairly low. comp.graphics covers many sub-topics (image formats, 3D, hardware), so two posts can use very different words; what they share is mostly stopwords and email header words.

![PCA, colored by category](../plots/蘇宗瞬_agentic-pipeline_plot_reduce_pca_4.png)

## 5. Conclusion

Overall, many results in this pipeline were dominated by words that are not related to the topics: stopwords, email header words and document length. They shaped the term frequency ranking, the negative correlations with comp.graphics, the red feature correlation matrix, the header-only patterns in sci.med, and the overlapping PCA plot. The variance filter removes rare noise but keeps these common words, because their variance is high. If we redo the analysis, we should first compare all categories and remove words that appear frequently in every category (stopwords and email headers), and then use correlation or TF-IDF to select words that are special to one category. Only after this step would the comparisons between topics become meaningful.

AI disclosure: I used an AI assistant to explain concepts and to translate my own Chinese answers into English. All results come from my own pipeline run and session log.
