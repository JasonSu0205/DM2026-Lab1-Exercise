# Homework Report: Reddit Stock Sentiment (3 classes)

## 1. Data loading, duplicates and document length

We loaded Reddit-stock-sentiment.csv (load_dataset_1): 847 posts/comments about stocks with three sentiment labels, -1.0 negative (315), 0.0 neutral (423) and 1.0 positive (109). Unlike the four newsgroup topics, which are objective and balanced, sentiment is subjective and the classes are imbalanced: neutral has almost four times as many documents as positive. inspect_data_2 also shows labels that are hard to agree on, such as “Seeing lots of red in the ticker.” labeled neutral. There are no missing values (missing_3), but there are 23 duplicate documents (duplicates_4), while 20-Newsgroups had none. Most are short replies without an opinion, so they are in the neutral class. When dropping them (duplicates_5), our tool removed 34 rows and left 813 instead of 824, because it counts duplicates with keep='first' but drops them with keep=False, which also deletes the original of each group. This is a bug in our tool that the newsgroup data never revealed; it affects 11 documents (about 1.3%). After deduplication (describe_data_6), neutral has 391, negative 315 and positive 107 documents. The documents are very short: the median length is only 80 characters (1235 in 20-Newsgroups) and the shortest is 1 character. With so few words per document, there are few sentiment words to separate the classes, and two documents often share no words at all.

![Text length by sentiment](../plots/蘇宗瞬_homework_plot_describe_data_6.png)

## 2. Document-term matrix and feature filtering

The document-term matrix (dtm_7) has 813 documents × 4313 terms, far fewer terms than 20-Newsgroups (35788), but the sparsity is almost the same (99.52%). Each comment is short and uses only a few words, so most cells are still 0. The top terms (tf_8) are all stopwords, such as “the” (874), “to” (554) and “and” (471). Unlike 20-Newsgroups, there are no email header words like “edu”, because Reddit comments have no headers. The DTM heatmap (dtm_heatmap_9) again shows number tokens like “00” and “10” with almost all zeros. The variance filter with threshold 0.01 (variance_filter_10) removed these rare terms and kept only 490 of 4313 (about 11%). We then correlated terms with the positive class ‘1.0’ using Pearson (pearson_filter_11) and Spearman (spearman_filter_12). Top terms include “technology”, “tldr”, “huge”, “nice” and “potential”, but all correlations are weak: the highest is only about 0.16, compared with 0.52 for “graphics” in 20-Newsgroups. People express positive sentiment with many different words, and each word appears in only a few short comments, so no single word represents the class. Many terms have exactly the same value (about 0.128) because each appears in only one or two positive comments. We only analyzed the positive class here. The feature correlation matrix (feat_corr_13) again uses 20 stopwords, but the correlations are much weaker than in 20-Newsgroups (mostly 0.2–0.69 instead of 0.75–0.94). Since all comments are short and use different words, stopwords do not increase together the way they do in long newsgroup posts.

![DTM heatmap](../plots/蘇宗瞬_homework_plot_dtm_heatmap_9.png)

![Pearson filter, positive class](../plots/蘇宗瞬_homework_plot_turn82.png)

![Spearman filter, positive class](../plots/蘇宗瞬_homework_plot_turn76.png)

![Feature correlation matrix](../plots/蘇宗瞬_homework_plot_feat_corr_13.png)

## 3. Frequent pattern mining (positive class)

We mined the top 20 frequent patterns for the positive class (‘1.0’) using TF-IDF and top-k (mine_pat_14). Unlike the 20-Newsgroups pipeline, where every pattern was an email header word (subject, lines…), these data are Reddit comments, so the patterns are content words such as economy, silver, market, good and better. However, all 20 patterns are single words, and even the most frequent one (“just”) appears in only 15 of the 107 positive comments (about 14%). Because the comments are short and people use different words, words rarely occur together, so no multi-word pattern reaches the threshold. Many patterns (just, doing, going, think, tariffs) are neutral words, so their frequency in positive comments does not mean they express positive sentiment. In addition, this word-counting approach cannot detect sarcasm: a sarcastic comment like “Great, I lost everything” contains a positive word but has negative meaning.

## 4. Dimension reduction, labels and similarity

PCA (reduce_pca_15) reduced the data to two components. PC1 explains 26.70% and PC2 5.12% of the variance, 31.82% in total, so about two-thirds of the information is lost in this 2-D view. The three classes overlap completely, with only a few outliers (a neutral comment at (42, 24), a negative one at (40, −10) and a positive one at (20, −24)). Because sentiment is subjective and expressed with very different words, the word counts are scattered and PCA cannot find directions that separate positive, negative and neutral comments. The labels were then converted to an 813 × 3 one-hot matrix (binarize_labels_16): one row per comment and one column per class, with a single 1 in each row. Finally, the cosine similarity between document 0 (“Calls on retards”, negative) and document 1 (neutral) is 0.0 (cosine_sim_17), because the two short comments share no words at all. In the 20-Newsgroups pipeline the similarity was 0.3532, since the long emails share many stopwords and header words.

![PCA, colored by sentiment](../plots/蘇宗瞬_homework_plot_reduce_pca_15.png)

## 5. Comparison with the 20-Newsgroups pipeline and conclusion

Compared with the 20-Newsgroups pipeline, this dataset is much harder for word-based methods. The emails were long, had clear topics and shared many stopwords and header words, while the Reddit comments are short, informal and labeled by subjective sentiment. As a result, almost every step showed weak signals: correlations with the positive class were low (about 0.13–0.16), frequent patterns were only single and often neutral words, PCA could not separate the three classes, and two comments could have a cosine similarity of 0. Counting words alone cannot capture sentiment well, especially for sarcasm, where a positive word can carry a negative meaning. To improve this, instead of comparing comments by raw word frequency, we could use a sentiment lexicon and compare the proportion of positive, negative and neutral words in each comment, which focuses on sentiment rather than topic, although it still cannot handle sarcasm.

AI disclosure: I used an AI assistant to explain concepts and to translate my own Chinese answers into English. All results come from my own homework agent session and session log.

