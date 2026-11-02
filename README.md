# Measuring Responsibility Attribution in Public Discourse on AI Harms

Code, word lists and derived data for the article "Measuring Responsibility Attribution in Public Discourse on AI Harms: Grammatical Agency, Moral Vocabulary and Compositional Change".

The study measures who is named as the agent of harm in Reddit discussions of AI harms (2020 to 2025), using the grammatical agents of 50 harm verbs, and how much moral vocabulary (Moral Foundations Dictionary, care.vice and fairness.vice) these discussions use. Both measures are validated by hand, and the estimates are corrected for changes in the composition of the corpus.

## Data

Reddit texts and usernames are not redistributed. The `data/` folder contains identifiers, metadata, derived variables and hand codes only.

| File | Content |
|---|---|
| `items.csv` | The 36,294 analysed items: Reddit identifiers and metadata. `row` links items to all other files. |
| `harm_clauses.csv` | The 2,704 harm clauses: item, verb, voice, agent category and period. |
| `item_topics.csv` | BERTopic topic of each item with at least one harm clause. |
| `topic_labels.csv` | Topic labels, mapping to Shelby et al. (2023) categories and inclusion flag. |
| `recall_200_codes.csv` | Recall sample: harm and agent codes for 200 sentences. |
| `mfd_150_codes.csv` | Moral Foundations Dictionary validation: codes for 150 matches. |
| `topic_robustness.csv` | Topic-fixed estimates under alternative topic models (Table A9). |

The texts can be retrieved from Reddit with the identifiers in `items.csv`. Items deleted since collection cannot be retrieved.

The Moral Foundations Dictionary (`mft_original.dic`, Graham et al., 2009) is not included; it is available from the Moral Foundations project.

## Code

The scripts were written as Google Colab cells and are meant to be run in order. Lines starting with `!pip` install packages in Colab. File paths at the top of each script point to `/content/` and may need to be adjusted.

| Script | Purpose | Paper |
|---|---|---|
| `00_setup.py` | Paths and shared functions | |
| `01_topic_model.py` | BERTopic model, topic-fixed agent shares, topic table | Table 1, Table A8 |
| `02_figure_corrections.py` | Figure of the estimate under successive corrections | Figure 1 |
| `03_authors.py` | Returning and new authors | Table A7 |
| `04_decomposition.py` | Kitagawa decomposition into between- and within-topic change | Section 4.1 |
| `05_topic_robustness.py` | Alternative topic models | Table A9 |
| `06_recall.py` | Recall sample, recall and agent accuracy with keyword baseline | Table A3 |
| `07_moral_vocabulary.py` | Moral vocabulary rates, subreddit and topic mix fixed | Table 2 |
| `08_mfd_validation.py` | Dictionary validation sample, precision, word-level analysis | Table A4 |
| `09_mfd_validated_subset.py` | Validated subset of the dictionary and Poisson model | Table 2 |
| `10_export_public_data.py` | Writes the public files in `data/` without texts or usernames | |

## Requirements

Python 3.10 or later. See `requirements.txt`.

## Citation

If you use this code or data, please cite the article (details to be added on publication).
