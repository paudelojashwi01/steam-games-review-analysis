# Predicting Steam "Hits" — Reach × Quality Classification

Can structured game attributes (price, playtime, achievements, platform support) predict which games become genuine hits on Steam not just well-reviewed, but well-reviewed *and* widely played?

## The question

Most Steam games get *some* positive reviews — the platform's ratings skew positive overall, so "is this game liked?" is a weak question on its own. The more useful question is whether a game achieves broad, well-reviewed reach: a "hit" defined here as a game that clears the **top quartile of review quality** (≥91% positive) **and** the **top quartile of review volume** (≥238 total reviews) simultaneously. Only 7.3% of games in the dataset (3,876 of 53,199) clear both bars a genuinely rare, meaningful target rather than an arbitrary cutoff.

`num_reviews_total` is deliberately excluded from the feature set, since it's used to define part of the label including it would let the model trivially learn the quartile cutoff instead of any real relationship.

## Iteration: why there are two notebooks here

**[`steam_games_v1_first_attempt.ipynb`](steam_games_v1_first_attempt.ipynb)** the first version, predicting a single threshold (`pct_pos_total >= 70`, Steam's own "Mostly Positive" cutoff). This returned a ROC-AUC of just **0.591** — barely better than random because the label itself was noisy: a game that just barely clears 70% and a universally beloved 95%-positive game both got the same label. Kept in the repo rather than deleted, because the diagnosis that came out of this failure directly motivated the fix below.

**[`steam_games_hit_classification.ipynb`](steam_games_hit_classification.ipynb)** the refined version below, isolating genuinely rare, unambiguous hits instead. This is the model whose results are described in this README.

## Approach (v2 / final)

- **Data:** [Steam Games Dataset 2025](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset) (Kaggle), 89,618 games, cleaned down to 53,199 after removing entries with too few reviews to score reliably.
- **Model:** Logistic regression (`class_weight='balanced'` to handle the 7.3% positive rate), trained on price, discount, achievements, playtime (average and median), peak concurrent users, required age, and platform availability.
- **What changed from v1:** sharpened the label to isolate genuinely rare, unambiguous hits, excluded `num_reviews_total` from the features (it's now part of the label definition, so including it would leak), and evaluated with average precision rather than raw accuracy, given the class imbalance.

## Results

| Metric | Value |
|---|---|
| ROC-AUC | 0.646 |
| Average Precision | 0.126 (vs. 0.073 random — a ~73% lift) |
| Recall (hit class) | 0.570 |
| Precision (hit class) | 0.115 |

Raw accuracy (0.649) looks worse than a naive "always predict not-a-hit" baseline (0.927) but that comparison is misleading with a 7.3% positive rate: the naive baseline catches zero real hits, while this model recovers 57% of them. That's the right tradeoff for a screening tool meant to flag promising titles, not make final calls alone.

## Key finding

**`average_playtime_forever` predicts hits positively (odds ratio ≈ 1.67); `median_playtime_forever` predicts them negatively (odds ratio ≈ 0.74).** Hits don't look like games with a uniformly engaged fanbase they look like games with a *skewed* engagement distribution: a smaller core of highly dedicated players driving the average up while the typical player logs comparatively less time. That pattern reads as passionate superfans rather than broad, even engagement.

Other notable coefficients:
- **`price` is positive** (odds ratio ≈ 1.51) free-to-play wasn't a shortcut to broad, well-reviewed success in this data.
- **`peak_ccu` is small and positive here**, in contrast to a strongly *negative* relationship with simple review positivity in the earlier single-threshold model suggesting high-concurrency titles are more polarizing on average, but the ones that combine high concurrency with high quality *and* volume are, unsurprisingly, more likely to be real hits.
- **Platform availability** (`mac`, `windows`, `linux`) all contributed positively, `mac` support most of all.

## Limitations

An intersection-of-quartiles label is inherently rare and harder to predict than either dimension alone a model that beats the random baseline by ~73% on average precision represents real, if modest, signal on a genuinely hard question. As with any model built on structured metadata alone, most of the remaining variance likely comes from factors outside this feature set: genre, marketing, community word-of-mouth, and qualitative game design.

## Repo contents

| File | Description |
|---|---|
| `steam_games_hit_classification.ipynb` | Final model -> predicts "hit" status (top quartile of both quality and reach) |
| `steam_games_v1_first_attempt.ipynb` | First attempt -> single-threshold target, kept to show the iteration |

## Tech stack

`Python` · `Pandas` · `NumPy` · `scikit-learn` (Logistic Regression) · `Matplotlib` · `Seaborn`

## Data source

[Steam Games Dataset 2025](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset)
