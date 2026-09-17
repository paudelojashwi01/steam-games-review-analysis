# Predicting Steam "Hits" :  Reach × Quality Classification

Can structured game attributes (price, playtime, achievements, platform support) predict which games become genuine hits on Steam, not just well-reviewed, but well-reviewed *and* widely played?

## The question

Most Steam games get *some* positive reviews — the platform's ratings skew positive overall, so "is this game liked?" is a weak question on its own. The more useful question is whether a game achieves broad, well-reviewed reach: a "hit" defined here as a game that clears the **top quartile of review quality** (≥91% positive) **and** the **top quartile of review volume** (≥238 total reviews) simultaneously. Only 7.3% of games in the dataset (3,876 of 53,199) clear both bars, a genuinely rare, meaningful target rather than an arbitrary cutoff.

`num_reviews_total` is deliberately excluded from the feature set, since it's used to define part of the label including it would let the model trivially learn the quartile cutoff instead of any real relationship.

## Approach

- **Data:** [Steam Games Dataset 2025](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset) (Kaggle), 89,618 games, cleaned down to 53,199 after removing entries with too few reviews to score reliably.
- **Model:** Logistic regression (`class_weight='balanced'` to handle the 7.3% positive rate), trained on price, discount, achievements, playtime (average and median), peak concurrent users, required age, and platform availability.
- **Iteration:** An earlier version predicted a single threshold (`pct_pos_total >= 70`) and returned a ROC-AUC of just 0.591, barely better than random. That blurred together games that just barely cleared "Mostly Positive" with universally beloved titles. Sharpening the label
