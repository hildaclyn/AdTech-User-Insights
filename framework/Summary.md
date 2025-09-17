# CTR Prediction Framework

This repository implements a **Click-Through Rate (CTR) prediction framework** combining gradient boosting (LightGBM, CatBoost) and deep sequence models (DeepFM, DIN). The project demonstrates an industry-style pipeline for feature engineering, model training, and ensemble blending.

---

## 1. Data Preparation

* Merged three core datasets:

  * **Impression log**: user–ad interactions (`user`, `adgroup_id`, `clk`, `time_stamp`)
  * **User profile**: demographic attributes (`age_level`, `shopping_level`, `final_gender_code`)
  * **Ad metadata**: product and campaign details (`brand`, `cate_id`, `campaign_id`)

* Converted timestamps into **date**, **hour**, and **weekday** fields.

* Temporal split strategy:

  * Training set = first seven days
  * Testing set = last day

---

## 2. Feature Engineering

### Multi-window CTR statistics

* Constructed **cumulative CTR features** over rolling 3-day and 7-day windows.
* Strictly excluded future information by applying historical shifts.
* Examples: `user_w3_ctr`, `adgroup_id_w7_ctr`, `brand_w3_ctr`.

### Frequency encoding

* Encoded entity frequencies to capture popularity and exposure bias.
* Examples: `user_freq`, `adgroup_id_freq`, `user_campaign_id_freq`.

### Cross features

* Built interaction keys between entities.
* Examples:

  * User × brand
  * User × campaign
  * Brand × category

---

## 3. Baseline

* Implemented **Logistic Regression** as a benchmark.
* Results:

  * AUC ≈ 0.4995 (close to random guessing)
  * LogLoss ≈ 0.1908

This highlighted the need for richer features and more expressive models.

---

## 4. Gradient Boosting Models

### LightGBM

* Designed for high-dimensional categorical and numerical features.
* Results:

  * AUC ≈ 0.5762
  * LogLoss ≈ 0.1856
* Important features: user-category CTRs, campaign-level CTRs, and frequency encodings.

### CatBoost

* Optimized for **high-cardinality categorical variables**.
* Results:

  * AUC ≈ 0.5737
  * LogLoss ≈ 0.1840
* Key signals included user–campaign frequency and user–brand interactions.

---

## 5. Deep Models

### DeepFM

* Combined **factorization machines** (capturing pairwise interactions) with **deep neural networks**.
* Results:

  * AUC ≈ 0.5750
  * LogLoss ≈ 0.2441
* Performed comparably to boosting methods, but suffered from higher variance and calibration issues.

### DIN (Deep Interest Network, Improved)

* Modeled **user click history sequences** with an **attention mechanism**, aligning past behavior with current ad context.
* Results:

  * AUC ≈ 0.5865
  * LogLoss ≈ 0.1899
* Outperformed DeepFM by better capturing sequential dependencies.

---

## 6. Model Ensembling

### Weighted blending

* Conducted grid search over weights constrained to the simplex.

* Blended models improved robustness and predictive accuracy.

* Best configuration (LGB + DIN):

  * Weights = (0.90, 0.00, 0.10)
  * AUC ≈ 0.5957
  * LogLoss ≈ 0.1844

* Alternative configuration (LGB + CatBoost + DeepFM):

  * Weights = (0.90, 0.06, 0.04)
  * AUC ≈ 0.5939
  * LogLoss ≈ 0.1862

### ROC comparison

* Blended models consistently shifted the ROC curve upwards relative to single models.
* Demonstrated that **ensemble averaging of tree-based and deep models is complementary**.

---

## 7. Insights

1. **Recent CTR features dominate**: short-term user engagement signals (3–7 days) were highly predictive.
2. **Cross features matter**: interactions such as user × campaign and user × brand explained variance not captured by single fields.
3. **Deep sequence modeling adds lift**: DIN achieved a clear gain by attending to user history.
4. **Ensembling is key**: boosting + DIN ensemble improved AUC from \~0.576 to \~0.596.

---

## 8. Future Work

* **Feature enrichment**: extend DIN with multi-field histories (brand, category) and time-decay attention.
* **Calibration**: apply post-training calibration to improve probabilistic interpretability.
* **Efficiency**: investigate GPU/CPU inference trade-offs for deployment.
* **Deployment**: integrate with serving pipelines and conduct online A/B testing.

---

## 9. Environment

* Python 3.12.11
* LightGBM 4.6.0
* CatBoost 1.2.8
* PyTorch 2.8.0 + cu126
* Hardware: Tesla T4 (15 GB GPU RAM)

---

## 10. Results Summary

| Model               | AUC        | LogLoss    |
| ------------------- | ---------- | ---------- |
| Logistic Regression | 0.4995     | 0.1908     |
| LightGBM            | 0.5762     | 0.1856     |
| CatBoost            | 0.5737     | 0.1840     |
| DeepFM              | 0.5750     | 0.2441     |
| DIN (Improved)      | 0.5865     | 0.1899     |
| **Blend (LGB+DIN)** | **0.5957** | **0.1844** |

---

## 11. Strategic Insights from AdTech CTR Prediction

The experimentation across logistic regression, gradient boosting, and deep interest models reveals several broader insights about advertisement data streams and the future of adtech:

* CTR prediction is inherently temporal and contextual

  *  Short-term engagement features (3–7 day CTR windows) consistently outperformed long-term averages.

  *  This highlights that user intent is dynamic, requiring adaptive models that reflect shifting interests.

* Cross-entity interactions drive personalization

  *  Signals such as user × brand and user × campaign carry stronger predictive power than standalone identifiers.

  *  In practice, this enables platforms to recommend more relevant ads by capturing the joint context of users and advertisers.

* Sequence models capture user intent evolution

  *  DIN outperformed factorization-based approaches by aligning historical behavior with current ad context.

  *  This reflects the evolution from static profiling → dynamic interest modeling, a cornerstone of modern recommendation systems.

* Ensembles remain a pragmatic solution

  * No single model dominated; however, a hybrid blend (LightGBM + DIN) delivered the highest AUC.

  * Industry practice mirrors this finding: ensembles combine explainability (trees) with expressiveness (deep models).

---

## 12. Implications for Advertisement Data Streams

* **Increased granularity of features**: The need for fine-grained, time-sensitive CTR signals pushes adtech systems to log richer event streams (impressions, clicks, dwell time, conversions).

* **Shift toward real-time modeling**: As user interest changes hourly, future systems must transition from batch-trained models toward streaming CTR prediction pipelines.

* **Privacy and efficiency trade-offs**: More granular user profiling raises regulatory and system performance challenges, requiring privacy-preserving learning and efficient GPU inference.

* **Feedback loop optimization**: Better CTR predictions directly impact auction dynamics, advertiser ROI, and user experience, reinforcing the centrality of predictive modeling in ad marketplaces.

---

## 13. Future Outlook

This project shows that even with limited historical logs, advanced modeling can meaningfully improve ad click prediction. Looking forward:
* **Context-aware architectures** (e.g., Transformer-based CTR models) will replace simpler attention networks.
* **Federated and privacy-aware** training will be necessary as data restrictions tighten.

* **Unified user-state representation** will integrate CTR models with recommendation and personalization pipelines.

Ultimately, CTR modeling is not just about predicting clicks—it is about shaping the data flows that power modern digital advertising ecosystems, bridging user behavior, advertiser goals, and platform optimization.

---

## Executive Takeaway

The final framework demonstrates that **hybrid modeling strategies**—combining tree ensembles with deep sequence models—produce the strongest CTR prediction performance.

The most effective configuration was a **90% LightGBM + 10% DIN blend**, achieving an AUC of \~0.596, representing a substantial improvement over any single model.

This pipeline mirrors industry standards in digital advertising, where ensemble-based CTR predictors are deployed to maximize accuracy, robustness, and scalability.

---
