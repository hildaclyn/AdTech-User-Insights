
# Model Analysis and Results

## Figure 1. Top 10 Features (LightGBM vs. CatBoost)

**Description**  
The charts below show the ten most important features identified by **LightGBM** (left) and **CatBoost** (right).  
- LightGBM importance is measured by split gain.  
- CatBoost importance is measured by prediction value change.  
Although the scales differ, the relative ranking of features provides insight into predictive drivers.

**Key Findings**  
- **Recent behavioral signals dominate**  
  - `user_cate_id_w3_ctr` (3-day user × category CTR) is the strongest predictor for LightGBM.  
  - `user_campaign_id_freq`, `user_freq`, and `user_brand_freq` rank highest for CatBoost.  
- **Short-window CTR is more effective than long-window CTR**  
  - Features over the past 3–7 days consistently outperform longer-term signals.  
- **Product and demographic attributes matter**  
  - `cate_id`, `brand`, and `age_level` contribute to baseline user differentiation.  
- **Temporal effects are significant**  
  - `weekday` appears among CatBoost’s top features, indicating systematic variations in engagement by day of week.

**Business Implications**  
- Prioritize **short-term user behavior** when designing campaigns.  
- Expand **user × product/campaign cross-features** for finer targeting.  
- Optimize **budget allocation by weekday** to match engagement patterns.  
- Use **aggregate brand/category CTR** as fallback for cold-start cases.

---

## Figure 2. Model Comparison: ROC Curves

**Description**  
The ROC (Receiver Operating Characteristic) curve plots the **True Positive Rate (TPR)** against the **False Positive Rate (FPR)** across thresholds.  
- The diagonal line corresponds to random guessing (AUC = 0.5).  
- Curves closer to the top-left corner indicate stronger discrimination ability.  
- **AUC (Area Under Curve)** summarizes performance across thresholds.

**Key Results**  
- **Single models**:  
  - LightGBM: AUC ≈ 0.576  
  - CatBoost: AUC ≈ 0.574  
- **With DIN (Deep Interest Network)**:  
  - AUC ≈ 0.586  
- **Blended model (LGB + CatBoost + DIN)**:  
  - AUC ≈ **0.596**, LogLoss ≈ **0.184**  
  - Represents a measurable improvement in ranking accuracy for CTR prediction.

**Business Implications**  
- Use **blended scores** for impression bidding and frequency control.  
- Select thresholds based on **business goals** (e.g., maximize recall or precision).  
- Support decisions with **A/B testing** using candidate thresholds.  
- Verify **probability calibration** and **temporal stability** before deployment.

---

## Executive Summary

- **Strongest predictors**: recent user interactions (3–7 day CTR, user × campaign/brand/category).  
- **Model performance**: AUC improved from ~0.576 (single models) to ~0.596 (ensemble).  
- **Value**: Higher AUC translates into more clicks captured for the same budget.  
- **Next steps**:  
  1. Engineer additional short-window and time-decay features.  
  2. Incorporate weekday scheduling in campaigns.  
  3. Validate calibration and stability of the blended model.  
  4. Explore lightweight sequence models for brand/category behavior.

---
