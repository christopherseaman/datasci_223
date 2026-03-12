# Decisions for lecture_05.md Refactoring (Round 3)

This document outlines potential ordering improvements after the second round of edits.

---

## 1. Content Flow After Demo 2
DECISION: no reordering.

*   **Issue:** Reviewing the logical flow of content following LIVE DEMO 2 (currently starting line 1037).
*   **Current Structure:**
    1. Model Interpretation (ends line 1036)
    2. LIVE DEMO 2 (Feature Eng, Model Comparison, Interpretation - starts line 1037)
    3. Practical Data Preparation (Intro, OHE, SMOTE, Order of Ops - starts line 1042)
    4. LIVE DEMO 3 (Imbalanced, Encoding, SMOTE, Interpretation - starts line 1148)
*   **Analysis:** The current order appears logical. The "Practical Data Preparation" section introduces techniques (Encoding, SMOTE, Order) that are then demonstrated in Demo 3. Demo 2 focuses more on applying models and interpretation after feature engineering (covered earlier).
*   **Recommendation:** **No major reordering needed** for the sections following Demo 2.

---
DECISION: remove all time series review content except for memes (if any)

## 2. Context for Demo 2 Time Series Content

*   **Issue:** Demo 2's description (line 1039) mentions "feature engineering from time series sensor data," but the main review section for time series features ("Time Series Features: Quick Review") is now much earlier (lines 784-826).
*   **Location 1:** Demo 2 description (Line 1039).
*   **Location 2:** Time Series Features review section (Lines 784-826).
*   **Recommendation:** **Consider adding a brief reminder sentence** just before the Demo 2 header (i.e., insert at line 1037) that links back to the earlier time series feature section. For example:
    ```markdown
    (Remember the time series feature extraction techniques we reviewed earlier - see [Time Series Features: Quick Review](#time-series-features-quick-review))
    ```
    *(Note: Markdown links to headers might need adjustment based on final rendering)*. This helps bridge the gap between the concept review and its application in the demo.

---
# Decisions for lecture_05.md Refactoring (Round 4)

This document outlines findings after a fresh review of `lectures/05/lecture_05.md` based on its current state.

---

## 3. Missing AUROC Image
DECISION: manually reverted

*   **Issue:** The image `![AUROC example](media/auroc.png)` is missing from the document. It was likely removed inadvertently during previous edits that targeted duplicated text blocks.
*   **Location:** This image originally followed the detailed evaluation metrics list.
*   **Recommendation:** **Re-insert** the image `![AUROC example](media/auroc.png)` into the "AUC/AUROC: Area Under the ROC Curve" section. A good placement would be after the `roc_auc_score` reference card (i.e., insert at line 251) to visually complement the AUC concept before the Google ML images.

---

## 4. Orphaned Demo Header
DECISION: I believe I have manually resolved this

*   **Issue:** A generic demo header `## 🏋️ LIVE DEMO` exists at line 780 without a specific demo link or description. It appears misplaced between the "How models fail" section and the "Time Series Features" section.
*   **Location:** Line 780.
*   **Recommendation:** **Remove** this orphaned header (Lines 780-783, including the blank line and link underneath if they belong to it) as it doesn't correspond to a structured demo section like Demo 1, 2, or 3.

---

## 5. Time Series Review Section Present
DECISION: remove

*   **Issue:** The section "### Time Series Features: Quick Review" (Lines 784-827) is still present, despite a previous decision to remove it.
*   **Location:** Lines 784-827.
*   **Recommendation:** **Confirm removal.** If the decision to remove this section still stands, delete lines 784-827. If it should be kept, no action is needed for this point.

---