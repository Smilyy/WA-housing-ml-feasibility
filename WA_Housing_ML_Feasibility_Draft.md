# WA Housing Price Prediction — ML Feasibility & Discovery Notes (Draft)

> This document provides an initial feasibility assessment to support early decision-making and scoping.

---

## 1. Problem Framing & Objective

### 1.1 Use Context (Assumed)

The current focus of this work is to explore residential housing prices in Western Australia (WA) and to assess what level of insight can be derived from free or publicly accessible data sources.

At this stage, the work is exploratory in nature. The objective is not to build an automated valuation product, but to evaluate feasibility, identify key price drivers, and understand data limitations before committing to further development.

The outcomes of this phase are expected to inform next steps, such as whether deeper modelling, additional data acquisition, or a more formal proof of concept (PoC) would be justified.

### 1.2 ML Objective

The machine learning objective is to estimate residential property sale prices in Western Australia using historical transaction data. The task is framed as a supervised regression problem, with sale price as the target variable.

### 1.3 Scope Decisions & Assumptions

To keep early exploration tractable, the following scope assumptions are made:

* **Geography:** analysis at suburb or postcode level, subject to data availability and coverage
* **Time:** transaction‑date‑based modelling with limited temporal aggregation
* **Property type:** residential properties only

These choices are intended to reduce complexity during the feasibility phase and may be revised as data constraints and project goals become clearer.

---

## 2. High‑Level Literature & Industry Approaches

### 2.1 Common Modelling Approaches

Residential housing price prediction is a well‑studied applied problem. Both academic literature and industry practice typically favour robust and interpretable models over highly complex architectures, particularly where data quality is uneven. Common approaches include:

#### *Hedonic regression models*

Hedonic pricing models are traditional econometric approaches that express property price as a function of structural, locational, and neighbourhood attributes (e.g. dwelling size, number of rooms, location). They are widely used as interpretable baselines in real estate appraisal and automated valuation models (AVMs).

* Background: [Hedonic regression (Wikipedia)](https://en.wikipedia.org/wiki/Hedonic_regression)
* Example implementation: *A GitHub repository focused on the development of hedonic models for housing rents and sale prices* ([ual/hedonic-models](https://github.com/ual/hedonic-models))

#### *Tree‑based ensemble models*

Tree‑based methods such as Random Forests and Gradient Boosting (including XGBoost and LightGBM) are commonly used for housing price prediction due to strong performance on tabular data, ability to capture nonlinear relationships, and relatively modest feature‑engineering requirements.

* Background: [Gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting)
* Libraries: [XGBoost](https://github.com/dmlc/xgboost), [LightGBM](https://github.com/microsoft/LightGBM)

#### *Additional ML baselines*

Simpler machine learning baselines such as linear regression variants, k‑nearest neighbours (KNN), and support vector regression (SVR) are often evaluated during early experimentation to benchmark performance and understand data sensitivity, rather than as final production choices.

* Reference: [Literature review on ML for house price prediction](https://www.researchgate.net/publication/372080726_A_Literature_Review_on_Using_Machine_Learning_Algorithm_to_Predict_House_Prices)

#### *Spatial‑aware modelling approaches*

Housing prices exhibit strong spatial autocorrelation. Spatial‑aware approaches explicitly account for this dependency using techniques such as geographically weighted regression (GWR), spatial lag models, or spatial features within ML pipelines. These methods can improve performance in dense markets but are highly dependent on geographic coverage and data quality.

* Reference: [Springer — spatial modelling of housing prices](https://link.springer.com/article/10.1007/s11146-022-09915-y)
* Libraries: [pyGWR](https://github.com/mkordi/pygwr), [GNNWR](https://github.com/zjuwss/gnnwr)

#### *Hybrid and ensemble approaches*

Some applied systems combine multiple models or feature representations (e.g. hedonic features combined with ML outputs or stacked regressors). These approaches aim to improve robustness and stability rather than maximise raw predictive accuracy.

* Reference: [MDPI — hybrid housing price models](https://www.mdpi.com/2073-445X/11/3/334)

#### *Deep learning and multimodal methods*

Deep learning approaches that incorporate images, text descriptions, or learned spatial embeddings primarily appear in research settings or large‑scale platforms with access to rich, high‑volume data. When relying on free or public tabular datasets, these methods often provide limited benefit relative to their complexity and interpretability cost.

* Reference: [arXiv — multimodal house price prediction](https://arxiv.org/abs/2409.05335)
* Example repositories: [mhpp](https://github.com/4P0N/mhpp), [Multi‑Modal House Price Estimation](https://github.com/Mehrab-Kalantari/Multi-Modal-House-Price-Estimation)

### 2.2 Key Takeaways

Across both research and applied systems, feature quality and spatial representation typically have a greater impact on performance than model complexity. As a result, simpler and interpretable approaches are generally preferred during early feasibility and discovery phases.

---

## 3. Data & Feature Hypotheses

### 3.1 Core Features — Rationale and Sources

The table below summarises core feature hypotheses, why each feature is needed, and indicative data sources. This list is intended to guide early exploration rather than define a final feature set.

| Feature                             | Why it matters                            | Indicative data source                   |
| ----------------------------------- | ----------------------------------------- | ---------------------------------------- |
| Sale price                          | Target variable                           | WA Sales Evidence Data                   |
| Transaction date                    | Captures market cycles and temporal drift | WA Sales Evidence Data                   |
| Suburb / postcode                   | Primary location proxy                    | WA Sales Evidence Data, SLIP boundaries  |
| Latitude / longitude (if available) | Enables spatial modelling                 | SLIP spatial datasets                    |
| Dwelling type                       | Controls for structural heterogeneity     | WA Sales Evidence Data                   |
| Bedrooms / bathrooms                | Core structural value drivers             | WA Sales Evidence Data                   |
| Land size / floor area              | Differentiates property scale             | WA Sales Evidence Data (where available) |
| Year built (if available)           | Captures age and depreciation effects     | WA Sales Evidence Data                   |
| Median household income             | Proxy for local purchasing power          | ABS Census                               |
| Population density                  | Indicates demand pressure                 | ABS Census                               |
| Tenure mix                          | Reflects neighbourhood stability          | ABS Census                               |
| Regional price index                | Macro‑economic context                    | WA Regional Price Index                  |

Feature inclusion is contingent on availability, geographic coverage, and data quality, and will be validated during data discovery.

### 3.2 Potential Free Data Sources

#### Primary transaction and geographic data (WA)

* **WA Sales Evidence Data** — transaction prices and basic property attributes
  [https://catalogue.data.wa.gov.au/dataset/sales-evidence-data](https://catalogue.data.wa.gov.au/dataset/sales-evidence-data)
  *Note:* Full access requires licensing and fees. Sample extracts in `.dat` and `.xlsx` formats are available for preliminary exploration.

* **DataWA / SLIP spatial datasets** — boundaries and geographic reference layers
  [https://catalogue.data.wa.gov.au/](https://catalogue.data.wa.gov.au/)

#### Socio‑economic and macro context

* **ABS Census and housing statistics** — income, population, dwelling characteristics
  [https://www.abs.gov.au/](https://www.abs.gov.au/)

* **WA Regional Price Index (RPI)** — region‑level cost‑of‑living comparisons
  [https://catalogue.data.wa.gov.au/dataset/regional-price-index-western-australia](https://catalogue.data.wa.gov.au/dataset/regional-price-index-western-australia)
  *Constraint:* Useful as a regional macro‑context feature rather than a transaction‑level input.

#### Broader / comparative datasets (benchmarking only)

* **UK Property Price Paid Data**
  [https://clickhouse.com/docs/getting-started/example-datasets/uk-price-paid](https://clickhouse.com/docs/getting-started/example-datasets/uk-price-paid)
  *Constraint:* Geography and market dynamics differ significantly from WA; suitable for method testing rather than model transfer.

* **International Residential Property Price Indices (BIS)**
  [https://data.bis.org/topics/RPP](https://data.bis.org/topics/RPP)
  *Constraint:* Highly aggregated; useful for macro‑trend comparison or validation, not property‑level prediction.

* **Multimodal Houses Dataset (images + text)**
  [https://github.com/emanhamed/Houses-dataset](https://github.com/emanhamed/Houses-dataset)
  *Constraint:* Limited in scale and not representative of WA; suitable only for experimental exploration of multimodal or deep learning approaches.

---

## 4. Expected Data Characteristics

### 4.1 Data Volume

At this stage, the overall dataset size is uncertain, as only sample transaction extracts are currently accessible. Actual volume will depend on the selected temporal window and licensing constraints. Coverage is expected to be uneven, with metropolitan areas likely over‑represented relative to regional or remote locations. Transaction density may vary substantially across suburbs and years, reducing the effective sample size after quality filtering.

### 4.2 Distributional Considerations

Housing prices typically exhibit strong right‑skew. Log‑transformation of the target variable is therefore likely required to stabilise model behaviour. Additional challenges include spatial sparsity, heterogeneous variance across locations and property types, and temporal non‑stationarity driven by macro‑economic cycles. These factors place practical limits on achievable model performance and necessitate cautious interpretation of results.

---

## 5. Data Risks & Constraints

Key risks relate primarily to **data completeness, consistency, and representativeness** rather than modelling capability. Essential structural attributes may be missing or inconsistently recorded, and spatial bias toward metropolitan areas is expected. Temporal drift further limits the stability of patterns learned from historical data. These risks will need to be validated through hands‑on data profiling before committing to downstream modelling.

---

## 6. Baseline & Evaluation Strategy

Initial evaluation will focus on feasibility rather than optimisation.

* **Baseline:** suburb‑ or postcode‑level median prices, potentially stratified by dwelling type
* **Candidate models:** tree‑based regressors (e.g. gradient boosting, random forests)
* **Metrics:** MAE and RMSE, interpreted relative to baseline performance

---

## 7. Feasibility Signals & Stop Criteria

### 7.1 Positive Signals

* Sufficient transaction density across a meaningful subset of regions
* Availability of core structural and location features with acceptable completeness
* Temporal coverage spanning multiple market phases

### 7.2 No‑Go Signals

* Severe missingness or inconsistency in sale price or location identifiers
* Conflicting definitions across data sources
* Usable data confined to a very small geographic area

---

## 8. Summary & Next‑Step Options

This assessment suggests that ML‑based housing price modelling in WA may be feasible, subject to data availability and quality. Feasibility is currently constrained more by data characteristics than by modelling techniques. Reasonable next steps include a focused data audit, followed by a small‑scale PoC to validate signal strength and baseline performance before any further investment.
