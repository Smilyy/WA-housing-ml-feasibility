# WA Housing Price Prediction — ML Feasibility & Discovery Notes (Draft)

> This document is an initial feasibility.  
> It is intended to support early decision-making,  
> not to define a final solution.

---

## 1. Problem Framing & Objective

### 1.1 Use Context (Assumed)
The current focus of this work is to explore residential housing prices in Western Australia and to understand what level of insight can be derived from free/publicly available data.

At this stage, the work is exploratory in nature. The aim is not to build an automated valuation product, but to assess feasibility, identify key price drivers, and understand the limitations of the data before committing to further development.

The outcomes of this phase are expected to inform next steps, such as whether deeper modelling, additional data sources, or a more formal PoC would be justified.

### 1.2 ML Objective
The ML objective is to estimate residential property sale prices in Western Australia based on historical transaction data. The task is framed as a regression problem, with sale price as the target variable.

### 1.3 Scope Decisions & Assumptions
To keep the initial analysis tractable, the following scope decisions are assumed:

Geography: analysis at suburb or postcode level, subject to data availability and coverage

Time: transaction-date based modelling, with limited temporal aggregation

Property type: residential properties only

These scope choices are intended to reduce complexity during early exploration and may be adjusted as data constraints and project goals become clearer.

---

## 2. High-Level Literature & Industry Approaches

### 2.1 Common Modeling Approaches
Residential housing price prediction is a well-studied applied problem, and industry practice typically favours robust, interpretable models over highly complex architectures. Common approaches can be broadly categorized as follows:

*Hedonic regression models*  
Hedonic pricing models are traditional econometric approaches that express property price as a function of structural, locational, and neighbourhood attributes (e.g., dwelling size, number of rooms, location). They are widely used as interpretable baselines in real estate appraisal and automated valuation models (AVMs), particularly where transparency is required.  
Reference: https://en.wikipedia.org/wiki/Hedonic_regression  
A Development of hedonic models of rents and sales prices for housing on github: https://github.com/ual/hedonic-models

*Tree-based ensemble models*  
Tree-based machine learning methods such as Random Forest and Gradient Boosting (including XGBoost and LightGBM) are commonly used in applied housing price prediction due to their strong performance on tabular data. These models capture nonlinear relationships and feature interactions with relatively modest feature engineering and are often competitive with more complex methods in practice.
Reference: https://en.wikipedia.org/wiki/Gradient_boosting  
XGBoost Github Repo: https://github.com/dmlc/xgboost  
LightGBM Guthub Repo: https://github.com/microsoft/LightGBM

*Additional machine learning baselines*  
Beyond tree-based ensembles, practitioners frequently evaluate simpler ML baselines such as linear regression variants, k-nearest neighbours (KNN), and support vector regression (SVR). These models are typically used during early experimentation to benchmark performance and understand data sensitivity rather than as final production choices.  
Reference: https://www.researchgate.net/publication/372080726_A_Literature_Review_on_Using_Machine_Learning_Algorithm_to_Predict_House_Prices

*Spatial-aware modelling approaches*  
Housing prices exhibit strong spatial autocorrelation, where nearby properties tend to have similar values. Spatial-aware approaches explicitly account for this dependency through techniques such as geographically weighted regression, spatial lag models, or the inclusion of spatial features and residual structures within ML pipelines. These methods are particularly relevant in dense urban markets but depend heavily on data quality and geographic coverage.  
Reference: https://link.springer.com/article/10.1007/s11146-022-09915-y  
A GWR Model Python library: https://github.com/mkordi/pygwr  
A PyTorch implementation of the Geographically Neural Network Weighted Regression (GNNWR) and its extensions: https://github.com/zjuwss/gnnwr?utm_source=chatgpt.com

*Hybrid and ensemble approaches*  
Some applied systems combine multiple models or feature representations, such as blending hedonic features with machine learning outputs or stacking multiple regressors. These hybrid approaches aim to improve robustness and stability rather than maximise raw predictive accuracy and are typically explored after simpler baselines are established.  
Reference: [https://www.mdpi.com/2073-445X/11/3/334](https://www.mdpi.com/2073-445X/11/3/334)

*Deep learning and multi-modal methods*  
Deep learning approaches, including neural networks that incorporate images, text descriptions, or learned spatial embeddings, appear primarily in research settings or large-scale platforms with access to rich, high-volume data. In most industry-level tabular housing datasets—particularly when relying on free or public data—deep learning is often unnecessary and offers limited benefit relative to its complexity and interpretability cost.  
Reference: https://arxiv.org/abs/2409.05335  
A Multi-Modal Deep Learning Based Approach for House Price Prediction Github Repo: https://github.com/4P0N/mhpp?utm_source=chatgpt.com  
House price estimation from visual and textual features using both machine learning and deep learning models Github Repo:https://github.com/Mehrab-Kalantari/Multi-Modal-House-Price-Estimation?utm_source=chatgpt.com


### 2.2 Key Takeaways
Across studies and applied systems, feature quality and spatial representation typically have a greater impact on performance than model complexity, making simpler, well-understood methods preferable during early feasibility and discovery phases.

---

## 3. Data & Feature Hypotheses

### 3.1 Core Feature Categories
Based on prior housing price modelling work, the following high-level feature categories are expected to explain the majority of price variation:

Location features
e.g. suburb or postcode identifiers, geographic coordinates, or region-level indicators.

Property attributes
e.g. dwelling type, number of bedrooms/bathrooms, land size, internal area, tenure type.

Temporal features
e.g. transaction date, year/quarter indicators to capture market cycles and trends.

Socio-economic context
e.g. income levels, population density, tenure mix, housing affordability metrics at area level.

Optional amenity indicators
e.g. proximity to schools, transport, or services, where coverage and quality permit.

These categories are intended to guide early exploration rather than define a fixed or exhaustive feature set.

### 3.2 Potential Free Data Sources (Indicative)
The initial analysis assumes reliance on free or publicly available datasets, which may include the following.

Primary transaction and geographic data (WA)

WA Sales Evidence Data — property transaction records including sale prices and attributes
https://catalogue.data.wa.gov.au/dataset/sales-evidence-data

Note: full access is subject to licensing/fees; sample extracts in formats such as .dat or .xlsx can be used for early exploration.

DataWA / SLIP spatial datasets — boundaries, regions, and geographic reference layers
https://catalogue.data.wa.gov.au/

Socio-economic and macro context

ABS Census and housing statistics — income, population, dwelling characteristics
https://www.abs.gov.au/

Western Australia Regional Price Index (RPI) — region-level cost-of-living comparisons
https://catalogue.data.wa.gov.au/dataset/regional-price-index-western-australia

Useful as a regional macro context feature, not as transaction-level input.
Broader / Comparative Datasets (Benchmarking & Methods)

The following datasets are not WA-specific and are not intended for direct production use, but are commonly used for benchmarking, validation, or methodological exploration:

UK Property Price Paid Data
Large, fully open dataset of residential transactions in England and Wales, often used in academic and ML baseline studies.
https://clickhouse.com/docs/getting-started/example-datasets/uk-price-paid

Constraint: geography and market dynamics differ significantly from WA; suitable for method testing only, not model transfer.

International Residential Property Price Indices (BIS)
Quarterly price indices across multiple countries.
https://data.bis.org/topics/RPP

Constraint: highly aggregated; useful for macro trend comparison or validation, not for property-level prediction.

Multimodal Houses Dataset (images + text)
Small research dataset (~2k properties) with images and textual metadata.
https://github.com/emanhamed/Houses-dataset

Constraint: limited scale and not representative of WA; suitable only for experimental exploration of multimodal or deep learning approaches.
Recommendation:
For this exploratory phase, priority should be given to WA-specific transaction data (Sales Evidence Data samples) combined with ABS socio-economic context and regional macro indicators. Broader and international datasets are best used for method validation, baseline comparisons, or feasibility assessment, rather than as direct training data.
Feature inclusion is contingent on data availability, geographic coverage, and data quality, and will be validated during data discovery before any modelling decisions are finalised.

---

## 4. Expected Data Characteristics

### 4.1 Data Volume (Order-of-Magnitude)
Based on typical availability of publicly accessible residential transaction data in Western Australia, the overall dataset size is currently unknown, as only a sample subset is available at this stage. Actual volume will depend on the temporal window selected and the completeness of accessible sources. Spatial and temporal coverage is expected to be uneven, with metropolitan areas likely to be over-represented relative to regional or remote locations. Transaction density may vary substantially across suburbs and years, and some regions may exhibit sparse or discontinuous records. In addition, certain property types or market segments may be under-represented. As a result, the effective sample size after applying quality, coverage, and consistency constraints may be significantly smaller than the raw record count.

### 4.2 Data Distribution Considerations
Residential housing prices typically exhibit strong right-skewed distributions, where a small number of high-value properties exert disproportionate influence on aggregate statistics. As a result, transformation of the target variable (e.g. log-price) is commonly required to stabilise model behaviour. Additional distributional challenges include spatial sparsity—particularly in regional or low-turnover suburbs—which may limit model reliability in those areas, as well as heterogeneous variance across locations and property types that leads to uneven prediction uncertainty. Housing markets are also subject to temporal cycles, where macroeconomic conditions and policy changes introduce non-stationarity over time. Together, these characteristics place practical limits on achievable model performance and reinforce the need for cautious interpretation of predictions, especially in low-data or rapidly changing market segments.

---

## 5. Data Risks & Constraints

This is a critical lead section.

Discuss risks such as:
- missing or inconsistent key features,
- spatial bias toward metro areas,
- temporal drift in housing markets,
- noise or inconsistencies in free datasets.

End with a clear statement:
> These risks directly constrain achievable model performance and reliability.

---

## 6. Baseline & Evaluation Strategy (Conceptual)

Keep this light and non-technical.

- Baseline reference (e.g., suburb-level median prices),
- Candidate model family for feasibility testing (tree-based regression),
- Evaluation metrics (e.g., MAE, RMSE).

No implementation detail.

---

## 7. Feasibility Signals & Stop Criteria

This section supports decision-making.

### 7.1 Positive Feasibility Signals
Examples:
- sufficient transaction density per region,
- availability of core structural features,
- reasonable temporal coverage.

### 7.2 Stop / No-Go Signals
Examples:
- severe missingness in location or size data,
- inconsistent price definitions across sources,
- insufficient coverage outside limited regions.

---

## 8. Summary & Next-Step Options

Close calmly and professionally:
- Restate feasibility as conditional, not guaranteed.
- Outline possible next steps *if* feasibility is confirmed:
  - focused data audit,
  - small-scale PoC,
  - scope refinement with stakeholders.

Avoid commitments or timelines.
