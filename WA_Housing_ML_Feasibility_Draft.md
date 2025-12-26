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

- Geography: analysis at suburb or postcode level, subject to data availability and coverage

- Time: transaction-date based modelling, with limited temporal aggregation

- Property type: residential properties only

These scope choices are intended to reduce complexity during early exploration and may be adjusted as data constraints and project goals become clearer.

---

## 2. High-Level Literature & Industry Approaches

### 2.1 Common Modeling Approaches
Residential housing price prediction is a well-studied applied problem, and industry practice typically favours robust, interpretable models over highly complex architectures. Common approaches can be broadly categorized as follows:

#### *Hedonic regression models*  
Hedonic pricing models are traditional econometric approaches that express property price as a function of structural, locational, and neighbourhood attributes (e.g., dwelling size, number of rooms, location). They are widely used as interpretable baselines in real estate appraisal and automated valuation models (AVMs), particularly where transparency is required.  
Reference: https://en.wikipedia.org/wiki/Hedonic_regression  
A GitHub repository developing hedonic models for housing rents and sale prices: https://github.com/ual/hedonic-models

#### *Tree-based ensemble models*  
Tree-based machine learning methods such as Random Forest and Gradient Boosting (including XGBoost and LightGBM) are commonly used in applied housing price prediction due to their strong performance on tabular data. These models capture nonlinear relationships and feature interactions with relatively modest feature engineering and are often competitive with more complex methods in practice.  
Reference: https://en.wikipedia.org/wiki/Gradient_boosting  
XGBoost Github Repo: https://github.com/dmlc/xgboost  
LightGBM Guthub Repo: https://github.com/microsoft/LightGBM

#### *Additional machine learning baselines*  
Beyond tree-based ensembles, practitioners frequently evaluate simpler ML baselines such as linear regression variants, k-nearest neighbours (KNN), and support vector regression (SVR). These models are typically used during early experimentation to benchmark performance and understand data sensitivity rather than as final production choices.  
Reference: https://www.researchgate.net/publication/372080726_A_Literature_Review_on_Using_Machine_Learning_Algorithm_to_Predict_House_Prices

#### *Spatial-aware modelling approaches*  
Housing prices exhibit strong spatial autocorrelation, where nearby properties tend to have similar values. Spatial-aware approaches explicitly account for this dependency through techniques such as geographically weighted regression, spatial lag models, or the inclusion of spatial features and residual structures within ML pipelines. These methods are particularly relevant in dense urban markets but depend heavily on data quality and geographic coverage.  
Reference: https://link.springer.com/article/10.1007/s11146-022-09915-y  
A GWR Model Python library: https://github.com/mkordi/pygwr  
A PyTorch implementation of the Geographically Neural Network Weighted Regression (GNNWR) and its extensions Github Repo: https://github.com/zjuwss/gnnwr?utm_source=chatgpt.com

#### *Hybrid and ensemble approaches*  
Some applied systems combine multiple models or feature representations, such as blending hedonic features with machine learning outputs or stacking multiple regressors. These hybrid approaches aim to improve robustness and stability rather than maximise raw predictive accuracy and are typically explored after simpler baselines are established.  
Reference: [https://www.mdpi.com/2073-445X/11/3/334](https://www.mdpi.com/2073-445X/11/3/334)

#### *Deep learning and multi-modal methods*  
Deep learning approaches, including neural networks that incorporate images, text descriptions, or learned spatial embeddings, appear primarily in research settings or large-scale platforms with access to rich, high-volume data. In most industry-level tabular housing datasets—particularly when relying on free or public data—deep learning is often unnecessary and offers limited benefit relative to its complexity and interpretability cost.  
Reference: https://arxiv.org/abs/2409.05335  
A Multi-Modal Deep Learning Based Approach for House Price Prediction Github Repo: https://github.com/4P0N/mhpp?utm_source=chatgpt.com  
House price estimation from visual and textual features using both machine learning and deep learning models Github Repo:https://github.com/Mehrab-Kalantari/Multi-Modal-House-Price-Estimation?utm_source=chatgpt.com


### 2.2 Key Takeaways
Across studies and applied systems, feature quality and spatial representation typically have a greater impact on performance than model complexity, making simpler, well-understood methods preferable during early feasibility and discovery phases.

---

## 3. Data & Feature Hypotheses

### 3.1 Core Feature Categories
Based on prior housing price modelling work, the following high-level feature categories are expected to explain the majority of observed price variation. These categories reflect commonly accepted economic drivers of residential property value and are intended to guide early exploration rather than define a fixed or exhaustive feature set.

*Location features*  
Location is typically the single strongest determinant of housing prices. Features such as suburb or postcode identifiers, geographic coordinates, or region-level indicators act as proxies for a wide range of latent factors, including neighbourhood desirability, access to employment, and local demand–supply dynamics.

*Property attributes*  
Structural characteristics describe the intrinsic qualities of a dwelling and are essential for differentiating prices within the same location. Attributes such as dwelling type, number of bedrooms and bathrooms, land size, internal area, and tenure type directly influence perceived utility and market value.

*Temporal features*  
Housing prices evolve over time in response to market cycles, policy changes, and macroeconomic conditions. Transaction dates or coarse temporal indicators (e.g. year or quarter) help capture these dynamics and reduce bias arising from pooling transactions across different market phases.

*Socio-economic context*  
Area-level socio-economic indicators provide context for local purchasing power and demand conditions. Features such as income levels, population density, tenure mix, and housing affordability metrics help explain systematic price differences across regions that are not fully captured by property attributes alone.

*Optional amenity indicators*  
Where data coverage and quality permit, proximity to amenities such as schools, transport, or services can act as secondary drivers of value by influencing neighbourhood attractiveness. These features are considered optional due to variability in availability and consistency across public data sources.

These feature categories provide a structured starting point for feasibility assessment, with final feature selection contingent on data availability, coverage, and observed signal strength.

### 3.2 Potential Free Data Sources
The initial analysis assumes reliance on free or publicly available datasets, which may include the following.

#### Primary transaction and geographic data (WA)

WA Sales Evidence Data — property transaction records including sale prices and attributes  
https://catalogue.data.wa.gov.au/dataset/sales-evidence-data  
Full access requires a licence and associated fees. Sample extracts in .dat and .xlsx formats are available for preliminary exploration.

DataWA / SLIP spatial datasets — boundaries, regions, and geographic reference layers  
https://catalogue.data.wa.gov.au/

#### Socio-economic and macro context

ABS Census and housing statistics — income, population, dwelling characteristics  
https://www.abs.gov.au/

Western Australia Regional Price Index (RPI) — region-level cost-of-living comparisons  
https://catalogue.data.wa.gov.au/dataset/regional-price-index-western-australia  
Suitable for regional macro-context analysis, not transaction-level inputs.

#### Broader / Comparative Datasets (Benchmarking)

The following datasets are not WA-specific and are not intended for direct production use, but are commonly used for benchmarking, validation, or methodological exploration:

UK Property Price Paid Data — Large, fully open dataset of residential transactions in England and Wales, often used in academic and ML baseline studies.  
https://clickhouse.com/docs/getting-started/example-datasets/uk-price-paid  
Constraint: Geography and market dynamics differ significantly from WA, so this is better suited for method testing than direct model transfer.

International Residential Property Price Indices (BIS)  
Quarterly price indices across multiple countries.  
https://data.bis.org/topics/RPP  
Constraint: Highly aggregated, making it useful for macro trend comparison or validation, but not for property-level prediction.

Multimodal Houses Dataset (images + text)  
Small research dataset (~2k properties) with images and textual metadata.  
https://github.com/emanhamed/Houses-dataset  
Constraint: Limited in scale and not representative of the WA market, making it suitable only for experimental exploration of multimodal or deep learning methods.

Note: For this exploratory phase, priority should be given to WA-specific transaction data (Sales Evidence Data samples) combined with ABS socio-economic context and regional macro indicators. Broader and international datasets are best used for method validation, baseline comparisons, or feasibility assessment, rather than as direct training data.
Feature inclusion is contingent on data availability, geographic coverage, and data quality, and will be validated during data discovery before any modelling decisions are finalised.

---

## 4. Expected Data Characteristics

### 4.1 Data Volume
Based on typical availability of publicly accessible residential transaction data in Western Australia, the overall dataset size is currently unknown, as only a sample subset is available at this stage. Actual volume will depend on the temporal window selected and the completeness of accessible sources. Spatial and temporal coverage is expected to be uneven, with metropolitan areas likely to be over-represented relative to regional or remote locations. Transaction density may vary substantially across suburbs and years, and some regions may exhibit sparse or discontinuous records. In addition, certain property types or market segments may be under-represented. As a result, the effective sample size after applying quality, coverage, and consistency constraints may be significantly smaller than the raw record count.

### 4.2 Data Distribution Considerations
Residential housing prices typically exhibit strong right-skewed distributions, where a small number of high-value properties exert disproportionate influence on aggregate statistics. As a result, transformation of the target variable (e.g. log-price) is commonly required to stabilise model behaviour. Additional distributional challenges include spatial sparsity—particularly in regional or low-turnover suburbs—which may limit model reliability in those areas, as well as heterogeneous variance across locations and property types that leads to uneven prediction uncertainty. Housing markets are also subject to temporal cycles, where macroeconomic conditions and policy changes introduce non-stationarity over time. Together, these characteristics place practical limits on achievable model performance and reinforce the need for cautious interpretation of predictions, especially in low-data or rapidly changing market segments.

---

## 5. Data Risks & Constraints
The primary risks in applying machine learning to publicly accessible housing transaction data relate to **data completeness, consistency, and representativeness** rather than model capability. Key structural features—such as land size, dwelling attributes, or precise location identifiers—may be missing, inconsistently recorded, or defined differently across data sources, limiting their usefulness as reliable predictors.

At this stage, I have not yet conducted a detailed, hands-on audit of all available datasets. The observations outlined here are therefore based on common characteristics observed in comparable public housing datasets, rather than confirmed data profiling results. As such, these risks should be treated as **anticipated constraints** to be validated during any subsequent data exploration phase.

Spatial bias is also expected, with metropolitan areas likely to dominate available records while regional and remote locations exhibit sparse coverage. This imbalance can lead to models that perform well in high-density urban markets but generalise poorly to lower-turnover regions. In addition, housing markets are subject to **temporal drift**, where changing macroeconomic conditions, interest rates, and policy interventions alter pricing dynamics over time, reducing the stability of patterns learned from historical data.

Finally, reliance on free or open datasets introduces inherent noise and potential inconsistencies, including delayed updates, incomplete transactions, and mismatched definitions across sources. Together, these risks directly constrain achievable model performance and reliability, and must be carefully considered when interpreting results or assessing the feasibility of downstream applications.

---

## 6. Baseline & Evaluation Strategy (Conceptual)
To ground early results and avoid overfitting to model complexity, a small number of simple baselines will be used as reference points.  
- **Baseline reference:**
Suburb- or postcode-level median sale prices, potentially stratified by basic property attributes (e.g. dwelling type), to establish a non-ML benchmark.

- **Candidate model family:**
Tree-based regression models (e.g. gradient-boosted trees or random forests) are suitable for initial feasibility testing due to their strong performance on tabular data, ability to capture non-linear relationships, and relatively low feature engineering overhead.

- **Evaluation metrics:**
Model performance will be assessed using standard regression metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE), with interpretation focused on relative improvement over the baseline rather than absolute accuracy.

At this stage, the goal of evaluation is to assess signal strength and feasibility, not to optimise or productionise models.

---

## 7. Feasibility Signals & Stop Criteria

### 7.1 Positive Feasibility Signals
The project would be considered technically feasible if exploratory analysis indicates **sufficient transaction density** across a meaningful subset of regions, allowing models to learn stable patterns rather than isolated cases. Availability of **core structural features** (e.g. basic dwelling attributes and coarse location identifiers) with acceptable completeness would further support feasibility. In addition, **reasonable temporal coverage** across multiple market phases would be required to mitigate the impact of short-term volatility and temporal bias.

### 7.2 Stop / No-Go Signals
The project should be reconsidered or halted if analysis reveals **severe missingness or inconsistency** in essential predictors such as location, dwelling size, or sale price. Similarly, **inconsistent price definitions or recording standards** across data sources would undermine model validity and comparability. Finally, if usable data coverage is largely confined to a small number of metropolitan areas, with insufficient representation elsewhere, model outputs would have limited generalisability and practical value.

---

## 8. Summary & Next-Step Options

This assessment indicates that applying machine learning to residential housing price prediction in Western Australia may be feasible, subject to the availability, quality, and coverage of publicly accessible data. Feasibility at this stage should be regarded as conditional, with outcomes primarily constrained by data characteristics rather than modelling techniques.

If initial feasibility is confirmed through further validation, reasonable next steps would include a **focused data audit** to verify key assumptions, followed by a small-scale proof of concept **(PoC)** to assess signal quality and baseline performance. Based on those findings, **scope refinement** with relevant stakeholders would be required to align expectations, define acceptable coverage, and determine whether further investment is justified.
