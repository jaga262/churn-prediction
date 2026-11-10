# End-to-End Customer Churn Prediction Using Quality-Gated Machine Learning Operations

**Author One** (*Computer Science Dept.*, `author1@institution.edu`) | **Author Two** (*Computer Science Dept.*, `author2@institution.edu`) | **Author Three** (*Computer Science Dept.*, `author3@institution.edu`)

***Abstract - Telecommunications providers experience recurring revenue loss when subscribers cancel contracts without proactive retention intervention. Although supervised classifiers identify high-risk accounts in static notebooks, standalone scripts often degrade in production due to schema corruption, untracked dataset revisions, missing artifact dependencies, and covariate drift. This paper presents an automated ten-stage machine learning operations platform for customer churn prediction built around a modular Random Forest and gradient boosting modeling framework. Starting from a baseline cohort of 7,043 customer records, the system enforces strict schema validation, stratified split preservation, cryptographic batch lineage tracking across 7,344 cumulative records, and reproducible experiment logging. Candidate models must exceed a 0.70 recall threshold before an automated lifecycle controller compares them against the live production model; candidate version seventeen achieved 0.7219 recall and was correctly archived behind the existing 0.7807 recall production champion. Fault injection verification confirmed safe halting or automated artifact recovery across fifteen tests, while drift analysis quantified feature shifts via statistical divergence tests. Finally, containerized FastAPI serving achieved 19.38 milliseconds average local latency, passing twenty-nine automated tests and confirming identical predictions inside Docker.***

***Index Terms - Churn prediction, drift monitoring, machine learning operations, model governance.***


## Introduction

Customer attrition represents a critical operational challenge in subscription telecommunications, where acquiring a new subscriber costs substantially more than retaining an existing customer. Predictive machine learning models enable retention teams to target at-risk accounts early; however, empirical evidence shows that predictive code comprises only a small fraction of a sustainable production system [1]. In practice, standalone training notebooks fail when exposed to malformed incoming records, inconsistent preprocessing transformations between training and inference, unversioned training snapshots, and post-deployment distribution shifts.

To bridge the gap between offline classification accuracy and reliable production execution, this work designs and validates an end-to-end Machine Learning Operations (MLOps) platform for Telco Customer Churn prediction. Rather than promoting any newly trained classifier by default, the platform implements a strict governance principle: a candidate model must first pass deterministic schema and output checks, then satisfy a recall-oriented quality gate, and finally outperform the incumbent Production model on a permanently fixed held-out evaluation split before container and Kubernetes deployment is permitted.

The primary engineering and experimental contributions of this study are as follows:
- Comparative evaluation of Dummy, Logistic Regression, Decision Tree, Random Forest, and Extreme Gradient Boosting (XGBoost) classifiers with imbalance compensation and decision-threshold calibration.
- Idempotent data ingestion governed by SHA-256 batch hashing, overlapping-identifier checks, versioned dataset archival, and Pandera schema enforcement.
- Fail-fast ten-stage orchestration integrated with MLflow experiment tracking, recall quality gating, and automated lifecycle promotion decisions.
- Resilience validation through fifteen fault-injection trials, multi-dimensional drift monitoring, and parity-verified FastAPI serving across Docker and Kubernetes.


## Dataset and Ingestion Governance


### I. Telco Customer Churn Dataset and Fixed Test Split

The initial experimental corpus comprises 7,043 customer records across 21 columns, capturing demographic attributes (gender and senior citizen status), account tenure, billing indicators (contract length, paperless billing, payment method, monthly charges, and total charges), and subscribed telecom services (phone, multiple lines, internet type, online security, backup, device protection, technical support, and streaming services). The binary target variable exhibits moderate class imbalance, with 26.54% positive churn cases overall and 374 positive churners among 1,409 records in the stratified 20% test split (1,035 retained accounts).

To eliminate optimistic evaluation drift as additional subscriber batches arrive, Stage 03 permanently locks the 1,409 test customer identifiers in a dedicated reference file. Every subsequent dataset expansion routes all newly ingested records exclusively into the training partition while evaluating every candidate classifier on the exact same held-out test population.


### II. Idempotent Batch Ingestion and Schema Validation

Before training begins, Stage 01 inspects incoming batch files and computes a cryptographic SHA-256 digest. This digest is compared against an append-only lineage ledger recording prior versions (v01 at 7,044 rows, v02 at 7,144 rows, v03 at 7,244 rows, and v04 at 7,344 rows across three 100-row batches). If the digest already exists in lineage, the loader outputs an already-ingested status and halts downstream retraining cleanly, guaranteeing idempotency across scheduled workflow triggers. When a genuinely new hash arrives, identity checks verify that customer identifiers are unique within the batch and disjoint from existing records before archiving the prior snapshot.

Stage 02 enforces strict schema contracts using Pandera across all 21 columns, rejecting invalid categorical levels, unexpected missing columns, non-numeric billing values, and negative values for tenure or monthly charges. Stage 04 subsequently verifies transformed NumPy matrices (5,635 training rows by 30 encoded features prior to the final batch, and 5,935 training rows after v04 ingestion) to ensure zero missing or non-finite entries reach the estimator.


## Methodology and Predictive Modeling


### I. Preprocessing Pipeline and Imbalance Compensation

Raw ingestion converts blank TotalCharges strings into numeric values and encodes the target into binary integers. To prevent data leakage, all transformers are fit solely on the training split inside a unified scikit-learn ColumnTransformer pipeline. Four numerical features undergo constant-zero imputation and standard z-score normalization, while fifteen categorical attributes undergo one-hot encoding with first-category dropping and unknown-category tolerance, yielding 30 model-ready features.

Because failing to identify a departing customer incurs substantially higher business cost than sending a retention offer to a loyal subscriber, recall on the positive churn class is prioritized alongside F1-score and ROC-AUC. Recall R is defined as

$$
R = \frac{TP}{TP + FN} \qquad (1)
$$

where TP denotes true positive churners and FN denotes false negatives. Class imbalance is addressed during training via inverse class-frequency weighting in Logistic Regression and Random Forest, synthetic minority oversampling experiments, and scale_pos_weight set to 2.77 in XGBoost.


### II. Comparative Classifier Evaluation and Threshold Calibration

Model exploration evaluated five candidate architectures on the 1,409 held-out test records, progressing from a majority-class dummy baseline to grid-searched and randomized-searched ensemble models. Table 1 summarizes the comparative evaluation across metrics. Moving from an unweighted Logistic Regression baseline (0.5187 recall) to class-weighted and boosted models raised recall above 0.77 while boosting ROC-AUC above 0.84. Tuned XGBoost (100 trees, depth 6, learning rate 0.05, subsample 0.70, column sample 0.80) achieved the highest default-threshold F1-score (0.6335) and PR-AUC (0.6610). Lowering the classification probability threshold from 0.50 to 0.45 increased churn recall to 0.8048 (detecting 301 of 374 churners) with 0.6238 F1-score. Feature importance analysis identified two-year contracts (0.2301), fiber-optic internet (0.1580), absence of internet service (0.0882), one-year contracts (0.0869), and tenure exceeding four years (0.0810) as the dominant predictors.


**Table 1.** Comparative model evaluation on the held-out telco churn test set (n = 1,409).

| Classifier / Configuration | Acc. | Prec. | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Dummy Majority Baseline | 0.7346 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| Logistic Regression (Unweighted) | 0.7991 | 0.6532 | 0.5187 | 0.5782 | — |
| Logistic Regression (L1, C = 0.1) | — | 0.5086 | 0.7914 | 0.6192 | 0.8419 |
| Decision Tree (Balanced Baseline) | 0.7395 | 0.5059 | **0.8075** | 0.6220 | 0.8333 |
| Random Forest (Tuned, 200 Trees) | — | 0.5301 | 0.7781 | 0.6306 | 0.8424 |
| XGBoost (Tuned, Threshold = 0.50) | — | 0.5318 | 0.7834 | **0.6335** | **0.8444** |
| XGBoost (Tuned, Threshold = 0.45) | 0.7424 | 0.5093 | 0.8048 | 0.6238 | 0.8444 |


## End-to-End MLOps Architecture and Lifecycle Governance

While offline tuning demonstrated strong XGBoost capability, production pipeline orchestration standardizes around a deterministic Random Forest classifier configured via YAML parameters and tightly coupled with MLflow tracking and model registration. Figure 1 illustrates the complete dual-loop architecture connecting continuous drift monitoring with the ten-stage training and promotion pipeline.


**Figure 1.** End-to-end churn prediction MLOps architecture. Panel A illustrates the hourly GitHub Actions monitoring loop and SHA-256 lineage gate; panel B traces the ten fail-fast stages from ingestion through promotion-gated Docker and Kubernetes rollout.


### I. MLflow Model Registry and Recall-Gated Promotion

Stage 05 trains the configured Random Forest classifier, logs hyperparameters, evaluation metrics, and the serialized preprocessor artifact to MLflow, and registers a new version under Telco_Churn_Production_Model. Stage 07 enforces a mandatory quality gate requiring recall of at least 0.70; candidates falling short halt execution prior to registry staging. Stage 08 then retrieves the current Production version from MLflow and promotes the Staging candidate only if its recall strictly exceeds that of Production.

During full integration testing after ingesting batch v04 (bringing total rows to 7,344), candidate version 17 (150 trees, maximum depth 12) achieved higher accuracy (0.7672), precision (0.5466), and ROC-AUC (0.8452) than prior runs, as detailed in Table 2. However, its recall of 0.7219—despite passing the 0.70 minimum gate—remained below Production version 1's recall of 0.7807. Consequently, the lifecycle manager archived version 17, retained version 1 in Production, and skipped deployment, demonstrating automated protection against recall regression. Figure 2 displays the confusion matrix (778 true negatives, 257 false positives, 87 false negatives, 287 true positives) and ROC curve (AUC = 0.84) for the local Random Forest baseline candidate (version 8 in the local registry).


**Table 2.** Lifecycle governance comparison across registered Random Forest versions.

| Model Version / Stage | Acc. | Prec. | Recall | F1 | Outcome |
| --- | --- | --- | --- | --- | --- |
| Production v1 (100 Trees, Depth 10) | 0.7580 | 0.5299 | **0.7807** | **0.6314** | Retained |
| Baseline Candidate v8 (Local DB) | 0.7559 | 0.5276 | 0.7674 | 0.6253 | Promoted Locally |
| Integrated Candidate v17 (7,344 Rows) | **0.7672** | **0.5466** | 0.7219 | 0.6221 | Archived (Gate Pass) |


**Figure 2.** Diagnostic evaluation plots for the local Random Forest pipeline baseline. Panel (a) shows the test confusion matrix across 1,409 held-out customers; panel (b) plots the corresponding ROC curve (AUC = 0.84).


## Reliability, Drift Monitoring, and Containerized Serving


### I. Reproducibility and Fault-Injection Recovery

Pipeline determinism was verified by executing end-to-end preprocessing, training, and evaluation twice under fixed random seed 42; both runs produced identical metrics down to four decimal places (0.7580 accuracy, 0.7807 recall, 0.6314 F1-score). To validate operational safety, a fault-injection suite injected five failure modes across three repeated trials (15 total executions). Missing input data, corrupted string datatypes in MonthlyCharges, missing schema columns (gender) with invalid contract categories, and NaNs injected into processed matrices all terminated cleanly with exit code 1 before training. Conversely, deleting the local serialized model triggered automated detection during evaluation and restored the verified Production artifact from the MLflow registry with exit code 0.


### II. Statistical Drift Analysis and Scheduled Automation

Post-deployment monitoring tracks numerical feature shifts using the two-sample Kolmogorov-Smirnov (KS) statistic D, defined between baseline empirical distribution F1 and evolved distribution F2 as

$$
D = \sup_{x} |F_1(x) - F_2(x)| \qquad (2)
$$

Categorical distributions are evaluated via Jensen-Shannon (JS) divergence against a unified 0.10 alert threshold. As shown in Figure 3, simulated production traffic revealed significant numerical drift in MonthlyCharges (KS = 0.1547) and TotalCharges (KS = 0.1036), while tenure (KS = 0.0972) and categorical features (JS between 0.0372 and 0.0396) remained stable. Prediction drift was likewise flagged across predicted classes (KS = 0.1015, p = 9.76e-7) and churn probabilities (KS = 0.2300, p = 4.62e-33). A GitHub Actions workflow scheduled hourly on a self-hosted Windows runner executes the drift trigger and safely skips redundant retraining when the incoming batch hash matches lineage.


**Figure 3.** Observed feature and prediction drift divergence scores. Dark bars exceed the 0.10 monitoring threshold (MonthlyCharges, TotalCharges, predicted class, and predicted probability), triggering conditional retraining.


### III. FastAPI Microservice, Docker Containerization, and Kubernetes Rollout

Serving is implemented via FastAPI with strict Pydantic schemas rejecting negative charges, invalid categories, null values, and malformed JSON using standardized HTTP 422 payloads, while logging every request to JSON Lines audit storage. Table 3 summarizes verification across local testing, Docker, and Kubernetes. Twenty-three functional robustness tests and six performance tests passed locally, achieving 19.38 milliseconds average latency and 51.68 requests per second throughput. Building and running the python:3.11-slim Docker image on port 8000 yielded exact floating-point prediction parity (prediction 0, probability 0.4173425977) with local execution, and Kubernetes manifests scaled to three ready pods.


**Table 3.** Serving test, latency, and container consistency results.

| Verification Layer | Measured Result |
| --- | --- |
| Functional & Robustness (pytest) | 23/23 tests passed (HTTP 200/422/503) |
| Local Mean / Median Latency | 19.38 ms / 18.79 ms (20 runs) |
| 20-Record Batch Latency & Reliability | 21.34 ms; 50/50 requests (100%) |
| Local Throughput | 51.68 requests/second |
| Docker Parity (churn-prediction-api:1.0) | Exact match (p = 0.4173425977) |
| Kubernetes Deployment (churn-api) | Rollout ready (3/3 pods verified) |


## Conclusion and Future Work

This paper presented a complete, reproducible, and governance-driven MLOps system for telecommunications churn prediction. Experimental evaluation established that imbalance-aware classifiers and calibrated decision thresholds achieve over 0.80 recall on departing customers, while automated pipeline gates, SHA-256 lineage tracking, MLflow lifecycle comparison, fault-injection testing, and containerized FastAPI serving prevent degraded models or corrupted batches from disrupting live operations. Future work will extend the monitoring loop to connect live streaming customer billing feeds, automate threshold recalibration under severe concept drift, and deploy persistent remote MLflow tracking servers across distributed cloud runner environments.


## References

[1] D. Sculley et al., “Hidden technical debt in machine learning systems,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 28, pp. 2503–2511, 2015.

[2] L. Breiman, “Random forests,” Mach. Learn., vol. 45, no. 1, pp. 5–32, Oct. 2001.

[3] T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining, 2016, pp. 785–794.

[4] F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” J. Mach. Learn. Res., vol. 12, pp. 2825–2830, Nov. 2011.

[5] M. Zaharia et al., “Accelerating the machine learning lifecycle with MLflow,” IEEE Data Eng. Bull., vol. 41, no. 4, pp. 39–45, Dec. 2018.

[6] N. Bansal et al., “Pandera: Statistical data validation of pandas dataframes,” in Proc. Python in Science Conf. (SciPy), 2020, pp. 116–124.

[7] F. J. Massey, “The Kolmogorov-Smirnov test for goodness of fit,” J. Amer. Statist. Assoc., vol. 46, no. 253, pp. 68–78, 1951.

[8] S. Ramirez, FastAPI Framework Documentation. [Online]. Available: fastapi.tiangolo.com.


## About the Authors

**Author One** is a researcher and machine learning systems engineer focusing on reproducible predictive analytics, automated continuous integration pipelines, and scalable container orchestration for enterprise customer intelligence platforms.

**Author Two** specializes in applied data engineering, statistical drift detection, and schema validation frameworks for production tabular machine learning services.

**Author Three** researches cloud-native microservice architecture, automated model governance, and fault-tolerant deployment pipelines using Docker and Kubernetes.
