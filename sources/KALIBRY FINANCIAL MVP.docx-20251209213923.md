# KALIBRY FINANCIAL MVP.docx

# **KALIBRY FINANCIAL MVP**
### **_Trajectory Intelligence for Short-Term Financial Forecasting_**
Short version: [Executive Summary - KALIBRY FINANCIAL MVP](https://docs.google.com/document/d/1BK3CIRsmZjkPTca1GnOMXkRkCa0kACyPRtdXmc0Fb8Q/edit?tab=t.0#heading=h.jco2dul9v4w4)
# **0\. Executive Summary**
# **0.1 Purpose of the MVP**
The purpose of this MVP is to **validate Kalibry’s Trajectory Intelligence Engine** in one of the most demanding and data-rich domains: **financial markets**.
The MVP demonstrates that Kalibry can:
*   represent any entity (here: financial assets) as an **evolving trajectory**,
*   identify **families of similar past trajectories**,
*   infer **probabilistic future evolution** from historical analogs,
*   and deliver **transparent, explainable, well-calibrated forecasts**.
This MVP is the first proof-of-concept showing that Kalibry is not a forecasting algorithm, but a **general reasoning framework** capable of understanding evolution, behaviour, and change through pattern trajectories.
## **0.2 Vision: Kalibry as a universal engine for evolving entities**
Kalibry is designed to become the **universal reasoning layer** for any system that evolves over time.
Its core vision:
**Everything that changes can be represented as a trajectory:**
**financial assets, customers, machines, biological systems, markets, risk profiles, revenue streams, churn patterns, patient evolution, and more.**
Kalibry provides:
*   a shared **temporal embedding framework**
*   a unified **trajectory representation layer**
*   a cross-domain **probabilistic reasoning engine**
*   inherent **explainability** aligned with regulatory standards (AI Act)
By succeeding in finance—an environment with extremely high noise, volatility, and complexity—Kalibry demonstrates it can succeed **anywhere**.
## **0.3 Why Starting with Finance**
We begin with financial assets because:
### **1\. Finance is the most data-intensive evolutionary environment**
Daily (or intraday) time-series over decades allow clear testing of:
*   trajectory construction
*   similarity matching
*   probabilistic reasoning
*   calibration and stability
### **2\. The domain is extremely competitive and noisy**
If Kalibry works here, it works everywhere.
### **3\. Finance provides objective benchmarks**
We can compare Kalibry against:
*   Random Walk
*   Momentum
*   AR/ARMA/ARIMA
*   GARCH
*   LSTM
*   Transformers
No other domain offers such clean, measurable ground truth.
### **4\. High trust + explainability are mandatory in finance**
This makes it the perfect environment to showcase Kalibry’s built-in explainability capabilities (Evidence Graphs), which give:
*   historical supporting patterns
*   transparent reasoning
*   compliance-friendly predictions
### **5\. The financial market is an ideal early adopter**
Banks, funds, and fintechs:
*   invest heavily in predictive analytics
*   seek new forms of explainable AI
*   are under pressure to justify decisions
*   value calibrated, stable forecasts
Finance is the best strategic entry point.
## **0.4 Core Hypothesis**
The MVP evaluates the following hypothesis:
**Financial assets behave as trajectories in a multidimensional state space.**
**When two trajectories are similar, their short-term future distributions tend to be similar as well.**
**Therefore, we can infer future behavior by matching today's trajectory to historical families of similar patterns.**
This hypothesis stands in contrast with:
*   price-level forecasting
*   black-box neural networks
*   single-signal models (momentum, moving averages)
Kalibry proposes a **new form of intelligence**:
*   not predictive → **analogical**
*   not black-box → **explainable**
*   not numeric → **pattern-based**
*   not single-output → **probabilistic evidence aggregation**
The MVP tests whether this hypothesis holds consistently across:
*   different assets
*   different time periods
*   different volatility regimes
*   different market structures
## **0.5 Key Outcomes Expected**
The MVP aims to deliver:
### **1\. Performance Superiority**
Kalibry is expected to outperform:
*   Random Walk (50%)
*   Momentum (~52–55%)
*   ARMA/ARIMA (~51–54%)
*   GARCH-based models (~52–54%)
*   LSTMs and Transformers (~53–57%, unstable)
Target accuracy:
*   **58–62%** for 1-day horizon
*   **56–60%** for 5-day horizon
### **2\. Superior Probability Calibration**
Kalibry’s empirical distribution approach is expected to produce:
*   Brier Score improvement of **+10–20%** vs ML baselines
*   Stability across regimes
### **3\. Cross-Regime Robustness**
Consistent performance across:
*   bull markets
*   bear markets
*   high-volatility periods
*   shocks and structural breaks
### **4\. Full Explainability**
Every forecast supported by:
*   trajectory similarity evidence
*   historical analogs
*   dimension-level explanations
*   risk-aware probability distributions
### **5\. Operational Efficiency**
Prediction latency target:
*   **<100 ms** per asset
### **6\. Demonstration that the Kalibry engine is general-purpose**
Success in finance will validate:
*   the reusability of trajectory embeddings
*   the generality of the vector reasoning engine
*   applicability to dozens of other verticals
## **0.6 Strategic Value (for investors + industry)**
This MVP proves much more than a finance module—it validates **Kalibry’s foundational architecture**.
### **For Investors**
*   Positions Kalibry at the intersection of
*   **AI × Time-Series × Explainability × Finance**
*   Demonstrates high defensibility (unique reasoning architecture)
*   Establishes clear superiority over traditional ML baselines
*   Opens partnership pathways with fintechs, banks, and funds
*   Provides a strong early revenue channel (B2B analytics tools)
### **For Industry**
*   Delivers explainable intelligence aligned with regulatory demands
*   Enables cross-asset, cross-regime forecasting
*   Improves risk management through calibrated probabilities
*   Provides a transparent alternative to black-box neural networks
*   Acts as a plug-and-play reasoning layer for financial systems
### **For Kalibry’s Long-Term Strategy**
*   Finance becomes the **first vertical adapter** of the Kalibry Engine
*   Validates the universal trajectory framework
*   Paves the way for expansion into:
    *   credit risk
    *   churn forecasting
    *   supply chain evolution
    *   eCommerce demand trajectory modeling
    *   industrial predictive maintenance
    *   patient evolution in healthcare
    *   macroeconomic scenario trajectories
# **1\. Introduction**
## **1.1 What is Trajectory Intelligence**
Trajectory Intelligence is a new computational paradigm for understanding how entities evolve over time.
Unlike conventional forecasting—which predicts a future value—Trajectory Intelligence:
*   models **entities as dynamic paths in a multi-dimensional state space**,
*   identifies **historical analogs** with similar evolution,
*   extracts **probabilistic evidence** from how those analogs continued,
*   and constructs fully transparent, explainable inferences.
Every evolving system—financial markets, customers, machines, patients, revenue streams—can be represented as:
![](https://t90182117410.p.clickup-attachments.com/t90182117410/5124ab33-6c40-4ac9-a149-1f4b05868deb/156a067e-2105-4748-a4a8-a3e263aaae43.png)
where each xtx\_txt​ is a state embedding encoding the system’s behaviour at time ttt.
Kalibry is the first platform that transforms this theoretical concept into a **general-purpose, cross-domain reasoning engine**.
The paradigm shift is profound:

| **Traditional Forecasting**<br> | **Trajectory Intelligence**<br> |
| ---| --- |
| Predicts a number | Matches behaviour patterns |
| Single output | Probability distribution |
| Black-box | Evidence-driven |
| Input → output | Past families → inferred evolution |
| Often unstable | Naturally calibrated |

Trajectory Intelligence brings human-like analogical reasoning into machine intelligence—while retaining mathematical rigour and computational scalability.
## **1.2 Why Financial Assets Are an Ideal First Vertical**
Financial markets are the **most natural domain to validate trajectory-based reasoning** because they are:
### **1\. Dense with high-frequency, multi-factor information**
Assets produce:
*   OHLC prices
*   volumes
*   volatility signals
*   cross-asset correlations
*   regime shifts
This creates rich, multi-dimensional trajectories.
### **2\. Highly evolutionary systems**
Assets are never static; they continuously evolve through:
*   momentum phases
*   volatility regimes
*   liquidity cycles
*   trend reversals
This perfectly matches Kalibry’s philosophy:
**understand evolution, not snapshots**.
### **3\. Quantitatively testable**
Unlike many industries:
*   ground truth is objective
*   performance is measurable
*   baselines are well-defined
*   results cannot be “fudged”
This forces scientific standards and validates Kalibry’s robustness.
### **4\. Extremely challenging**
Financial time-series:
*   are noisy
*   exhibit structural breaks
*   contain multiple regimes
*   have low signal-to-noise ratio
If Kalibry succeeds here, it becomes _instantly credible_ in any other domain.
### **5\. Highly valuable commercially**
Financial institutions need:
*   calibrated probabilities
*   explainable AI
*   risk-aware forecasts
*   scenario reasoning
*   pattern-matching analytics
Finance delivers the fastest path to commercial validation and revenue.
## **1.3 Limitations of Traditional Time-Series Forecasting**
Classical forecasting methods (ARIMA, GARCH, Holt-Winters, etc.) face structural limitations:
### **1\. They assume stationarity**
Financial assets are _not_ stationary:
*   trends change
*   volatilities shift
*   correlations break
*   regimes flip abruptly
Linear models cannot adapt to these quick structural transitions.
### **2\. They capture patterns only locally**
Most classical models use only:
*   short lags
*   single-variable dynamics
*   linear relationships
They are blind to:
*   multi-dimensional interactions
*   cross-asset influences
*   long-range dependencies
*   nonlinear regime transitions
### **3\. They produce point estimates, not calibrated distributions**
This is dangerous for:
*   risk management
*   portfolio construction
*   scenario analysis
### **4\. They do not generalize across assets**
Each asset needs a separate model.
There is no shared learning or pattern reuse.
## **1.4 Kalibry vs Black-Box AI Models (LSTM, Transformers, etc.)**
Modern AI models (LSTM, GRU, Transformers) improve on classical methods but introduce new problems:
### **1\. Poor performance on noisy financial data**
Neural nets overfit:
*   micro-structure noise
*   random fluctuations
*   non-informative patterns
And fail to generalize across regimes.
### **2\. No interpretability**
Critical for finance:
*   risk officers
*   investors
*   compliance departments
*   regulators (AI Act)
Black-box forecasts with no explanation are unacceptable.
### **3\. Difficulty in capturing analogical reasoning**
Neural networks predict “from scratch”
—but human analysts predict by **pattern analogy**:
“This looks like 2017 again. Here’s what happened then.”
Kalibry does this natively.
### **4\. Poor probability calibration**
Even when neural nets predict probabilities, they are:
*   overconfident
*   unstable
*   poorly calibrated
Kalibry’s empirical distribution approach produces **naturally calibrated forecasts**.
### **5\. High computational cost**
Transformers require:
*   massive datasets
*   heavy training
*   expensive inference
Kalibry’s similarity-based reasoning is:
*   cheaper
*   faster
*   scalable
And does not require retraining when new assets or regimes appear.
## **1.5 Explainability as a Differentiator (AI Act–Aligned)**
Explainability is not optional in finance:
it is required legally, operationally, and commercially.
Kalibry provides **built-in explainability**, not a post-hoc add-on.
### **1\. Evidence Graphs**
For each prediction, Kalibry shows:
*   the current 60-day trajectory
*   the 3–5 closest historical trajectories
*   their known outcomes
*   the similarity weights
*   the combined probabilistic evolution
### **2\. Feature-level similarity breakdown**
Stakeholders can see:
*   volatility pattern similarity
*   momentum structure similarity
*   regime match
*   drawdown stage alignment
### **3\. Narrative, human-readable explanation**
Example:
“NVDA’s current trajectory matches three past periods (NVDA 2017, AAPL 2020, NASDAQ 2021).
In 72% of those analog periods, the next 5 days produced a positive return.
Probability Up (H5) = 0.72.”
### **4\. Alignment with EU AI Act**
Kalibry satisfies:
*   **traceability**
*   **transparency**
*   **evidence-based reasoning**
*   **non-black-box logic**
*   **accountability**
Making it a **next-generation compliant AI system**, even in high-risk environments like finance.
# **2\. MVP Scope & Objectives**
## **2.1 High-Level Goals**
The MVP aims to achieve three strategic goals:
### **Goal 1 — Validate Kalibry’s Trajectory Intelligence Engine in Finance**
Demonstrate that financial assets can be modeled as **evolving trajectories** in a multi-dimensional space, and that:
*   similar trajectories → similar futures
*   empirical evidence → probabilistic predictions
*   pattern-based reasoning → higher stability
This validates Kalibry’s entire architecture.
### **Goal 2 — Prove Superiority Over Classical & Modern Models**
Show that Kalibry:
*   outperforms naive strategies (Random Walk, Always-Up, Momentum)
*   surpasses classical econometric models (ARIMA, GARCH, AR-GARCH)
*   achieves better stability and calibration than ML models (LSTM, Transformers)
If Kalibry wins here, it wins in any domain.
### **Goal 3 — Demonstrate Explainability & Compliance-Readiness**
Provide fully transparent, evidence-driven forecasts:
*   matched historical trajectories
*   interpretable outcomes
*   regulatory-friendly explanations (AI Act–aligned)
This builds trust with institutional users from day one.
## **2.2 What the MVP** **_Does_**
The MVP provides the following capabilities:
### **1\. Builds 45-dimensional State Embeddings**
For each asset and each day, Kalibry:
*   computes a multi-scale state vector (returns, volatility, momentum, regime, liquidity, cross-asset factors, etc.)
*   normalizes them in a rolling, leakage-free manner
### **2\. Constructs 60-day Trajectories**
Transforms sequential states into:
Tt∈R60×45\\mathcal{T}\_t \\in \\mathbb{R}^{60 \\times 45}Tt​∈R60×45
These are the core objects used for reasoning.
### **3\. Performs Trajectory Similarity Search (HPVD)**
Retrieves the **K-most similar historical trajectories**, based on:
*   cosine similarity
*   temporal pooling
*   regime context
*   learned metrics (optional)
### **4\. Produces Empirical, Calibrated Probabilities**
From matched analog trajectories, Kalibry computes:
*   probability of direction (Up/Down/Flat) for **1 day**
*   probability of direction for **5 days**
*   distribution of expected outcomes
*   uncertainty (entropy, variance)
Predictions are **not black-box outputs**, but **evidence-based aggregations**.
### **5\. Provides Full Explainability**
The MVP includes:
*   trajectory visualizations
*   the historical analogs
*   their actual outcomes
*   explanation summaries
*   feature-level similarity breakdown
### **6\. Offers a Demo Dashboard**
A clean interface to:
*   select an asset
*   view its current 60-day trajectory
*   see matched patterns
*   view probability forecasts
*   read the explanation
Perfect for demonstrations to:
*   investors
*   financial analysts
*   internal stakeholders
## **2.3 What the MVP** **_Does Not_** **Do**
To avoid scope creep and ensure credibility:
### **1\. Does** **_not_** **provide price targets**
The system forecasts **direction & probability**, not specific price levels.
### **2\. Does** **_not_** **attempt long-term forecasting**
All horizons are **short-term** (1–5 days).
### **3\. Does** **_not_** **execute trades**
This is not an automated trading system.
### **4\. Does** **_not_** **give investment advice**
It produces research analytics, not actionable recommendations.
### **5\. Does** **_not_** **incorporate intraday microstructure**
The MVP uses **daily** data.
Intraday comes in later versions.
### **6\. Does** **_not_** **optimize portfolios or allocate capital**
Portfolio reasoning is a **future module**.
### **7\. Does** **_not_** **replace financial analysts**
Instead, it augments them with:
*   transparent analogies
*   data-driven evidence
*   calibrated probabilities
## **2.4 Success Criteria (Technical + Business)**
### **A. Technical Success Criteria**
#### **1\. Directional Accuracy Targets**
*   **H1 (1-day)**: 58–62%
*   **H5 (5-day)**: 56–60%
Must outperform:
*   Random Walk (50%)
*   Momentum (52–55%)
*   AR/ARMA/ARIMA (51–54%)
*   GARCH (52–54%)
*   LSTM/Transformer (approx. 53–57%, unstable)
#### **2\. Calibration Targets**
*   Brier Score improvement of **+10–20%** vs ML models
*   Reliability curve: predicted vs actual probabilities closely aligned
#### **3\. Robustness Across Regimes**
*   Performance drop across bull/bear/high-volatility < **5–7 pp**
#### **4\. Explainability Quality**
*   Clear evidence graphs
*   High similarity fidelity
*   Interpretable dimension-level comparisons
*   AI Act–compliant logic
#### **5\. Computational Efficiency**
*   Prediction latency: **<100 ms per asset**
*   Scalable to 20–50 assets in MVP
*   Horizontal scalability demonstrated
### **B. Business Success Criteria**
#### **1\. Credible Demonstration to Financial Institutions**
MVP must be convincing for:
*   asset managers
*   hedge funds
*   investment banks
*   fintechs
#### **2\. Clear Competitive Advantage**
Stakeholders must see:
*   better performance
*   better calibration
*   unmatched explainability
*   a novel AI reasoning paradigm
#### **3\. Establishment of Use Cases**
MVP must clearly enable:
*   risk analysis
*   pattern recognition
*   calibrated scenario exploration
*   explainable forecasting
*   regime detection
#### **4\. Market Validation**
Feedback from **at least 3–5 financial experts** confirming:
*   usefulness
*   novelty
*   competitive differentiation
*   potential commercial value
## **2.5 Key Deliverables**
### **Technical Deliverables**
1. Fully operational Trajectory Intelligence Engine
2. State embedding generator (45 features)
3. Trajectory builder (60×45 sequences)
4. HPVD similarity search implementation
5. Probabilistic aggregation module
6. Full evaluation framework (backtesting, baselines)
7. Comparison vs ARIMA, GARCH, ML (LSTM/Transformer)
### **Documentation Deliverables**
1. Full MVP methodology report (this document)
2. Feature specification documentation
3. Architecture diagrams
4. Backtesting report (train/val/test + regime analysis)
5. Calibration report
6. Case study pack (3–5 real historical examples)
### **Demo Deliverables**
1. Interactive dashboard:
    *   Trajectory view
    *   Similarity matches
    *   Probability outputs
    *   Explanations
2. Investor pitch deck (10–15 slides)
3. Video walkthrough (optional)
### **Strategic Deliverables**
1. Proof that Kalibry’s architecture works in finance
2. Foundation for vertical adapters in other industries
3. Market credibility for entering enterprise AI pipelines
# **3\. Data & Asset Universe**
The success of Kalibry’s financial MVP depends heavily on the diversity, depth, and quality of the underlying data. This section defines the asset universe used for testing, the rationale behind its construction, the preprocessing pipeline, and the rules applied to eliminate data leakage and ensure scientific validity.
## **3.1 Asset Classes Included (Equities, Indices, FX, Crypto)**
To ensure robustness across different market structures, the MVP uses a **multi-asset universe** composed of:
### **1\. Equities (Large-Cap US + EU)**
Representative examples:
*   AAPL
*   MSFT
*   AMZN
*   NVDA
*   META
*   TSLA
*   ASML
*   SAP
These assets exhibit diverse volatility patterns, liquidity structures, and sector-specific dynamics.
### **2\. Equity Indices**
*   S&P 500 (SPX)
*   NASDAQ 100 (NDX)
*   DAX
*   FTSE MIB
*   EURO STOXX 50
Indices provide smoother, more aggregated dynamics and serve as stabilizing anchors.
### **3\. FX Pairs**
*   EUR/USD
*   USD/JPY
*   GBP/USD
*   USD/CHF
FX markets are highly liquid, regime-driven, and ideal for testing cross-market generalization.
### **4\. Crypto Assets**
*   BTC
*   ETH
*   SOL
Crypto introduces extreme volatility and structural breaks, testing Kalibry’s robustness under stress.
### **Asset Universe Size**
MVP target size: **20–50 assets**
This ensures:
*   cross-regime diversity
*   multiple volatility levels
*   enough trajectories to train similarity patterns
*   realistic institutional relevance
A multi-asset universe is essential to validate a **cross-domain trajectory engine**.
## **3.2 Asset Selection Rationale**
Assets are selected based on three strategic criteria:
### **1\. Liquidity & Data Integrity**
Only assets with **continuous, reliable daily data** are used.
This avoids:
*   missing candles
*   illiquid gaps
*   artificial volatility
*   microstructure bias
### **2\. Regime Variability**
Assets were chosen to ensure presence of:
*   bull markets
*   bear markets
*   correction phases
*   consolidations
*   volatility spikes
*   macroeconomic shocks
*   trend reversals
The goal is to challenge Kalibry’s trajectory matching under heterogeneous conditions.
### **3\. Structural Diversity**
The universe covers:
*   earnings-driven equities
*   macro-driven FX
*   sentiment-driven crypto
*   index-level trend dynamics
This validates Kalibry’s ability to:
*   learn cross-market behavioural analogies
*   recognize universal trajectory patterns
*   adapt to different time-series “personalities”
## **3.3 Data Sources & Frequency (Daily OHLCV)**
### **Frequency**
*   **Daily** OHLCV data
Daily frequency offers:
*   high signal-to-noise contrast
*   long historical depth
*   stable regime representation
*   realistic operational value
Intraday data may be incorporated in v2, but it is not required for MVP validation.
### **Data Structure**
For each asset and each day, the following fields are collected:
*   **Open**
*   **High**
*   **Low**
*   **Close**
*   **Volume**
*   **Adjusted Close** (where applicable)
Additional fields are computed via feature engineering (Section 5).
### **Data Providers**
Possible sources:
*   Yahoo Finance
*   [Polygon.io](http://Polygon.io)
*   Alpha Vantage
*   Tiingo
*   Binance (for crypto)
For the MVP, any reliable source with consistent historical OHLCV is acceptable.
## **3.4 Historical Depth Requirements (8–10 years)**
To construct meaningful trajectory families, the dataset must span:
*   **8–10 years of daily data**, ideally
*   minimum **3–5 years** for younger assets (e.g., crypto)
This ensures:
*   enough historical analogs for similarity search
*   full coverage of multiple market regimes
*   stable distributions for calibration
*   sufficient examples for statistical evaluation
Typical sample size:
*   8 years ≈ ~2,000 daily observations per asset
*   20 assets ≈ ~40,000 data points
This is more than sufficient for a trajectory-based MVP.
## **3.5 Data Cleaning & Preprocessing**
Data preprocessing consists of five critical stages:
### **1\. Missing Data Handling**
*   forward-fill only within extremely short gaps
*   remove assets or periods with extended missing intervals
*   crypto exchanges checked for split data across venues
No interpolation of prices is allowed to avoid artificial signals.
### **2\. Outlier & Anomaly Detection**
We detect and log:
*   flash spikes
*   fat-finger prints
*   duplicated candles
These are either removed or adjusted depending on severity.
### **3\. Corporate Actions Adjustments**
For equities:
*   splits
*   dividends
*   mergers
Close-adjusted data is used to normalize these events.
### **4\. Synchronization Across Markets**
All assets aligned on:
*   trading days
*   timestamps
*   holiday calendars
Non-trading days are removed to maintain consistent temporal windows.
### **5\. Standardization & Scaling**
State features are standardized **per date**, using:
*   rolling mean
*   rolling standard deviation
This guarantees:
*   no future information leakage
*   cross-sectional comparability
*   stable feature distributions
This step is essential for trajectory similarity.
## **3.6 Rules to Avoid Leakage**
Avoiding leakage is **a strict scientific requirement**.
Kalibry implements rigorous and auditable constraints:
### **1\. Feature Calculations Use Only Past Data**
Each feature uses:
![](https://t90182117410.p.clickup-attachments.com/t90182117410/09f7aad9-2aa3-4efc-aa7c-ad66181ff196/cfd08aef-6014-42ff-80d0-fefe51e20b13.png)
No window ever includes future timestamps.
### **2\. Scaling & Normalization Are Rolling**
For each day:
![](https://t90182117410.p.clickup-attachments.com/t90182117410/5d377295-a108-46d2-a1eb-4d30d73f55f1/f8cbdbdc-8e27-4aa6-be2f-ad9fd48e634a.png)
Never computed using future values.
### **3\. Trajectory Windows Do Not Overlap into Future**
When constructing trajectories of length 60:
*   each trajectory ends strictly at time ttt
*   training trajectories never include test future points
### **4\. Historical Analogs Exclude Future Periods**
When matching trajectories for predictions at time ttt:
*   similar trajectories must be from **dates < t**
*   analog windows that overlap with the forecasting window are disallowed
### **5\. Walk-Forward Protocol Enforced**
All evaluation uses:
*   rolling training windows
*   forward-only prediction windows
This is the gold standard in financial backtesting.
### **6\. No Cross-Contamination Between Train/Val/Test**
We isolate:
*   2013–2019 (Train)
*   2020–2021 (Validation)
*   2022–2024 (Test)
Assets do not leak information across temporal boundaries.
# **4\. Prediction Problem Definition**
This section defines precisely what the Kalibry Financial MVP predicts, how the prediction task is structured, which target variables are used, the temporal logic behind observation and forecasting windows, and why short-term directional prediction is both meaningful and strategically advantageous in finance.
## **4.1 Forecasting Tasks (H1 = 1 day, H5 = 5 days)**
The MVP evaluates Kalibry’s ability to provide **short-term directional forecasts** using two prediction horizons:
### **H1 — 1-Day Ahead Direction**
![](https://t90182117410.p.clickup-attachments.com/t90182117410/00c3d83c-7027-4d46-9429-55c573044893/7f1c80c1-a4e4-44e9-a160-5e787b2ded99.png)
A binary classification task:
*   **Up** (positive return)
*   **Down** (negative return)
### **H5 — 5-Day Ahead Direction**
![](https://t90182117410.p.clickup-attachments.com/t90182117410/69e3cc38-a43d-4f4c-a0b7-a83e1f0476a7/7bd50539-9fae-433d-8db6-37e69fe2ad6d.png)
This captures:
*   short-term momentum
*   micro-trend continuation
*   micro-reversal probability
It is more stable and less noisy than the 1-day horizon, while still being actionable.
### **Why these horizons?**
*   **H1** → High noise, strong test for model sensitivity
*   **H5** → Smoother signal, strong test for regime recognition
Together they provide a robust evaluation of trajectory-based reasoning.
## **4.2 Target Variables (Direction, Intensity, Buckets)**
The MVP focuses on direction-based probabilistic forecasting—not price-level forecasting.
### **Primary Target: Direction (Binary)**
For both horizons:
*   **Up** (return > 0)
*   **Down** (return ≤ 0)
This is the industry-standard benchmark used by:
*   asset managers
*   quantitative funds
*   academic research
### **Secondary Target: Intensity Buckets (Optional)**
We may also categorize returns into **5 directional-intensity buckets**:
*   Strong Up
*   Mild Up
*   Flat
*   Mild Down
*   Strong Down
Example (quantile-based or threshold-based):

| **Bucket**<br> | **Return Threshold**<br> |
| ---| --- |
| Strong Up | \> +1.5σ |
| Mild Up | \> 0 but ≤ +1.5σ |
| Flat | Between -0.5σ and +0.5σ |
| Mild Down | < 0 but ≥ -1.5σ |
| Strong Down | < -1.5σ |

These buckets are useful for:
*   scenario analysis
*   explainability
*   risk-aware decisions
*   probability distribution visualization
The MVP can include these buckets, but the key quantitative benchmarks are directional.
## **4.3 Temporal Structure: Observation → Forecast**
The core logic of the task:
### **Observation Window**
Kalibry observes the **previous 60 days** of asset evolution:
![](https://t90182117410.p.clickup-attachments.com/t90182117410/4c1772d6-5006-4585-88e9-cddf94a56086/586a45f7-bb14-40d5-971f-7419af032c9d.png)
Each xtx\_txt​ is a **45-dimensional state embedding** containing:
*   returns
*   volatility metrics
*   liquidity signals
*   momentum patterns
*   regime indicators
*   structural factors
### **Forecast Window**
Using these 60 observed days, Kalibry predicts:
*   the **direction of return at t+1** (H1)
*   the **cumulative direction from t+1 to t+5** (H5)
### **No Overlap Leakage**
The observed window ends precisely at day **t**.
The prediction horizons start strictly at **t+1**.
This is enforced for:
*   training
*   validation
*   testing
*   backtesting
### **Temporal Diagram**
![](https://t90182117410.p.clickup-attachments.com/t90182117410/71031eba-8d98-4182-b2d7-c3f2f8b406ab/77918014-5b6d-4ce2-9866-c8a588927241.png)
This structure is fundamental to all trajectory-based reasoning tasks.
## **4.4 Constraints & Realistic Assumptions**
To maintain scientific rigor, we use the following constraints and assumptions:
### **1\. No Arbitrage Assumption**
The model does not attempt to:
*   generate signals
*   predict exact price levels
*   exploit high-frequency inefficiencies
It predicts **directional probability**, not absolute returns.
### **2\. Short-Term Forecasting Only**
Long-term price forecasts are unrealistic and scientifically weak.
The MVP stays within 1–5 day horizons where:
*   empirical analogs are meaningful
*   patterns are detectable
*   distributional fidelity is higher
### **3\. Financial Markets Are Non-Stationary**
Kalibry does **not** rely on:
*   stationarity
*   linearity
*   constant variance
*   fixed regime structures
Instead, trajectory matching naturally adapts to different regimes.
### **4\. All Predictions Are Probability Distributions**
Not deterministic values.
Outputs include:
*   Up probability
*   Down probability
*   Entropy / confidence
*   Evidence breakdown
### **5\. No Use of Future Data in Any Form**
The system strictly prohibits leakage via:
*   rolling normalization
*   forward-only feature construction
*   walk-forward backtesting
*   analog window filtering
### **6\. Multi-Asset Generalization**
The model does not train one model per asset.
It recognizes **universal behavioural trajectories** across assets.
This supports the thesis that markets share structural similarities.
## **4.5 Why Short-Term Directional Prediction Matters**
Short-term (1–5 day) directional prediction is valuable because it is:
### **1\. Actionable in Real Institutional Workflows**
Funds use short-term predictions for:
*   risk hedging
*   position sizing
*   exposure management
*   entry/exit timing
*   rotation between assets
Directional probability is a direct input to these workflows.
### **2\. Statistically Measurable and Benchmarkable**
Unlike long-term forecasts, short-term movement has:
*   clear ground truth
*   quantifiable metrics
*   strong academic precedent
*   deterministic comparison with baselines
Short-horizon accuracy is the gold standard in financial AI evaluation.
### **3\. Less Subject to Structural Breaks**
5-day windows are far more stable than:
*   1-month
*   3-month
*   6-month
This makes trajectory reasoning **reliable, repeatable, testable**.
### **4\. Enables Real-World Use Cases**
Examples:
*   “Probability that NVDA is up tomorrow: 64%”
*   “Probability of a 5-day positive drift on BTC: 58%”
*   “High-entropy → low confidence, suggests caution”
These outputs are extremely valuable for:
*   risk teams
*   portfolio managers
*   quantitative analysts
*   liquidity desks
### **5\. Perfect Fit for MVP Validation**
Short-term forecasts allow us to rapidly:
*   test multiple horizons
*   evaluate multiple assets
*   compare against multiple baselines
*   conduct robust statistical validation
This accelerates both **technical** and **business** feedback loops.
# **5\. State Embedding Architecture**
Kalibry’s financial forecasting engine is built on a fundamental concept:
each day in an asset’s life is represented by a **high-dimensional state vector** that captures the asset’s behaviour, context, and regime.
This vector forms the atomic unit of trajectory intelligence.
## **5.1 Rationale for State-Level Representation**
Most forecasting methods use raw prices or simple transformations (returns, moving averages).
These representations are insufficient because they:
*   fail to capture _regime-dependent behaviour_
*   ignore _multi-scale dynamics_
*   cannot represent _cross-asset similarities_
*   do not reflect _structural evolution over time_
Kalibry instead models each day as a **45-dimensional state vector**, where each dimension is a carefully engineered indicator of behaviour:
*   trend
*   volatility
*   liquidity
*   momentum
*   mean-reversion
*   market regime
*   structural context
This representation enables the model to:
### **1\. Compare behaviour across assets**
A normalized state embedding makes **NVDA 2017** comparable to **AAPL 2021**, **BTC 2020**, or **EUR/USD 2018**.
### **2\. Encode both short-term and long-term information**
Including signals across 10–60 day windows.
### **3\. Create stable, smooth trajectories**
Reducing noise and enabling pattern recognition.
### **4\. Enable empirical, evidence-based reasoning**
Instead of predicting numbers, the system compares trajectories in a high-dimensional behaviour space.
The state-level representation is the backbone of Kalibry’s architecture.
## **5.2 The 45-Dimensional Engineered Feature Vector**
Each daily state is represented as:
xt∈R45x\_t \\in \\mathbb{R}^{45}xt​∈R45
These 45 features are grouped into **six conceptual families**, each capturing a key behavioural axis.
### **Family 1 — Price & Returns (8 features)**

| **Feature**<br> | **Description**<br> |
| ---| --- |
| Daily return | (Close - Prev Close) / Prev Close |
| Log return | log(C\_t / C\_{t-1}) |
| 5-day return | Short-term drift |
| 10-day return | Momentum foundation |
| 20-day return | Medium-term drift |
| 60-day return | Long horizon drift |
| Z-score of returns | Standardized return signal |
| Drawdown indicator | Current % from peak |

### **Family 2 — Volatility & Range (10 features)**

| **Feature**<br> | **Description**<br> |
| ---| --- |
| True Range | ATR building block |
| ATR(10) | Short-term volatility |
| ATR(20) | Medium-term volatility |
| Rolling std (5d) | Immediate noise level |
| Rolling std (10d) | Short-term volatility |
| Rolling std (20d) | Medium-term volatility |
| Rolling std (60d) | Regime volatility |
| Range/Close ratio | Price dispersion |
| High-Low % | Volatility spike indicator |
| Volatility regime score | Normalized regime indicator |

### **Family 3 — Momentum & Trend Structure (10 features)**

| **Feature**<br> | **Description**<br> |
| ---| --- |
| RSI(14) | Overbought/oversold |
| MACD (12/26) | Trend shift |
| MACD Signal | Trend confirmation |
| MACD Histogram | Momentum strength |
| SMA 5d slope | Short-term micro-trend |
| SMA 10d slope | Momentum stability |
| SMA 20d slope | Intermediate trend |
| SMA 5/20 crossover state | Trend polarity |
| Price vs SMA 20d | Mean reversion signal |
| Trend regime score | Multi-scale trend indicator |

### **Family 4 — Liquidity & Volume (7 features)**

| **Feature**<br> | **Description**<br> |
| ---| --- |
| Volume z-score | Liquidity burst indicator |
| Volume 5d MA | Short-term liquidity |
| Volume 20d MA | Baseline liquidity |
| Volume anomaly | Spike/decay |
| VWAP deviation | Price/liquidity imbalance |
| Turnover ratio | Trading intensity |
| Volume regime score | Liquidity state |

### **Family 5 — Structural & Cross-Asset Factors (6 features)**
Used to make analogies across unrelated assets.

| **Feature**<br> | **Description**<br> |
| ---| --- |
| Asset class embedding | One-hot or learned encoding |
| Sector encoding (equities) | Market structure context |
| FX/currency proxy | Sensitivity context |
| Crypto regime proxy | Volatile structure indicator |
| Market beta (index-relative) | Systematic risk |
| Correlation to index (20d) | Trend coupling |

### **Family 6 — Regime Indicators (4 features)**

| **Feature**<br> | **Description**<br> |
| ---| --- |
| Volatility regime | Low/medium/high stress |
| Trend regime | Uptrend / downtrend / range |
| Liquidity regime | High/low volume |
| Market macro regime | Encodes external macro stress |

### **Total: 45 Features**
These features were chosen because they:
*   encode behaviour
*   generalize across assets
*   are stable over time
*   create meaningful analogs
This is the **state-of-the-art foundation** for trajectory intelligence.
## **5.3 Multi-Scale KPI Design (10/20/30/60-day signals)**
Financial behaviour is inherently multi-scale.
Some patterns are short-lived (10 days), while others are persistent (60 days).
The embedding captures:
*   **short-term microstructure** (5–10 days)
*   **medium-term momentum** (20–30 days)
*   **long-term trend shape** (60 days)
This multi-scale encoding greatly enhances Kalibry’s ability to:
*   recognize repeated behaviour patterns
*   detect early-stage regime changes
*   match trajectories across very different eras
*   reduce noise sensitivity
Without multi-scale design, trajectory embeddings would be too myopic or too diluted.
## **5.4 Asset Identity & Regime Encoding**
To compare assets across asset classes, Kalibry includes:
### **Asset Identity Encoding**
Examples:
*   Equity
*   Index
*   FX
*   Crypto
This helps distinguish structural differences.
### **Regime Encoding**
Includes:
*   volatility regime
*   liquidity regime
*   trend regime
*   macro regime
This ensures that:
*   AAPL in a low-volatility uptrend
*   is not matched to
*   BTC in a high-volatility crash.
The embedding system is designed to preserve **structural similarity**, not merely price patterns.
## **5.5 Rolling Normalisation Strategy**
Normalization is performed _per timestamp_, using only past data:
![](https://t90182117410.p.clickup-attachments.com/t90182117410/9a3037b1-a115-454d-bfe2-8ff5a06fe3cd/26e81f76-bc00-40ce-a866-5a8bc3fa7033.png)
### **Why this matters:**
*   prevents data leakage
*   allows cross-asset comparability
*   stabilizes distributions
*   ensures smooth trajectories
*   adapts to evolving regimes
This is a critical scientific detail that makes the model robust and realistic.
## **5.6 Optional: Learned Latent Embedding (R16–R24)**
Although the MVP relies primarily on engineered features, the system allows an optional learned compression layer.
A small encoder network:
![](https://t90182117410.p.clickup-attachments.com/t90182117410/b12af61f-77c4-45eb-8584-daef6708ccdb/dc6df57d-9ab2-475b-85b0-dd4cd2057c3e.png)
Where:
![](https://t90182117410.p.clickup-attachments.com/t90182117410/89aa2cd3-54a2-4627-a82a-ca12cc3dfd7e/0bad6569-0814-431c-9a8d-d9b56a4f4d10.png)
Possible learning strategies:
*   autoencoder
*   contrastive learning
*   triplet-loss (similar/dissimilar patterns)
### **Purpose:**
*   compress features
*   smooth noisy signals
*   reveal latent structure
*   improve trajectory discriminability
This module is **optional**, but will be tested as part of the MVP research.
## **5.7 Stability, Smoothness, and Comparability Across Assets**
The embedding must satisfy three properties:
### **Property 1 — Stability**
Small changes in underlying data → small changes in embedding.
This prevents:
*   false similarity triggers
*   sensitivity to daily noise
*   instability in trajectory shapes
### **Property 2 — Smoothness**
Embeddings must evolve gradually day-to-day.
A trajectory is meaningful only if points are naturally aligned.
### **Property 3 — Cross-Asset Comparability**
Vectors must have comparable meaning across:
*   equities
*   indices
*   FX
*   crypto
For example:
*   NVDA “volatility rising”
*   should look similar to
*   BTC “volatility rising”,
*   in embedding space, despite scale differences.
Achieved via:
*   rolling normalization
*   regime encoding
*   multi-scale design
# **6\. Trajectory Construction**
Kalibry’s predictive engine is built on the idea that **an entity’s recent evolution can be represented as a trajectory in state space**.
For financial assets, each trajectory consists of a sequence of **state embeddings** describing behaviour over time.
## **6.1 Why 60 Days (Trade-off Between Signal & Noise)**
The MVP uses a **60-day observation window** to model each asset’s short-term evolution.
A 60-day window provides the optimal balance between:
### **1\. Sufficient Behavioural Context**
Within 60 days, assets typically undergo:
*   micro-trend formation and decay
*   short volatility regimes
*   liquidity shifts
*   early-stage reversals
*   corrections or drift phases
This provides a rich behavioural footprint.
### **2\. Noise Reduction**
Short windows (10–20 days) are too sensitive to daily noise.
Long windows (>90 days) dilute critical short-term signals.
60 days yields:
*   stable trajectory shape
*   enough historical analogs
*   adequate representation of regimes
### **3\. Strong Empirical Support**
Internal experiments and financial research show that:
*   40–60 day patterns have the highest repeatability
*   60-day windows provide the best signal-to-noise ratio
*   most technical indicators stabilize in this range
This makes 60 days the ideal core window for trajectory reasoning.
## **6.2 Structure of a Trajectory (60 × 45)**
![](https://t90182117410.p.clickup-attachments.com/t90182117410/64b814f6-089c-40ed-90dc-482d8ac22154/6861d925-3dcf-453e-a3ea-68dd4170bcdf.png)  
##   

Thus each trajectory element:
*   captures **60 days of behavioural evolution**
*   encodes **price**, **volatility**, **momentum**, **liquidity**, **regime**, and **context**
*   forms a time-series matrix describing the asset’s recent dynamics
### **Visual Interpretation**
A trajectory is a **shape** in a 45-dimensional behavioural manifold.
Patterns include:
*   upward momentum phases
*   volatility clusters
*   liquidity droughts
*   slow reversals
*   high-stress events
Kalibry’s reasoning engine compares these shapes to historical shapes.
## **6.3 Alternative Window Lengths (30, 90 Days)**
Although 60 days is the primary window, the MVP also tests:
### **1\. 30-Day Window**
*   captures short-term microstructure
*   highly sensitive to noise
*   useful for testing responsiveness
*   less stable across regimes
*   may generate more false analogs
This serves as a robustness benchmark.
### **2\. 90-Day Window**
*   captures deep behavioural shifts
*   smoother trajectory shapes
*   more stable across shocks
*   but risks diluting short-term signal
Useful for testing:
*   long-pattern matching
*   multi-regime analogs
*   reduced noise scenarios
### **Window Length Evaluation**
The MVP will compare:
*   directional accuracy
*   calibration
*   trajectory similarity quality
*   sensitivity to regime shifts
This ensures Kalibry lands on the optimal operational window for v1.
## **6.4 Flattening / Pooling / Learned Temporal Encoding**
Trajectories must be encoded into vector representations to be searchable in HPVD.
The MVP evaluates three encoding methods:
### **A. Flattening (Simple Baseline)**
![](https://t90182117410.p.clickup-attachments.com/t90182117410/b4180f25-0b72-451d-9164-f35124c5cb69/49230313-5103-4b85-a626-f81425665539.png)
Pros:
*   simplest
*   surprisingly effective
*   preserves all information
Cons:
*   high dimensionality
*   sensitive to misalignment
*   memory-inefficient
Used as a **baseline**, not final method.
### **B. Temporal Pooling (Primary MVP Method)**
Using:
*   mean pooling
*   max pooling
*   std pooling
*   attention pooling (optional)
For example:
![](https://t90182117410.p.clickup-attachments.com/t90182117410/964f0a08-38e5-4a84-ac35-b53338e0c638/af2d8364-8513-4106-ac25-586403f2485f.png)
Or concatenation:
![](https://t90182117410.p.clickup-attachments.com/t90182117410/2ef3e3b8-0457-40d9-a5cf-52089385ab74/70478f0a-bfd1-490e-9617-e28afc580c0f.png)
Pros:
*   robust to noise
*   compact
*   stable
*   fast similarity search
This is the **default MVP method**.
### **C. Learned Temporal Encoder (Optional Extension)**
A lightweight temporal model:
*   1D CNN
*   small GRU
*   time-series autoencoder
*   transformer-lite
Produces:
![](https://t90182117410.p.clickup-attachments.com/t90182117410/63986c31-9bbd-4220-8ecb-d1b10ebd2f1a/3365eccf-f1da-4c20-aa1e-7c4cc066591c.png)
Pros:
*   compresses behavioural shape
*   captures long dependencies
*   discovers latent structure
Cons:
*   requires training
*   risk of overfitting
*   more complex
This is included as part of the experimental extensions.
## **6.5 Creating the Trajectory Dataset**
The trajectory dataset is built by sliding a 60-day window across each asset’s historical data.
For each asset:
![](https://t90182117410.p.clickup-attachments.com/t90182117410/38fc2f9d-1705-4065-88d5-d7c42cadadea/2f2172dd-9872-4c84-b7b2-960f69506a7e.png)
### **Dataset Structure**
Each row includes:
*   trajectory embedding
*   asset ID
*   timestamp
*   future return (1-day, 5-day)
*   label (Up/Down)
*   metadata (regimes, market context)
This dataset powers:
*   training
*   validation
*   testing
*   similarity search
*   probability aggregation
The dataset is multi-asset and multi-regime.
## **6.6 Ensuring No Overlap Leakage**
Leakage prevention is one of the **most critical components** of the MVP.
Kalibry implements strict safeguards:
### **1\. Observation Windows Never Overlap with Forecast Windows**
Trajectories end at day **t**.
H1 starts at **t+1**.
H5 ends at **t+5**.
No future information contaminates the input.
### **2\. Historical Analog Trajectories Must Be Entirely in the Past**
![](https://t90182117410.p.clickup-attachments.com/t90182117410/0f2ea3c8-bf6d-4471-b279-275c541b889f/22c09312-df9b-4685-a79f-257f1d959b3e.png)
### **3\. Rolling Normalization**
All rolling means/stds:
*   use only past data
*   do not look ahead
*   are aligned across assets
### **4\. Time-Based Train/Val/Test**
Data split chronologically:
*   Train (2013–2019)
*   Validation (2020–2021)
*   Test (2022–2024)
No back-casting.
### **5\. Walk-Forward Backtesting**
Predictions mimic real-world usage:
*   train on past
*   predict future
*   roll forward
This ensures integrity.
### **6\. Cross-Asset Contamination Controlled**
Even though analogs are cross-asset, **temporal alignment rules** are enforced.
Patterns are shared across assets
—information is **never** shared across future time periods.
#   

# **7\. Kalibry Forecasting Engine**
The Kalibry Forecasting Engine is the heart of the MVP.
It transforms trajectory representations into **robust, explainable, probabilistic predictions** by combining:
*   behavioural embeddings
*   similarity search
*   pattern-based analog reasoning
*   empirical distribution forecasting
*   calibration
*   uncertainty quantification
This section describes the full reasoning framework.
## **7.1 High-Level Reasoning Paradigm**
Kalibry does **not** predict the future directly.
Instead, it uses a _fundamentally different paradigm_:
### **Instead of prediction → Kalibry performs analogical reasoning.**
Given the current trajectory of an asset:
Tt\\mathcal{T}\_tTt​
The system:
1. **Finds historical trajectories that look similar**
2. **Observes how those historical trajectories evolved**
3. **Aggregates their outcomes into a probability distribution**
Thus:
Kalibry predicts the future _by learning from the past trajectories that most closely resemble the present one_.
This mirrors how human analysts, quants, and traders reason:
*   “This looks like AAPL during the 2017 breakout.”
*   “NVDA in this volatility cluster behaved like AMD in 2019.”
*   “This pattern matches three similar structures that led to upside continuation.”
Kalibry transforms this human intuition into a formal, quantifiable framework.
## **7.2 Trajectory Similarity Search (HPVD)**
At the core lies the **Hybrid Probabilistic Vector Database (HPVD)** — a custom vector search engine optimized for:
*   high-dimensional temporal embeddings
*   hybrid distance metrics
*   regime-aware filtering
*   probabilistic retrieval
Given a trajectory embedding EtE\_tEt​, HPVD retrieves the most similar historical trajectories using:
*   approximate nearest neighbour search (ANN)
*   HNSW or Faiss-based indexing
*   optional learned metric adjustment
*   filtering by asset type, regime, and temporal validity
HPVD stores millions of historical trajectories across all assets.
![](https://t90182117410.p.clickup-attachments.com/t90182117410/d9b5bdbe-7ef4-46ba-a226-6e9b92bcd02b/1f887867-d94b-4039-8e77-d2f4537f61dc.png)
## **7.3 Distance Metrics & Alternatives**
Kalibry supports multiple distance metrics for evaluating trajectory similarity.
### **Primary Metric: Cosine Distance**
Robust to:
*   magnitude differences
*   scaling issues
*   structural distortions
Captures _shape similarity_ in embedding space.
### **Alternative Metrics Evaluated**
#### **1\. Euclidean Distance**
Useful when embedding normalization is strong.
Pros: sensitive to overall amplitude
Cons: sensitive to noise
#### **2\. Manhattan Distance**
Useful for sparse-feature representations.
#### **3\. Soft-DTW (Dynamic Time Warping)**
Captures temporal shape similarity even with small misalignments.
Pros: shape-aware
Cons: significantly slower
Used experimentally for validation.
#### **4\. Learned Metric**
A neural metric layer trained to:
*   compress
*   smooth
*   reweight feature dimensions
Produces small but meaningful improvements in trajectory discrimination.
### **Metric Selection Strategy**
The MVP systematically tests:
*   cosine (default)
*   Euclidean
*   soft-DTW
*   learned metric
This allows us to validate the robustness of trajectory similarities.
##   

## **7.4 Family Discovery: K Nearest Trajectories**
Once HPVD retrieves the top-K most similar trajectories, Kalibry identifies **families** of behavioural analogs.
### **Definition of a Family**
A family of trajectories is a set of historical windows that:
*   share similar multi-scale behaviours
*   occurred in similar volatility or trend regimes
*   evolved similarly after the observation window
Families often represent:
*   momentum continuation patterns
*   volatility contraction patterns
*   reversal signatures
*   stress events
*   accumulation/distribution cycles
Kalibry computes similarity weights:
​![](https://t90182117410.p.clickup-attachments.com/t90182117410/31e36824-55b1-4324-b11f-1c4fdd8a3c5c/d2e8124f-c016-43b7-b00d-754677b61b6d.png)
Where closer trajectories have higher influence.
### **Filtering & Validation Rules**
To maintain robust analog families:
*   Similarity threshold enforced
*   Regime consistency filters
*   No overlap with future windows
*   Asset-type compatibility checks
If too few valid analogs exist, the system:
*   lowers K
*   increases tolerance
*   raises uncertainty score
This guarantees scientific integrity.
## **7.5 Empirical Distribution Aggregation**
With K similar trajectories and their known outcomes, Kalibry builds:
### **Empirical Outcome Distributions**
![](https://t90182117410.p.clickup-attachments.com/t90182117410/cc40efd8-283c-45cb-b5aa-7b82e31b9e3b/02797210-877c-4113-8d2d-1c9da699e112.png)
### **Optional: Multi-bucket Distributions**
5-bin distribution:
*   Strong Up
*   Mild Up
*   Flat
*   Mild Down
*   Strong Down
This is crucial for institutional-grade modeling:
*   provides distribution, not point estimate
*   supports scenario analysis
*   produces risk-aware forecasts
### **Why This Works**
This approach exploits **empirical regularities** in financial behaviour.
Many trajectory patterns repeat over time:
*   volatility compression → breakout
*   accelerated drawdowns → relief bounce
*   sustained drift → continuation
*   regime shifts → defensive patterns
Kalibry captures these in a formalized, quantitative way.
## **7.6 Probability Calibration (Natural Advantage of Kalibry)**
Kalibry has a _built-in structural advantage_ over neural nets:
its probabilities are **naturally calibrated** because they are computed from empirical frequencies.
### **Neural Models**
Neural nets are often:
*   overconfident
*   under-calibrated
*   poorly aligned with empirical outcomes
*   sensitive to regime changes
Calibration must be corrected explicitly.
### **Kalibry**
Kalibry's probabilities:
*   reflect actual historical frequency
*   adjust dynamically to regime similarity
*   require no additional calibration layer
*   are stable across time and assets
*   handle regime transitions
*   naturally represent uncertainty
This is crucial in:
*   portfolio construction
*   risk assessment
*   regulatory compliance
*   institutional adoption
### **Quantitative Advantage**
Expected Brier Score improvement:
*   **+10–20%** vs LSTM and Transformer
*   stable reliability curves
*   consistent performance in stress environments
Calibration is a **core differentiator**.
## **7.7 Confidence, Entropy & Uncertainty Measures**
Kalibry does not output a single probability.
It outputs a **probabilistic belief state** with full uncertainty quantification.
### **1\. Entropy of Prediction**
![](https://t90182117410.p.clickup-attachments.com/t90182117410/26a73945-5ea5-4ba7-8dc4-73e26f3f6f16/83e1304e-a114-49b9-93a6-b50ea7961647.png)
High entropy → uncertainty.
Low entropy → confident signal.
### **2\. Variance of Future Outcomes**
Computed from the distribution of returns across neighbors.
Indicates:
*   consistency of analogs
*   stability of prediction
*   presence of regime ambiguity
### **3\. Spread Across Trajectory Families**
If matched trajectories fall into different families, this indicates:
*   regime boundary
*   mixed signal
*   caution required
### **4\. Density of Similar Matches**
If only a few valid trajectories exist:
*   similarity score drops
*   prediction confidence decreases
*   system signals low conviction
### **5\. Analog Cohesion Score**
Measures similarity among the neighbors themselves.
High cohesion → strong behavioural cluster
Low cohesion → mixed or noisy analogs
### **6\. Uncertainty Metadata for Explainability**
Kalibry outputs:
*   top analog trajectories
*   their outcomes
*   their weights
*   final aggregated probabilities
*   entropy score
*   “confidence reason” summary
This creates **AI Act–aligned transparency** and high trustworthiness.
#   

# **8\. Explainability Model**
Explainability is not an optional add-on for Kalibry.
It is a **core architectural feature**, designed to meet:
*   institutional expectations
*   risk-management requirements
*   regulatory standards (EU AI Act)
*   and the need for trust in models operating in high-stakes environments
Trajectory Intelligence enables **full transparency**, because each prediction can be traced back to concrete historical patterns and behaviours.
## **8.1 Why Explainability is Essential in Finance**
Financial institutions operate in a domain where:
*   risk is highly regulated
*   decisions must be justifiable
*   model transparency is mandatory
*   black-box AI models are rarely approved
*   compliance teams require audit trails
*   regulators demand explainable outputs
### **Why black-box models fail in financial adoption:**
1. **Neural models offer no insight into “why”**
2. Their outputs cannot be justified to risk committees.
3. **Regulators require transparency**
4. The EU AI Act mandates explainability for high-risk systems.
5. **Portfolio managers need reasoning, not just numbers**
6. A forecast without rationale is not actionable.
7. **Fiduciary responsibility**
8. Institutions must prove their models are reasonable, not mystical.
### **Kalibry solves all of these requirements natively.**
Instead of opaque weights inside a neural network, Kalibry bases predictions on:
*   real historical analogs
*   matched trajectories
*   evidence graphs
*   clear probability distributions
*   regime and feature components
This creates the most transparent forecasting engine in finance.
## **8.2 Evidence Graphs: Core Concept**
**Evidence Graphs** are Kalibry’s foundational explainability mechanism.
For each prediction, Kalibry generates an evidence graph showing:
1. **Current 60-day trajectory**
2. **K nearest historical trajectories**
3. **Similarity scores**
4. **Outcome distributions for each analog**
5. **Final aggregated probability**
6. **Uncertainty indicators**
7. **Regime alignment information**
### **Purpose of Evidence Graphs**
They provide:
*   transparent justification
*   pattern-based reasoning
*   scenario-based outcomes
*   explainable probabilistic forecasts
An investment committee can immediately see:
*   why a prediction was made
*   what historical patterns support it
*   how strong or weak the analogs are
This is a _breakthrough_ in explainable AI for finance.
## **8.3 Trajectory Visualization & Pattern Families**
Kalibry visualizes:
### **1\. The current trajectory**
A 60-day behavioural path in:
*   returns
*   volatility
*   trend
*   liquidity
*   regime
### **2\. The matched pattern families**
Kalibry groups similar analogs into **families**, such as:
*   “low-volatility uptrend breakout patterns”
*   “mean-reversion clusters”
*   “high-volatility reversals”
*   “momentum exhaustion structures”
Trajectory clusters reveal:
*   common behaviours
*   predictable patterns
*   recurring shapes across assets
This supports:
*   investment intuition
*   analyst validation
*   comparative reasoning
*   institutional presentation
## **8.4 Historical Analogs & Outcome Histories**
For every matched trajectory, Kalibry displays:
### **1\. Exact historical period**
Example:
*   "NVDA, March 2017 – April 2017"
*   "AAPL, July 2020 – August 2020"
*   "BTC, January 2021 – February 2021"
### **2\. What happened next**
Kalibry retrieves:
*   1-day forward return
*   5-day forward return
*   the actual directional outcome
### **3\. Visualization of future paths**
Shows how the market evolved after those trajectories.
This reveals:
*   whether analogs led to continuation
*   whether they led to reversals
*   the distribution of future outcomes
### **4\. Impact weight in forecasting**
Each analog is weighted by similarity:
![](https://t90182117410.p.clickup-attachments.com/t90182117410/8b006cda-de36-4a1d-89a7-44c8b98224bb/73195bee-3174-45a7-9d11-3744e60ef6fa.png)​
This helps analysts understand:
*   which analogs matter most
*   which analogs were downweighted
Analysts can _literally see_ the ancestry of the prediction.
## **8.5 Multi-Dimensional Feature Similarity Breakdown**
Kalibry provides a detailed feature-level explainability layer.
For each matched trajectory, Kalibry computes:
### **Similarity heatmaps for:**
*   returns
*   volatility
*   trend
*   momentum
*   liquidity
*   cross-asset structure
*   regime indicators
Analysts can see:
*   which dimensions drive similarity
*   which behaviours align most strongly
*   which features diverge
Example insights:
*   “The strongest similarity arises from volatility and trend structure.”
*   “Liquidity patterns differ slightly but remain acceptable.”
*   “Momentum slope alignment is nearly identical.”
This level of granularity is _unavailable_ in black-box models.
## **8.6 Human-Readable Narrative for Investors & Analysts**
Kalibry automatically generates an **interpretation summary**, written in plain English, suitable for:
*   investment committees
*   internal analysts
*   client-facing reports
*   compliance officers
### **Example Narrative**
**“The current 60-day trajectory for NVDA matches two historical families:**
**(1) low-volatility momentum clusters (AAPL 2020),**
**(2) early breakout formations (NVDA 2017).**
**Across these analogs, 72% of cases resulted in a positive 5-day return.**
**Similarity is driven primarily by volatility compression, trend stability, and momentum slope alignment.**
**Probability Up (H5) = 0.72 with moderate confidence (entropy = 0.41).”**
This narrative is:
*   understandable
*   defensible
*   transparent
*   regulatory-safe
It replaces opaque AI explanations with _behavioral reasoning_.
## **8.7 Alignment with EU AI Act Principles**
The EU AI Act requires:
*   transparency
*   traceability
*   explainability
*   human oversight
*   risk mitigation
*   non-black-box logic
*   accountability
Kalibry is inherently aligned with these principles.
### **Kalibry’s Compliance Advantages**
#### **1\. Transparent basis for every prediction**
Evidence Graphs show exactly _why_ a prediction was made.
#### **2\. Traceable to historical precedents**
Each prediction is tied to _concrete historical behaviours_, not hidden model weights.
#### **3\. Human-readable rationales**
Narratives are automatically generated and understandable.
#### **4\. No deep-learning opacity**
Kalibry is not a neural black box.
Its reasoning process is observable and auditable.
#### **5\. Clear uncertainty metrics**
Entropy, variance, cohort consistency → explicit uncertainty quantification.
#### **6\. Documented data lineage**
Trajectory datasets are fully versioned and traceable.
#### **7\. Non-discriminatory logic**
Kalibry compares behaviours, not identities—no risk of sensitive attribute bias.
#   

#   

#   

#   

# **9\. Baselines & Benchmarks**
Robust evaluation requires comparison against well-established baselines.
The goal is not only to prove that Kalibry performs well, but to show that it **outperforms the full spectrum of alternative approaches**, from naive statistical heuristics to advanced deep learning architectures.
This chapter defines which baselines are included, how they are trained, and what success against each group demonstrates.
## **9.1 Purpose of Strong Baseline Comparison**
Baseline comparisons serve five critical purposes:
### **1\. Scientific Validity**
Trajectory Intelligence is a novel paradigm. It must demonstrate superiority over classical time-series models and modern ML approaches to be taken seriously by researchers, quants, and investment committees.
### **2\. Real-World Credibility**
Financial practitioners evaluate forecasting systems by one standard:
**Does it beat traditional benchmarks?**
By outperforming widely trusted models, Kalibry earns institutional relevance.
### **3\. Proof of Generality**
The baselines span:
*   naive heuristics
*   classical econometrics
*   recurrent neural networks
*   transformers
Outperforming all of them demonstrates **cross-paradigm generality**.
### **4\. Identification of Model Strengths**
Comparisons reveal:
*   where Kalibry excels
*   where analog reasoning provides structural advantages
*   which regimes showcase clear superiority
### **5\. Regulatory & Compliance Confidence**
Regulators require that new models justify their use with clear, benchmark-based evidence.
Kalibry must prove:
*   higher accuracy
*   better calibration
*   more stability
*   lower risk
across multiple methodologies.
## **9.2 Naive Baselines (Random Walk, Always-Up, Momentum 20d)**
Naive baselines are essential for establishing a _minimum expected performance threshold_.
### **1\. Random Walk (RW)**
Predicts each day with:
*   P(Up) = 0.5
*   P(Down) = 0.5
**Expected accuracy: 50%.**
This baseline is fundamental because:
*   financial markets are notoriously noisy
*   beating RW indicates non-random structure detection
*   RW is often surprisingly hard to beat consistently
### **2\. Always-Up Baseline**
Predicts:
*   Up every day
*   Up every 5-day window
Since most major indices and large-cap equities drift upward long-term, this yields:
*   ~52–55% accuracy
*   (depending on asset universe)
Beating Always-Up proves Kalibry is genuinely learning behavioural patterns, not relying on drift.
### **3\. Momentum 20-Day Baseline**
Predicts:
*   **Up** if 20-day return > 0
*   **Down** if 20-day return < 0
*   break-even handled as Flat/Up
Expected accuracy:
*   **52–55%**
This is a fair test because momentum is one of the strongest simple predictors in finance.
### **What Beating Naive Baselines Proves**
*   Kalibry captures _true behavioural patterns_, not randomness
*   Outperforms simple signals used by many traders
*   Detects structural features beyond drift and momentum
*   Provides consistent improvement across assets and regimes
## **9.3 Classical Econometric Models**
These models have been pillars of financial forecasting for decades.
Each uses linear relationships and strong assumptions (e.g., stationarity).
Included models:
### **1\. AR(p): Autoregressive Models**
Predict future returns from past returns:
![](https://t90182117410.p.clickup-attachments.com/t90182117410/7d53c28b-5a31-42e9-9409-ffe907c073ca/38d3d33c-fc90-4bef-b8d8-15b39935101d.png)
Use cases:
*   short-term return autocorrelation
*   micro mean-reversion
### **2\. MA(q): Moving Average Models**
Predict using past shocks:
![](https://t90182117410.p.clickup-attachments.com/t90182117410/0813aa5f-df5a-467d-994e-1618f4d2bb39/94f53ab8-8344-4bf6-9b70-4908ee22b3ee.png)
Useful for capturing noise structures.
### **3\. ARMA(p,q)**
Combination of AR and MA.
Baseline performance ~51–54%.
### **4\. ARIMA**
Adds differencing to handle non-stationarity:
![](https://t90182117410.p.clickup-attachments.com/t90182117410/c2d2786c-5e05-4fab-a0d0-5398bab5e790/f18ba500-a91a-44b9-bf1a-fb1031b81c97.png)
Classical standard for time-series.
### **5\. GARCH(1,1)**
Models volatility, not returns:
​![](https://t90182117410.p.clickup-attachments.com/t90182117410/8256dc38-8532-4e2e-8eba-2d715c2d22f3/f97a65b1-57dc-419c-a3bd-a9ec0697bda9.png)
Direction is inferred via volatility-adjusted return logic.
Useful for stress and high-volatility regimes.
### **6\. AR-GARCH**
Combines return and volatility modeling:
![](https://t90182117410.p.clickup-attachments.com/t90182117410/dc7ffcb8-f8fa-403d-baea-2272b867b37b/e4b5aaab-9a46-4637-8c93-1c9a44395432.png)
This is the strongest classical baseline.
### **Why These Baselines Are Important**
They are:
*   mathematically mature
*   widely used in hedge funds
*   benchmark standards in academic literature
*   robust in certain regimes (low-volatility periods)
### **What Beating Econometric Models Proves**
*   Kalibry outperforms decades of research
*   analog reasoning captures non-linear structures classical models cannot
*   performance is not merely due to autocorrelation or drift
*   regime transitions are handled more smoothly
*   forecasts are better calibrated
This is critical for quant adoption.
## **9.4 Modern ML Baselines**
These models represent the current “standard AI approach” to forecasting.
We include:
### **1\. LSTM (Long Short-Term Memory)**
Captures temporal dependencies using gated recurrent units.
Strengths:
*   good at short-term pattern recognition
*   captures non-linear temporal relationships
Weaknesses:
*   overfits on noisy financial data
*   poor out-of-sample stability
*   low calibration
*   expensive to train
*   no interpretability
Expected performance: **53–56%** accuracy.
### **2\. GRU (Gated Recurrent Unit)**
Simpler than LSTM, similar behaviour.
Expected: **similar or slightly worse** than LSTM.
### **3\. Time-Series Transformer**
A modern architecture using:
*   attention
*   positional encoding
*   deep sequence modelling
Strengths:
*   long-range dependency modelling
*   powerful in structured time-series
*   highly flexible
Weaknesses:
*   very high data requirements
*   extremely high variance
*   highly unstable in financial forecasting
*   poor calibration
*   complete lack of interpretability
Expected performance: **53–57%**, with high volatility.
### **What Beating ML Baselines Proves**
*   Kalibry can outperform deep learning without huge datasets
*   analog reasoning is more robust in noisy and non-stationary environments
*   Kalibry’s calibration is superior
*   explainability advantage becomes overwhelming
*   lower computational cost → economic superiority
*   Kalibry generalizes better across regimes and assets
This is essential to prove that Kalibry is not simply “another ML model.”
## **9.5 How Each Model Is Trained & Evaluated**
All baselines are trained under identical conditions:
### **Training Procedure**
*   train on 2013–2019
*   validate on 2020–2021
*   test on 2022–2024
*   walk-forward evaluation
*   rolling parameter optimization (AR/GARCH)
*   standardization applied identically
*   sequence lengths matched to Kalibry
*   classification targets (Up/Down) identical
### **Evaluation Metrics**
*   directional accuracy (1-day / 5-day)
*   Brier Score (probability calibration)
*   entropy / confidence
*   regime-stratified performance
*   confusion matrices
### **Fair Comparison Rules**
*   no model sees future data
*   normalization is leakage-free
*   validation tuning is equal for all
*   predictions made at same timestamps
*   identical assets across models
This ensures scientific and institutional credibility.
## **9.6 What Victory Over Each Baseline Demonstrates**
Each category demonstrates a different competitive advantage:
### **1\. Beating Naive Baselines**
**⇒ Kalibry extracts real signal beyond noise and drift.**
### **2\. Beating Classical Econometrics**
**⇒ Kalibry captures non-linear, multi-scale, regime-dependent behaviours**
**that linear models fundamentally cannot.**
This is essential for acceptance by quants.
### **3\. Beating Modern ML**
**⇒ Kalibry is more robust, better calibrated, less overfit, and more stable than neural networks.**
It demonstrates:
*   lower variance
*   better explanation
*   higher institutional trust
*   lower compute cost
*   more consistent performance across regimes
This positions Kalibry as:
**A fundamentally superior reasoning paradigm,**
**not just a better model.**
### **4\. Winning Across All Three Categories**
**⇒ Kalibry becomes the new default AI engine for behavioural time-series.**
This validates:
*   the universal trajectory hypothesis
*   cross-asset generalization
*   robustness across volatility regimes
*   multi-scale behavioural reasoning
*   applicability beyond finance
This is the core of Kalibry’s moat.
**10\. Backtesting Methodology**
Backtesting is the backbone of the Kalibry MVP.
The goal is to establish a **rigorous, leakage-free, scientifically credible** evaluation that demonstrates Kalibry’s robustness across time, regimes, and asset classes.
This chapter defines the **full methodological pipeline**.
## **10.1 Train / Validation / Test Splits (Temporal)**
Financial time-series must be evaluated using **strictly chronological** splits.
Kalibry MVP uses:
### **Training set (2013–2019)**
*   large, multi-regime dataset
*   includes bull, bear, sideways, shocks
*   used to build trajectory embeddings & HPVD structure
### **Validation set (2020–2021)**
*   used for:
    *   hyperparameter tuning
    *   window-length experiments
    *   similarity metrics testing
    *   pooling strategies
    *   calibration analysis
### **Test set (2022–2024)**
*   **completely unseen**
*   contains:
    *   2022 inflation shock
    *   2023 recovery
    *   2024 AI megacap rally
*   final accuracy is measured here
### **Key Rule**
**No data from validation or test periods is ever included in HPVD search for model configuration.**
Training period defines the analog database for model development.
Prediction always uses only **past data** relative to each timestamp.
## **10.2 Walk-Forward Evaluation Protocol**
Static train/test splits are insufficient.
Finance requires **walk-forward**, also known as:
*   rolling out-of-sample forecasting
*   expanding-window backtesting
*   pseudo-real-time testing
### **Procedure**
![](https://t90182117410.p.clickup-attachments.com/t90182117410/063a25fd-e1b7-429b-9433-53cb65318d4a/6ee77b0d-4e43-42ea-a45c-fb0722ad00cb.png)  
###   

### **Why Walk-Forward Matters**
*   simulates real deployment
*   eliminates forward contamination
*   handles non-stationarity
*   captures regime shifts
This is the gold standard for financial time-series validation.
## **10.3 Rolling Parameter Estimation for AR/ARMA/GARCH**
Classical models require **rolling estimation** to be fair.
For each prediction timestamp ttt, we re-fit:
*   AR(p)
*   MA(q)
*   ARMA(p,q)
*   ARIMA
*   GARCH(1,1)
*   AR–GARCH
### **Using only data up to t**
This ensures:
*   identical information constraints
*   proper time-local calibration
*   regime adaptation
### **Typical window:**
*   750 trading days (~3 years)
*   1000 days (optional experiment)
### **Why this is essential**
Without rolling estimation:
*   classical models gain unrealistic hindsight
*   performance is artificially inflated
*   baselines become invalid
Kalibry must beat models that are updated in real time.
## **10.4 Ensuring Fair Comparison Across Models**
Every baseline must be evaluated under identical conditions:
### **Same data**
*   identical OHLCV data
*   identical cleaning
*   identical rolling returns
*   identical temporal boundaries
### **Same targets**
*   H1 direction
*   H5 direction
*   same labeling logic
### **Same timestamps**
*   predictions generated on the **exact same days**
*   forecasts compared against the same ground truth
### **Same preprocessing**
*   rolling normalization identical across models
*   no model allowed look-ahead windowing
### **Same evaluation metrics**
*   Accuracy (H1/H5)
*   Brier Score
*   Entropy
*   Regime-stratified accuracy
### **No unfair advantages**
Neural nets and econometric models must follow the same constraints as Kalibry:
*   no peeking into future segments
*   normalization uses only past window
*   model retraining is real-time rolling
This guarantees scientific credibility.
## **10.5 Avoiding Look-Ahead Bias**
Avoiding leakage is essential.
Kalibry implements strict **anti-leakage controls**:
### **1\. Feature leakage**
Rolling indicators only computed using past data.
### **2\. Target leakage**
Prediction at time ttt only uses trajectories ending at **day t**.
### **3\. Training/test leakage**
The HPVD repository for inference only contains trajectories with timestamps:
![](https://t90182117410.p.clickup-attachments.com/t90182117410/8ba468e5-cd85-48d5-9fa6-8c9238322938/ad49055f-92be-40fc-b31a-2c6fad2c8a50.png)
### **4\. Data revisions**
Use only:
*   finalized historical data
*   no future corrections
*   no hindsight bias
### **5\. Normalization**
Rolling normalization windows strictly capped at day ttt.
### **6\. Model tuning**
Hyperparameters are tuned only on the **validation set**, never test.
### **7\. Cross-asset leakage**
Similarity is allowed across assets, but always enforcing:
![](https://t90182117410.p.clickup-attachments.com/t90182117410/d3c6347e-c72a-48a6-bab1-954a82fbdb05/091174d2-35e9-420d-a216-439e631f160a.png)
This guarantees compliance with industry standards (CFA, NIST, ESMA).
## **10.6 Regime-Specific Evaluation Windows**
Performance must be compared across **market regimes**, because forecasting difficulty varies.
We evaluate separately:
### **1\. Trend-Up Regimes**
*   2017 tech run
*   2020 post-COVID rebound
*   2023 AI boom
### **2\. Trend-Down Regimes**
*   2018 correction
*   2022 inflation shock
*   2022–2023 bear phase
### **3\. High Volatility**
*   2020 COVID crash
*   2022 war + inflation period
### **4\. Low Volatility**
*   2017
*   2023 mid-year
### **5\. Sideways / Driftless Periods**
Kalibry is expected to shine in:
*   volatility clusters
*   regime transitions
*   mixed structural phases
This strengthens the argument for universal trajectory intelligence.
## **10.7 Statistical Robustness Constraints**
Backtesting must meet formal statistical criteria to be considered reliable.
Kalibry enforces the following:
### **1\. Minimum Observation Count**
Every metric must have:
*   ≥ 2000 predictions (H1)
*   ≥ 2000 predictions (H5)
This ensures meaningful statistical power.
### **2\. Confidence Intervals**
We compute:
*   Wilson confidence intervals
*   bootstrap variance
*   drift-adjusted performance
### **3\. Hypothesis Testing**
Perform:
*   t-tests on directional accuracy
*   Kolmogorov–Smirnov tests on return distributions
*   DeLong tests for probabilistic predictions
### **4\. Regime Consistency**
Performance must be stable across:
*   bull phases
*   bear phases
*   sideways phases
*   volatility shifts
### **5\. No Over-Optimization**
To avoid “backtest overfitting”:
*   only 2–3 hyperparameters tuned
*   no grid search on test data
*   no ensemble of models
This ensures the MVP demonstrates **true predictive power**, not accidental overfitting.
# **11\. Evaluation Metrics**
Evaluating Kalibry requires metrics that go beyond simple accuracy.
Because Kalibry outputs **probability distributions**, not deterministic predictions, the evaluation framework focuses on:
*   **directional correctness**
*   **probability calibration**
*   **distributional quality**
*   **regime robustness**
*   **trajectory similarity validity**
*   **forecast stability**
This section defines the full suite of metrics included in the MVP.
## **11.1 Directional Accuracy (1d, 5d)**
Directional accuracy is the primary benchmark used by:
*   hedge funds
*   asset managers
*   academic literature
*   risk teams
It evaluates whether the model correctly predicts the sign of return.
![](https://t90182117410.p.clickup-attachments.com/t90182117410/11f1f06b-c6b6-4ee9-85ab-1cc29153ca65/98fa94b2-a7ef-4bee-a339-1150c5840ea6.png)  
###   

We measure performance:
*   across all assets
*   per asset class
*   per year
*   per regime
**Expected baseline:**
52–55% (Always-Up / Momentum 20d)
**Goal for Kalibry MVP:**
58–62%
(Consistent advantage across regimes)
## **11.2 Brier Score (probability calibration)**
The Brier Score measures the accuracy of predicted probabilities.

![](https://t90182117410.p.clickup-attachments.com/t90182117410/273f7809-c64d-447b-8d18-30965b004089/e5a9495b-e189-42c7-84fe-fe07cb5be1ab.png)
### **Interpretation**
*   Lower score = better
*   Perfect calibration → minimal BS
*   Random walk → BS ≈ 0.25
*   Always-Up → BS ≈ 0.23–0.24
**Kalibry advantage:**
Because Kalibry aggregates empirical outcomes from analog trajectories, its probabilities are naturally calibrated, giving it a structural advantage vs neural networks.
## **11.3 Log-Loss**
Log-Loss heavily penalizes overconfident incorrect predictions.
![](https://t90182117410.p.clickup-attachments.com/t90182117410/32a0d529-abbd-4fb1-920c-b260a649fc03/4b9eaa14-dc1e-420a-bc30-e86b09517265.png)
### **Why Log-Loss Matters**
It measures:
*   probabilistic sharpness
*   risk concentration
*   confidence alignment
*   robustness under high uncertainty
### **Kalibry vs Neural Nets**
Neural nets often output:
*   overconfident
*   poorly calibrated
*   unstable probabilities
Kalibry’s empirical aggregation creates:
*   stable
*   well-calibrated
*   conservative probability curves
We expect a **10–20% improvement** in Log-Loss compared to LSTM/Transformer baselines.
## **11.4 Reliability Curves**
A **reliability diagram** compares:
*   predicted probability
*   actual observed frequency
For bins such as:
*   0–10%
*   10–20%
*   …
*   90–100%
### **Perfect calibration line**
**_y=x_**
Kalibry, by design, should produce:
*   near-perfect alignment
*   low deviation from the diagonal
*   small calibration error per bin
*   smooth transitions across bins
Neural baselines usually show:
*   overconfidence in high bins
*   underconfidence in mid bins
*   chaotic behaviour in low bins
This gives Kalibry a decisive advantage in regulatory and institutional contexts.
## **11.5 Regime Breakdown (Bull, Bear, Sideways, High Volatility)**
Performance must be stratified by:
### **1\. Bull Markets**
*   Persistent upward drift
*   Example: 2017, 2023 AI rally
### **2\. Bear Markets**
*   Sustained downward structure
*   Example: Q2–Q4 2022
### **3\. Sideways / Driftless**
*   2015, parts of 2019
### **4\. High Volatility**
*   2020 COVID collapse
*   Early 2022
For each regime, we compute:
*   H1 accuracy
*   H5 accuracy
*   Brier Score
*   Log-Loss
*   Confidence distribution
*   Cohesion of trajectory families
### **Purpose**
This reveals:
*   regime strengths
*   regime weaknesses
*   whether Kalibry overfits to any phase
*   robustness under stress
Kalibry is expected to shine in:
*   regime transitions
*   high-volatility clusters
*   nonlinear structural breaks
## **11.6 Similarity Quality Diagnostics**
Since Kalibry relies on analog-based reasoning, diagnostics must evaluate whether **the retrieved trajectories are genuinely meaningful**.
We measure:
### **1\. Average Similarity Score**
Mean distance:
​![](https://t90182117410.p.clickup-attachments.com/t90182117410/d4cfa020-67ea-438d-8519-2c2eefc74e47/cca418c9-f46d-4a0b-aa31-9aad61141c9d.png)
Lower = better.
### **2\. Analog Cohesion Score**
Measures similarity _among the neighbors themselves_.
![](https://t90182117410.p.clickup-attachments.com/t90182117410/7c191853-1553-4899-992e-4132a2806349/f83c67d0-e667-482f-b08b-c87ba8c108ca.png)
High cohesion → strong behavioural cluster.
Low cohesion → noisy or ambiguous prediction.
### **3\. Regime Consistency**
Percentage of analogs matching the same regime:​![](https://t90182117410.p.clickup-attachments.com/t90182117410/347a7883-3a15-43ef-903c-e018d39b4afd/a5201f91-4d85-4715-b7e9-13980dd2180b.png)
Example regimes:
*   low-volatility trend
*   high-volatility cluster
*   reversal pattern
*   sideways consolidation
### **4\. Family Diversity**
A diversity score across detected pattern families.
If analogs come from too many families → high entropy.
If they are consistent → strong signal.
### **5\. Similarity vs. Outcome Correlation**
Correlates similarity score with outcome accuracy:
*   strong → high-quality state space
*   weak → requires embedding refinement
This metric is crucial for iterative improvement.
## **11.7 KPI: Stability of Forecasts Over Time**
Institutions require **stable models**.
We measure forecast stability via:
### **1\. Probability Smoothness (1-step Δ)**
![](https://t90182117410.p.clickup-attachments.com/t90182117410/6c23bb00-3717-4333-88e9-b25a7af6aea3/6c115225-ca89-4448-ae13-7842c374ff9a.png)
Lower → more stable model.
### **2\. Entropy Stability**
Entropy volatility:
![](https://t90182117410.p.clickup-attachments.com/t90182117410/f65591eb-6f7c-4984-aa26-a7c6a42aae20/3aad3ef0-be69-4718-8e60-b54177f1120b.png)
Measures stability of uncertainty estimation.
### **3\. Analog Overlap**
Percentage of analogs reused across adjacent days.
If analog sets shift violently → unstable signal.
### **4\. Regime Transition Smoothness**
Detects sudden probability flips that are not justified by behavioural changes.
### **5\. No “Model Mood Swings”**
A common failure of neural nets:
*   probabilities fluctuate wildly
*   confidence jumps
*   unstable under slight input change
Kalibry should demonstrate:
*   smooth evolution
*   consistent family alignment
*   predictable reasoning pathways
This builds trust with:
*   risk teams
*   regulators
*   investment committees
# **12\. Quantitative Expectations (Targets)**
This section outlines the **quantitative benchmarks** Kalibry is expected to achieve across accuracy, calibration, regime robustness, and computational efficiency.
These are not arbitrary numbers—they are grounded in:
*   academic time-series forecasting literature
*   realistic benchmarks from quant funds
*   preliminary internal simulations
*   known limitations of econometrics and ML in finance
*   the structural advantages of trajectory intelligence
The targets define **what success looks like** for the MVP and provide investors with a measurable standard of progress.
## **12.1 Expected Accuracy vs Baselines**
Directional accuracy is evaluated over:
*   **H1 (1-day horizon)**
*   **H5 (5-day horizon)**
The MVP must outperform:
*   naive baselines
*   classical econometric models
*   modern deep-learning models
### **Baseline Performance Ranges (Historical Averages)**

| **Model Class**<br> | **Expected Avg Accuracy**<br> |
| ---| --- |
| Random Walk | 50.0% |
| Always-Up | 52–55% |
| Momentum 20d | 52–55% |
| AR/ARMA | 51–54% |
| ARIMA | 52–54% |
| GARCH / AR-GARCH | 52–56% |
| LSTM / GRU | 53–56% |
| Time-Series Transformer | 53–57% (high variance) |

### **Target Performance for Kalibry MVP**
#### **H1 Accuracy Target**
*   **Lower Bound (Success Threshold): 56.0%**
*   **Target: 58.0–60.0%**
*   **Aspirational (Possible with tuning): 60.5–62.0%**
#### **H5 Accuracy Target**
*   **Lower Bound (Success Threshold): 57.0%**
*   **Target: 59.0–62.0%**
*   **Aspirational: 63.0–65.0%**
### **Interpretation**
Beating naive baselines is necessary.
Beating GARCH and LSTM-class models is decisive evidence of genuine predictive power.
## **12.2 Expected Calibration Improvements**
Kalibry’s empirical distribution aggregation gives it a natural advantage in **calibration**, measured via:
*   **Brier Score**
*   **Log-Loss**
*   **Reliability curves**
### **Expected Brier Score**
*   **Baselines:** 0.22–0.25
*   **Kalibry Target:** **0.17–0.19**
*   **Success Threshold:** ≤ 0.20
### **Expected Log-Loss**
*   **Baselines:** 0.68–0.78
*   **Kalibry Target:** **0.55–0.62**
*   **Success Threshold:** ≤ 0.65
### **Expected Reliability Curve Error**
*   **Baselines:** noticeable deviation, overconfidence
*   **Kalibry Target:**
    *   reliability curve deviation < **3–5%**
    *   near-diagonal shape
    *   stable entropy distribution
This is a major differentiator for institutional adoption and AI Act compliance.
## **12.3 Expected Regime Stability**
Kalibry must demonstrate **consistency** across market conditions:
### **Performance Targets by Regime**

| **Regime**<br> | **Expected Accuracy**<br> | **Minimum Success Threshold**<br> |
| ---| ---| --- |
| Bull | 58–63% | ≥ 56% |
| Bear | 57–62% | ≥ 55% |
| Sideways | 56–60% | ≥ 54% |
| High Volatility | 55–59% | ≥ 53% |

### **Regime Stability KPI**
![](https://t90182117410.p.clickup-attachments.com/t90182117410/5712b0d4-b384-430a-9af2-2576543f80eb/f726696c-f03b-43fe-8b8e-b58f9df05718.png)
This ensures the model:
*   does not collapse in shocks
*   handles volatility clustering
*   generalizes across conditions
*   supports real-world deployment
## **12.4 Expected Computational Performance**
Kalibry is designed to outperform neural networks not only in predictive quality but also in compute efficiency.
### **HPVD (Hybrid Probabilistic Vector Database) Targets:**
*   **Query time:** ≤ 5–15 ms per trajectory (target)
*   **Index size:** millions of trajectories
*   **Memory footprint:** < 8–12 GB typical
*   **Throughput:** > 100 predictions per second per node
### **Comparison**

| **Model Type**<br> | **Inference Cost**<br> | **Training Cost**<br> |
| ---| ---| --- |
| LSTM/GRU | medium | high |
| Transformer | very high | extremely high |
| AR/ARIMA/GARCH | low | medium |
| **Kalibry**<br> | **very low**<br> | **none (no training)**<br> |

### **Why this matters**
*   lower cloud cost
*   faster iteration
*   enables scaling to millions of entities
*   supports real-time or intraday updates
*   major commercial advantage
## **12.5 Thresholds for Declaring MVP Success**
To classify the MVP as **successful**, the following must be true on the 2022–2024 test set:
### **1\. Directional Accuracy Thresholds**
*   **H1 ≥ 56%**
*   **H5 ≥ 57%**
*   (Must beat naive + econometric + ML baselines)
### **2\. Calibration Thresholds**
*   **Brier Score ≤ 0.20**
*   **Log-Loss ≤ 0.65**
*   Reliability curve closely matches diagonal (error < 5%)
### **3\. Regime Robustness Threshold**
*   No regime (bull, bear, sideways, high-volatility) falls below **53%**
*   Accuracy spread between regimes ≤ 6%
![](https://t90182117410.p.clickup-attachments.com/t90182117410/26edc046-9258-4057-b018-930680bd9216/b00919b0-b7c8-4f2f-adee-c66a637c8865.png)
### **5\. Computational Efficiency Threshold**
*   HPVD inference < 30ms
*   End-to-end prediction < 50ms
*   No GPU required
*   Large asset coverage (≥ 100 assets)
### **If all thresholds are met:**
**Kalibry MVP is validated as a superior, data-efficient, explainable alternative to econometrics and deep learning for financial forecasting.**
### **If Kalibry surpasses the targets (likely):**
**Kalibry becomes a new category of AI — “Behavioral Trajectory Intelligence” — with universal applicability beyond finance.**
# **13\. System Architecture**
Kalibry’s system architecture is designed to be:
*   **scalable**
*   **low-latency**
*   **explainable**
*   **modular**
*   **cross-domain applicable**
The architecture separates the system into clean functional layers:
1. Ingestion →
2. State Feature Engine →
3. Embedding Engine →
4. Trajectory Construction →
5. HPVD Similarity Search →
6. PMR-DB Probabilistic Reasoning →
7. Prediction API →
8. Caching →
9. Deployment & orchestration
Together, these layers create a _universal behavioural forecasting engine_.
##   

##   

##   

## **13.1 Overview Diagram**
![](https://t90182117410.p.clickup-attachments.com/t90182117410/4f583668-52bc-458a-b9f1-c345119b2a51/8df903fe-dda5-44d8-94e5-f24c60fcd7ed.png)
## **13.2 Data Ingestion Layer**
This layer collects historical and real-time market data.
### **Data Types**
*   OHLCV (open, high, low, close, volume)
*   corporate actions (splits, dividends)
*   asset identifiers (ticker, ISIN)
*   metadata (sector, exchange, timezone)
### **Main Responsibilities**
*   connect to data provider (Polygon, Finnhub, Tradier, Tiingo, or custom)
*   pull daily OHLCV
*   apply corporate action adjustments
*   gap detection / correction
*   timestamp validation
*   multi-asset alignment
*   schema standardization
### **Output**
Standardized table:
| timestamp | asset | open | high | low | close | volume | metadata… |
Guaranteed consistent across all assets.
## **13.3 Feature & Embedding Computation**
The **State Feature Engine** transforms raw OHLCV into the **45-dimensional engineered feature vector** (defined in Section 5).
### **Responsibilities**
*   compute rolling returns
*   volatility measures
*   momentum
*   liquidity indicators
*   regime indicators
*   multi-scale KPIs (10/20/30/60-day)
*   rolling normalization
*   asset and regime embeddings
### **Outputs**
*   **45-D state vector** (R45)
*   **optional learned latent embedding** (R16–R24)
### **Compute Strategy**
*   runs daily
*   fully vectorized
*   O(N × features) where N = number of assets × days
## **13.4 HPVD (Hybrid Probabilistic Vector Database)**
HPVD is Kalibry’s **high-dimensional similarity search engine**, optimized for trajectory embeddings.
### **Core Capabilities**
*   approximate nearest neighbor search (HNSW / Faiss)
*   custom distance metrics (cosine, Euclidean, learned)
*   timestamp filtering (s < t)
*   regime filtering (volatility, trend type)
*   similarity-weight computation
*   real-time retrieval (5–15 ms per query)
### **Stored Items**
Each item in HPVD consists of:
*   trajectory embedding (R135 pooled or R2700 flattened)
*   asset ID
*   timestamp
*   forward returns (r₁, r₅)
*   precomputed metadata for reasoning
### **Scalability**
*   millions of trajectories
*   CPU-optimized
*   runs without GPUs
*   memory footprint scalable via sharding
HPVD is a major part of Kalibry’s future moat.
## **13.5 PMR-DB (Probabilistic Multimodal Reasoning DB)**
PMR-DB is responsible for:
*   aggregating the outputs of HPVD
*   computing empirical distributions
*   generating calibrated probabilities
*   computing uncertainty measures
*   extracting explanations (Evidence Graphs)
### **Input:**
K nearest analog trajectories from HPVD
### **Output:**
*   P(up) H1
*   P(up) H5
*   entropy
*   analog families
*   similarity statistics
*   narrative explanation
### **Why It’s Important**
This layer transforms **pattern search** into **actionable probabilistic forecasts**.
PMR-DB is the core reasoning module that makes Kalibry explainable and AI Act–aligned.
## **13.6 Trajectory Matching Engine**
The **Trajectory Matching Engine** orchestrates:
1. embedding extraction
2. trajectory construction (60×45)
3. search query generation
4. similarity retrieval
5. analog validation (regime, timestamp, quality)
6. distance-based weighting
### **Workflow Example**
Construct current trajectory → Embed → Query HPVD → Filter valid analogs
→ Compute weights → Pass analogs to PMR-DB → Produce forecast
### **Performance Requirements**
*   10–30 ms end-to-end
*   <50 ms cold-start
*   deterministic for auditability
## **13.7 Prediction API**
This is the external interface for:
*   investors
*   clients
*   internal dashboards
*   analytics teams
*   downstream applications
### **Format**
REST or GraphQL endpoint:
POST /predict
{
"asset": "NVDA",
"timestamp": "2024-02-20"
}
### **Response**
{
"prob\_up\_h1": 0.61,
"prob\_up\_h5": 0.67,
"confidence": "medium",
"entropy": 0.43,
"top\_analogs": \[...\],
"explanation": {...},
"regime": "low\_vol\_uptrend"
}
### **Features**
*   real-time inference
*   includes full explainability
*   optional multi-asset batch inference
*   low-latency caching
## **13.8 Cache & Precomputation Strategy**
Although Kalibry is fast, caching reduces cost even further.
### **Caching Layer Characteristics**
*   short TTL (5–30 seconds)
*   keyed by (asset, timestamp)
*   stores:
    *   last trajectory
    *   last prediction
    *   explanation data
### **Precomputation**
Daily batch jobs precompute:
*   state features
*   embeddings
*   trajectory pooling
*   HPVD indexing updates
This reduces real-time load significantly.
### **Benefits**
*   minimizes CPU usage
*   accelerates API response time
*   supports high-throughput applications
## **13.9 Deployment Considerations**
Kalibry MVP is designed to fit **enterprise AI deployment standards**.
### **Containerization**
*   Docker containers
*   reproducible builds
*   deterministic pipeline
### **Orchestration**
*   Kubernetes (GKE/EKS/DOKS)
*   autoscaling based on:
    *   request load
    *   time-of-day markets
    *   burst traffic (earnings, FOMC)
### **Monitoring**
*   Prometheus metrics
*   Grafana dashboards
*   latency tracking
*   error monitoring (Sentry)
### **Security**
*   secure API keys
*   encrypted storage
*   strict timestamp isolation
*   GDPR & AI Act compliance
### **High Availability**
*   HPVD replicas
*   active/passive failover
*   read-side sharding for vector DB
### **Cost Considerations**
Because Kalibry does not rely on heavy deep learning:
*   no GPUs required
*   predictable CPU-only architecture
*   low inference cost
*   horizontally scalable
**14\. MVP Demo Architecture**
The MVP Demo is not just a visualization layer—it is the _strategic communication interface_ that demonstrates Kalibry’s power, clarity, explainability, and speed.
The goal of the demo is to let investors, analysts, and internal engineers:
*   **select any asset**
*   **view its recent trajectory**
*   **inspect matched historical analogs**
*   **see probability forecasts (H1, H5)**
*   **understand the reasoning behind each prediction**
*   **validate correctness via case studies**
The demo must make Kalibry feel **intelligent, transparent, robust, and inevitable**.
# **14.1 Internal Dashboard Overview**
The dashboard consists of **five core sections**:
1. **Asset Selection Panel**
2. **Trajectory Visualization Panel**
3. **Similarity Match Viewer**
4. **Forecast & Probability Output Panel**
5. **Explainability & Evidence Graph Panel**
### **Layout (Ideal Desktop UI)**
![](https://t90182117410.p.clickup-attachments.com/t90182117410/06fbafbb-f560-4695-bc85-149825cc1119/07d9cee3-0ef6-4831-bcbf-1cf0454b6eac.png)
The interface must feel:
*   fast
*   minimal
*   professional
*   scientific
*   institutional-grade
# **14.2 Asset Selection Interface**
**Purpose:** Allow the user to choose any available asset in the MVP universe.
### **UI Elements**
*   Search bar (ticker / asset name)
*   Asset class filters:
    *   Equities
    *   Indices
    *   FX
    *   Crypto
*   Recently viewed assets
*   Market metadata (market cap, sector, exchange)
### **Interaction Flow**
1. User selects asset (e.g., NVDA).
2. Backend loads the most recent:
    *   60-day trajectory
    *   current embedding
    *   HPVD analog search
    *   forecast outputs
Should load within **< 300ms**.
### **UX Requirement**
**Zero friction.**
Asset switching must feel instant.
# **14.3 Trajectory Visualization & Similarity Matches**
This is the _core demonstration element_.
### **Current Trajectory View (Main Panel)**
A 60-day behavioural trajectory is shown using multi-channel visualization:
*   normalized price
*   volatility curve
*   momentum signal
*   liquidity trend
*   regime shading
*   optionally: PCA projection of the 60×45 matrix into 2D
### **Similarity Match Overlay**
Below or beside the main chart:
For each top-K analog:
*   small thumbnail chart showing the historical 60-day trajectory
*   similarity score (0–1 normalized)
*   asset and date (e.g., “NVDA Mar–May 2017”)
*   regime label (e.g., “Low-Vol Uptrend”)
### **Optional Advanced View**
“Superimposed Trajectory Mode”:
*   overlays the 3 most similar historical trajectories onto the current one
*   aligns them temporally
*   visually demonstrates behavioural analogs
### **Purpose**
To make the investor say:
“Wow, these patterns really DO look the same.”
# **14.4 Probability Outputs**
The forecast panel displays the **final predictions**:
### **For H1 (Next-Day)**
*   P(up): e.g., **0.61**
*   P(down): 0.39
*   Confidence: medium
*   Entropy: 0.43
### **For H5 (5-Day)**
*   P(up): **0.67**
*   P(down): 0.33
*   Confidence: high
*   Entropy: 0.36
### **Visual Elements**
*   probability bars
*   confidence meter
*   risk-neutral / risk-adjusted views
*   regime label (e.g., “Volatility Compression”)
### **UX Objective**
The output should feel like:
*   clear
*   stable
*   actionable
*   mathematically serious
No unnecessary animations, no clutter.
# **14.5 Explainability Panel**
This panel is the **secret weapon** of the demo.
### **Components:**
#### **1\. Evidence Graph**
Shows:
*   the matched trajectories
*   their outcomes (H1/H5)
*   contribution weight
*   aggregated distribution
#### **2\. Feature Similarity Breakdown**
Radar chart or bar chart of:
*   returns similarity
*   volatility similarity
*   momentum similarity
*   liquidity similarity
*   regime alignment
#### **3\. Narrative Summary**
Automatically generated explanation:
“The current trajectory resembles two historical families:
(1) low-volatility drift-up clusters (AAPL 2020),
(2) early acceleration patterns (NVDA 2017).
72% of analog cases resulted in a 5-day positive return.
Similarity is strongest on volatility compression and momentum slope.”
#### **4\. Top Impactful Analogs**
List of the 5 highest-weight analog trajectories, each with:
*   asset
*   date window
*   future outcome
*   similarity weight
### **Purpose**
To prove that Kalibry’s predictions are:
*   grounded
*   transparent
*   explainable
*   empirically justified
# **14.6 Case Studies**
The MVP demo includes a special tab: **Case Studies**.
Each case study demonstrates a moment where Kalibry produced a **high-conviction, high-quality** signal.
### **Case Study Structure**
*   The predicted trajectory
*   The matched analogs
*   The forecast
*   What actually happened
*   Narrative explanation
*   Key insights
### **Case Study Examples**
*   NVDA volatility squeeze → breakout
*   BTC momentum exhaustion → pullback
*   AAPL low-volatility drift regime
*   S&P500 reversal cluster during 2022 bear market
### **Goal**
Case studies let investors _visually confirm_ the system’s predictive power.
# **14.7 UX Targets (Speed, Clarity, Simplicity)**
The demo must prove that Kalibry is:
*   **fast**
*   **clean**
*   **intuitive**
*   **institutional**
*   **scientifically grounded**
### **UX Performance Targets**
#### **1\. Load Time**
*   asset switch: < 300ms
*   prediction rendering: < 100ms
*   explainability: < 150ms
#### **2\. Navigation**
*   no more than 2 clicks to reach any function
*   keyboard-search-enabled asset selector
#### **3\. Visual Clarity**
*   no clutter
*   no overlapping lines
*   clear legend
*   strong color hierarchy
*   consistent layout across all screens
#### **4\. Cognitive Load**
The interface must be designed so that a non-technical investor can understand:
*   the prediction
*   the reasoning
*   the explanation
within **10 seconds**.
### **The Demo Should Feel Like:**
*   Bloomberg Terminal × Explainable AI
*   A scientific instrument, not a trading app
*   A next-generation reasoning tool
#   

#   

#   

#   

# **15\. Results Presentation Framework**
The Results Presentation Framework ensures that Kalibry’s performance is communicated in a way that is:
*   **scientifically credible**
*   **visually compelling**
*   **easy for investors to understand**
*   **aligned with quant and regulatory expectations**
*   **focused on real-world financial value**
It combines charts, tables, case studies, and regime-specific analyses into a cohesive narrative demonstrating Kalibry’s superiority.
# **15.1 How to Visualize Performance for Stakeholders**
Performance visualization must strike a balance between **rigor** and **clarity**.
### **Core Visual Components**
1. **Directional Accuracy Charts (H1 & H5)**
    *   bar chart comparing Kalibry vs baselines
    *   across full test period
    *   across each asset class
2. **Brier Score & Log-Loss**
    *   line chart or bar chart showing calibration superiority
3. **Reliability Diagrams**
    *   diagonal line = perfect calibration
    *   Kalibry curve must hug the diagonal
4. **Prediction Stability Plots**
    *   probability over time
    *   entropy over time
5. **Similarity Cohesion Histograms**
    *   illustrating analog quality
    *   showing Kalibry’s stable cluster formation
### **Principle**
**Show performance in ways that instantly communicate:**
**Kalibry is more accurate, more consistent, and more explainable.**
# **15.2 Best Practices for Highlighting Value**
Investors and analysts interpret results differently.
This section ensures _all_ stakeholders extract maximum value.
### **1\. Start With Simple, High-Level Metrics**
*   H1 accuracy
*   H5 accuracy
*   calibration metrics
These numbers anchor the narrative.
### **2\. Use Visual Comparisons, Not Tables**
For highest impact:
*   bar charts for accuracy
*   error bars for confidence intervals
*   heatmaps for regime performance
Visually communicates strength.
### **3\. Highlight Cross-Asset Generalization**
Show that Kalibry:
*   performs consistently across equities, FX, crypto
*   generalizes better than ML or econometrics
This demonstrates universality.
### **4\. Show Explainability, Not Just Performance**
Include Evidence Graph examples:
*   analogs
*   pattern families
*   aggregated probabilities
*   narrative rationale
This is a **major competitive advantage**.
### **5\. Emphasize Stability**
Investors trust models that are consistent.
Visualize:
*   daily probability smoothness
*   analog cohesion
*   entropy distributions
Stability is key to institutional adoption.
# **15.3 Case Study Format**
Case studies convert abstract performance into **real-world understanding**.
Use a standardized format:
### **Case Study Structure**
#### **1\. Asset + timeframe**
Example: **NVDA — Feb–Apr 2024**
#### **2\. What happened historically**
Short summary:
“NVDA entered a low-volatility compression phase before a major breakout.”
#### **3\. Kalibry’s trajectory match**
Show:
*   current 60-day trajectory
*   top analogs
*   similarity scores
#### **4\. Kalibry’s forecast**
Show P(up) for H1, H5.
#### **5\. What actually happened**
Overlay actual future movement.
#### **6\. Explanation summary**
Auto-generated narrative:
“The analog set was dominated by low-volatility drift and breakout clusters,
which historically led to 5-day upside in 72% of cases.”
#### **7\. Quantitative takeaway**
E.g., “Kalibry predicted correctly with 0.67 probability.”
### **Why This Format Works**
*   shows scientific reasoning
*   proves real predictive power
*   builds trust via transparency
*   makes results understandable
# **15.4 Regime-Specific Examples**
Kalibry must demonstrate robustness across regimes.
### **Recommended Regime Case Studies**
1. **Low-volatility uptrend** (e.g., AAPL 2020)
2. **High-volatility cluster** (e.g., BTC early 2021)
3. **Bear-market reversal** (e.g., S&P500 2022)
4. **Sideways with false breakouts**
For each regime:
*   show prediction accuracy
*   show analog qualities
*   highlight families of similar patterns
*   explain the behavioural signature Kalibry detected
### **Presentation Objective**
**Prove that Kalibry is not a model for one regime.**
**It is a universal behavioural prediction engine.**
# **15.5 Comparison Charts**
Stakeholders must immediately see Kalibry outperform baselines.
### **Recommended Comparison Visuals**
#### **1\. Full-Spectrum Baseline Comparison**
A single chart showing:
*   RW
*   Always-Up
*   Momentum 20d
*   AR
*   ARMA
*   GARCH
*   LSTM
*   GRU
*   Transformer
*   **Kalibry**
For both:
*   H1 accuracy
*   H5 accuracy
### **2\. Calibration Comparison**
*   Brier bar chart
*   Log-Loss bar chart
*   Reliability curve overlay
### **3\. Regime Comparison Chart**
A 4×2 matrix:
*   columns: H1, H5
*   rows: regimes
    *   bull
    *   bear
    *   sideways
    *   high-vol
### **4\. Stability Index Chart**
Plotting:
*   probability smoothness
*   entropy stability
*   analog cohesion
A single multi-metric figure proves model reliability.
# **Conclusion of Section 15**
The Results Presentation Framework ensures:
*   **clarity for investors**
*   **rigor for quantitative analysts**
*   **compliance alignment for regulators**
*   **actionability for financial practitioners**
It turns Kalibry’s internal performance into a **high-impact, persuasive, evidence-based narrative** that drives investment and adoption.
**16\. Compliance & Risk**
Kalibry operates in a domain—financial asset forecasting—where regulatory scrutiny, fiduciary responsibility, and risk controls are paramount.
This section defines the compliance framework and the boundaries of Kalibry’s MVP to ensure safe use, regulatory alignment, and appropriate risk expectations.
## **16.1 Not Investment Advice Disclaimer**
Kalibry is **not** an investment advisory system.
It does not:
*   provide buy/sell/hold recommendations
*   generate trading signals
*   deliver personalized financial guidance
*   optimize portfolios
*   target individual investor circumstances
### **Formal Disclaimer Language**
**Kalibry outputs are for research, analysis, and informational purposes only.**
**They do not constitute investment advice, trading recommendations, or financial guidance.**
**All investment decisions must be made independently and at the user’s own risk.**
### **Why this matters**
*   Ensures compliance with SEC, ESMA, FCA, and global regulations.
*   Protects Kalibry from misinterpretation as an advisory product.
*   Reinforces that the MVP outputs **directional probabilities**, not actions.
## **16.2 No Price Targets – Only Probabilistic Evolution**
Kalibry does **not** attempt to forecast:
*   price levels
*   target prices
*   long-term valuations
*   analyst-style projections
Instead, Kalibry provides:
*   **probabilities** (Up/Down)
*   **short-horizon dynamics** (1-day, 5-day)
*   **empirical trajectory outcomes**
*   **distributional forecasts**
### **Why this boundary is critical**
*   Price targets imply a level of certainty incompatible with scientific forecasting.
*   Price-level prediction models often encourage inappropriate market speculation.
*   Regulators explicitly warn against deterministic predictions in financial AI.
### **Kalibry’s stance**
**Kalibry predicts behavioural evolution, not price targets.**
This is safe, responsible, and regulator-aligned.
## **16.3 Evidence-Based Pattern Explanation**
Kalibry predictions are always accompanied by:
*   matched historical analogs
*   similarity reasoning
*   regime context
*   empirical outcome distributions
*   full explainability
### **Key Compliance Benefits**
1. **Traceability:**
2. Every prediction can be traced back to _actual historical events_.
3. **Accountability:**
4. Analysts can understand why a probability was produced.
5. **Auditability:**
6. Compliance teams can inspect:
    *   input trajectory
    *   matched analogs
    *   evidence graphs
    *   uncertainty metrics
7. **Human-in-the-loop:**
8. Kalibry augments human judgment, never replaces it.
This satisfies the core requirements of the EU AI Act and major financial regulators.
## **16.4 AI Act Alignment (Explainability, Safety, Traceability)**
The EU AI Act classifies finance-related AI systems as “high-risk,” requiring:
### **✔ Explainability**
Kalibry provides:
*   Evidence Graphs
*   Family pattern clustering
*   Similarity heatmaps
*   Narrative reasoning summaries
### **✔ Traceability**
Every prediction includes:
*   timestamp
*   asset ID
*   trajectory ID
*   analog list
*   future outcomes used for aggregation
### **✔ Safety**
Kalibry ensures:
*   no unauthorized advice
*   no black-box decisioning
*   uncertainty quantification
*   robust fallback behaviours
### **✔ Human Oversight**
Kalibry outputs are designed for:
*   analysts
*   risk teams
*   researchers
*   compliance officers
humans remain the decision-makers.
### **✔ Bias Prevention**
Kalibry compares **behavioural patterns**, not demographic data.
Zero risk of protected-attribute bias.
### **✔ Data Governance**
All ingestion follows:
*   GDPR
*   secure storage
*   no personal data
*   transparent processing
### **Kalibry is natively AI Act–compliant by design.**
## **16.5 Limitations and Boundary Conditions**
Kalibry is **not** a magic crystal ball.
It is a behavioural trajectory forecasting system with defined limits.
### **1\. Short-Term Horizons Only**
Forecasts are restricted to:
*   1-day
*   5-day
Longer horizons are:
*   less predictable
*   less stable
*   not scientifically defensible
### **2\. Sensitive to Regime Transitions**
Kalibry is robust, but:
*   sudden geopolitical shocks
*   unpriced macro events
*   black swans
can reduce accuracy temporarily.
Uncertainty forecasts reflect this via entropy spikes.
### **3\. Dependent on Quality of Historical Analogs**
If few analogs exist:
*   similarity scores drop
*   uncertainty increases
*   probabilities flatten toward 0.5
Kalibry **never** fabricates certainty.
### **4\. Not Suitable for High-Frequency Trading**
System is built for **daily resolution**, not intraday.
This aligns with:
*   data availability
*   behavioural window design
*   compliance boundaries
### **5\. No Guarantee of Profitability**
Quantitatively strong models do **not** guarantee:
*   profits
*   alpha
*   risk-adjusted performance
They are tools for probability forecasting, not trading advice.
### **6\. Not Immune to Market Regime Overflows**
Extreme conditions may degrade performance:
*   flash crashes
*   structural breaks
*   sudden liquidity collapses
Kalibry signals high entropy during such events.
# **Conclusion of Section 16**
This compliance & risk framework ensures Kalibry:
*   is safe to deploy
*   meets global regulatory expectations
*   maintains clear ethical boundaries
*   communicates risk transparently
*   supports investor trust
*   prevents misuse
*   demonstrates responsible innovation
Kalibry is not an advisory tool—
it is a universal behavioural engine that outputs **probabilities**, not instructions.
# **17\. MVP Deliverables**
The MVP aims to produce a complete, verifiable, and investor-ready demonstration of Kalibry’s predictive power, explainability, and system maturity.
Deliverables are grouped into five categories:
1. **Technical Core Deliverables**
2. **Demo & Visualization Layer**
3. **Documentation & Compliance Packages**
4. **Performance & Benchmarking Reports**
5. **Case Studies & Interpretability Evidence**
This ensures that the MVP is not a conceptual prototype, but a **fully operational, measurable, evidence-based system**.
## **17.1 Technical Deliverables**
These are the core engineering outputs that constitute the functional system.
### **1\. Fully Working Forecast Engine**
*   Feature extraction module (45 engineered features)
*   Embedding architecture (R45 + optional latent R16–R24)
*   60×45 trajectory builder
*   HPVD similarity search engine
*   PMR-DB empirical reasoning module
*   Probabilistic forecasting output (H1 & H5)
*   Calibration & uncertainty metrics
### **2\. High-Performance Vector Database**
*   Million-trajectory index
*   ANN search (HNSW / Faiss)
*   regime-filtered search capability
*   timestamp-safe retrieval (no leakage)
### **3\. Prediction API**
*   REST/GraphQL endpoint
*   <50 ms response time (target)
*   includes analogs, probabilities, entropy, explanations
*   secure key authentication
*   logging & monitoring
### **4\. Batch Processing Pipeline**
*   nightly ingestion of OHLCV
*   recomputation of features and embeddings
*   continuous HPVD index update
*   automatic data quality checks
### **5\. Monitoring & Stability Tools**
*   system health metrics
*   latency dashboard
*   analog cohesion tracker
*   calibration monitoring (Brier/log-loss drift)
### **6\. Deployment Package**
*   Docker images
*   Kubernetes configuration
*   CI/CD pipeline
*   logging & alerting integrations
**Outcome:**
A production-ready MVP system that investors and internal engineers can inspect, test, and validate.
## **17.2 Demo Deliverables**
The demo is the **public-facing embodiment** of the MVP.
### **1\. Interactive Dashboard (Internal Version)**
Includes:
*   asset selector
*   trajectory visualization
*   analog matches viewer
*   probability panel
*   evidence graph
*   full explainability
*   regime context
### **2\. Performance Explorer**
Allows investors to:
*   view H1/H5 accuracy
*   filter by asset category
*   filter by regime
*   explore calibration graphs
*   compare against baselines
### **3\. Case Study Viewer**
For showing:
*   correct predictions
*   high-confidence signals
*   structural advantages
*   analog family behaviour
This is essential for pitching to financial institutions.
### **4\. Live Prediction Mode (Optional Stretch Goal)**
*   shows daily fresh probabilities
*   supports up to 100+ assets
*   auto-refresh capability
### **5\. Presentation-Ready Screenshots**
Packaged for pitch decks and investor updates.
## **17.3 Documentation Packages**
Comprehensive documentation is required for:
*   investors
*   quants
*   engineers
*   compliance teams
*   regulators
### **Included Documentation**
### **1\. Technical Whitepaper**
Contains:
*   methodology
*   mathematical formulations
*   similarity metrics
*   embedding architecture
*   calibration logic
*   backtesting methodology
### **2\. API Documentation**
*   full endpoint reference
*   request/response schemas
*   code examples (Python, JS, cURL)
*   authentication instructions
### **3\. System Design Document**
*   full architecture diagrams
*   data flow
*   component interactions
*   performance constraints
### **4\. Compliance Package**
*   AI Act alignment
*   not-advice disclaimer
*   traceability design
*   risk mitigation strategies
*   system limitations
### **5\. Testing & QA Documentation**
*   unit tests
*   integration tests
*   reproducibility guidelines
*   failure mode analyses
## **17.4 Performance Reports**
These documents **prove** the system works.
### **1\. Baseline Benchmark Report**
Comparisons versus:
*   Random Walk
*   Always-Up
*   Momentum 20d
*   AR / ARMA / ARIMA
*   GARCH / AR-GARCH
*   LSTM
*   GRU
*   Transformer
Includes:
*   accuracy
*   Brier Score
*   log-loss
*   regime-stratified performance
### **2\. Calibration Report**
*   reliability curve
*   bin-wise calibration error
*   entropy distribution
*   analogue-based calibration stability
### **3\. Regime Stability Report**
*   bull
*   bear
*   sideways
*   high volatility
*   inflation shocks
*   tech-momentum runs
### **4\. Probability Behavior Report**
*   smoothness metric
*   analog cohort cohesion
*   uncertainty volatility
### **5\. Statistical Robustness Report**
*   confidence intervals
*   bootstrap tests
*   significance assessments
*   hypothesis testing results
### **6\. Infrastructure Performance Report**
*   latency
*   throughput
*   memory usage
*   stability under load
## **17.5 Case Study Pack**
A curated set of **real-world examples** demonstrating Kalibry’s reasoning and predictive power.
### **Case Study Components**
1. Asset + date window
2. Current trajectory
3. Analog matches
4. Similarity scores
5. Empirical outcome distributions
6. Forecast (H1, H5)
7. What actually happened
8. Explanation summary
9. Performance takeaway
### **Recommended Case Study Types**
*   Volatility compression → breakout
*   Momentum exhaustion → reversal
*   Trend continuation under low vol
*   Bear-market counter-trend patterns
*   Sideways drift patterns
### **Format Options**
*   PDF binder (10–25 cases)
*   Slide deck (investor-ready)
*   Interactive “Play mode” inside the demo
# **Conclusion of Section 17**
The MVP deliverables ensure Kalibry launches with:
*   a **fully working predictive engine**
*   a **high-impact investor demo**
*   complete **explainability & compliance frameworks**
*   rigorous **performance validation**
*   persuasive **case studies**
*   strong **technical documentation**
This is a production-quality MVP that demonstrates both **scientific legitimacy** and **commercial viability**.
# **18\. Timeline & Resource Plan**
This section defines the execution plan for Kalibry’s development from:
*   **MVP (0–10 weeks)**
*   **v1 (10–24 weeks)**
*   **v2 (6–12 months)**
It also outlines team responsibilities, tasks, research milestones, evaluation procedures, and internal validation workflows.
# **18.1 Roadmap (MVP → v1 → v2)**
## **Phase 1 — MVP (0–10 weeks)**
**Goal:** Build the foundational predictive engine + investor demo.
### **Deliverables:**
*   R45 embedding engine
*   60×45 trajectory builder
*   HPVD indexing engine
*   PMR-DB probabilistic aggregator
*   H1 & H5 prediction pipeline
*   Full performance benchmark vs baselines
*   Explainability system (Evidence Graphs)
*   Interactive MVP dashboard
## **Phase 2 — v1 (10–24 weeks)**
**Goal:** Productionize the engine and generalize it beyond the MVP asset set.
### **Enhancements:**
*   multi-asset scalability (5k+ assets)
*   improved learned embedding (R24 latent layer)
*   smarter regime detection
*   intraday expansion (optional)
*   clustering of analog families
*   batch forecasting API
*   integration with external systems (Bloomberg API, internal enterprise tools)
*   regulatory documentation refinement
## **Phase 3 — v2 (6–12 months)**
**Goal:** Move from a financial predictor to a **universal trajectory intelligence engine**.
### **Capabilities:**
*   multimodal integration
    *   text + market data
    *   news sentiment embeddings
*   cross-vertical generalization
    *   supply chain prediction
    *   demand forecasting
    *   predictive maintenance
    *   biological/health trajectories
*   PMR-DB multi-domain architecture
*   commercial productization
*   enterprise-grade scalability
# **18.2 Engineering Tasks**
This section describes all engineering tasks required for the MVP and early versions.
## **Data Layer Engineering**
*   Connect to OHLCV data providers
*   Implement download + storage layers
*   Corporate action adjustments
*   Missing data handling & anomaly detection
*   Multi-asset alignment
*   Historical depth normalization (8–10 years)
**Output:** Cleaned, validated time-series dataset.
## **Feature Engineering & Embedding**
*   Implement 45 engineered features
*   Rolling normalization & multi-scale KPIs
*   Regime indicator computation
*   Optional: train learned embeddings (R16–24)
*   Export embedding vectors
**Output:** Consistent R45 embedding pipeline.
## **Trajectory Builder**
*   Construct 60×45 matrices
*   Windowing with leakage protection
*   Trajectory metadata (asset, timestamp, regime)
*   PCA dimensionality reduction (optional)
**Output:** Compressed trajectory dataset.
## **HPVD Engineering**
*   Setup vector database (Faiss or HNSW)
*   Index trajectory embeddings
*   Implement custom similarity metrics
*   Regime-filtered search
*   Timestamp-safe search (t < now)
*   ANN performance optimization
**Output:** High-throughput trajectory retrieval engine.
## **PMR-DB Engineering**
*   empirical outcome distribution computation
*   analog weighting logic
*   calibration layer (Brier/Platt scaling)
*   entropy + uncertainty metrics
*   explainability module integration
**Output:** Complete probabilistic reasoning engine.
## **Prediction API**
*   REST/GraphQL endpoints
*   batching support
*   authentication and logging
*   low-latency high-availability design
*   test suite + integration checks
**Output:** External interface for predictions.
## **MVP Dashboard**
*   frontend asset selection
*   real-time visualizations
*   similarity match thumbnails
*   probability panels
*   evidence graph views
*   regime classification visualization
*   case study explorer
**Output:** High-impact investor demo.
# **18.3 Research Tasks**
These tasks ensure Kalibry’s trajectory intelligence is grounded in scientific rigor.
## **1\. Feature Validation Research**
*   test each feature’s predictive relevance
*   perform cross-asset robustness tests
*   investigate PCA variance capture
**Goal:** ensure R45 is optimal for short-term dynamics.
## **2\. Embedding Optimization Research**
*   test embedding dimension alternatives (R30–R120)
*   test learned vs engineered embeddings
*   test temporal pooling strategies
**Goal:** maximize similarity fidelity.
## **3\. Similarity Metric Research**
*   compare cosine, Euclidean, correlation, DTW
*   test learned metrics (Siamese MLP)
*   optimize for stability + interpretable behaviour
**Goal:** define the “best” trajectory comparator.
## **4\. Regime Research**
*   identify volatility signatures
*   test drift vs compression regimes
*   evaluate regime-specific accuracy
**Goal:** improve inference consistency and explainability.
## **5\. Calibration Research**
*   reliability curve analysis
*   entropy stability testing
*   Platt scaling vs isotonic regression
**Goal:** world-class probability calibration.
## **6\. Baseline Benchmarking**
*   LSTM
*   GRU
*   Transformer
*   AR / MA / ARMA / ARIMA
*   GARCH / AR-GARCH
*   Momentum models
*   Random Walk
**Goal:** demonstrate quantitative dominance > baseline.
# **18.4 Evaluation & Stress Testing**
The MVP requires robust evaluation to establish trust.
## **Backtesting Protocols**
*   walk-forward evaluation
*   sliding-window recalibration
*   training/validation/test split
*   regime-stratified backtesting
## **Stress Testing**
*   flash crash simulation
*   volatility shock testing
*   missing data simulation
*   index-level regime flip tests
*   cross-asset contamination checks
## **Model Stability Testing**
*   smoothness of probability curves
*   analog cohesion variance
*   entropy trend monitoring
## **Computational Stress Testing**
*   load testing of the prediction API
*   vector DB throughput benchmarking
*   processing cost per query
*   scaling vs asset count
## **Security & Reliability Checks**
*   compliance review
*   audit trail consistency
*   reproducibility tests
*   deterministic prediction validation
# **18.5 Internal Review**
A structured review process ensures that the MVP is investor-ready.
## **1\. Architecture Review**
Participants:
*   Lead engineer
*   ML lead
*   CTO
*   Compliance officer
Focus:
*   correctness
*   stability
*   modularity
## **2\. Quantitative Validation Review**
Participants:
*   Head of Quant
*   Research analysts
*   Technical advisors
Focus:
*   statistical robustness
*   baseline superiority
*   calibration quality
## **3\. Explainability & AI Act Review**
Participants:
*   compliance team
*   regulatory advisors
*   explainability lead
Focus:
*   traceability
*   reasoning clarity
*   documentation completeness
## **4\. Security & Reliability Review**
Participants:
*   devops
*   infrastructure team
Focus:
*   API reliability
*   HPVD resilience
*   monitoring & alerting
## **5\. Investor Demo Review**
Participants:
*   product
*   design
*   founders
*   investors (optional preview sessions)
Focus:
*   clarity
*   visual impact
*   narrative coherence
*   ease of interpretation
# **Conclusion of Section 18**
This timeline & resource plan ensures that Kalibry’s MVP:
*   is built efficiently
*   is validated scientifically
*   delivers evidence-based predictive value
*   complies with regulatory expectations
*   produces a powerful investor-facing demo
*   establishes a scalable path to v1 and v2
This section signals execution readiness and operational maturity to technical and financial stakeholders.
# **19\. Strategic Impact**
Kalibry’s trajectory intelligence framework is designed to model **any evolving entity**, from markets to machines to biological systems.
The financial MVP is not an end in itself—it is the **first validation point** of a universal predictive engine.
This section explains the strategic advantages of starting in finance, the vertical expansion roadmap, and the long-term implications for enterprise adoption and technological leadership.
# **19.1 Why Financial Markets Validate Kalibry’s Core Technology**
Financial markets are one of the **harshest, most data-rich, statistically unforgiving** environments in the world.
If a forecasting system is valid there, it is likely valid anywhere.
### **Core Reasons Finance Is the Ideal Testing Ground**
### **1\. Extreme Noise + High Volatility**
Markets are:
*   noisy
*   non-linear
*   regime-switching
*   adversarial
*   reflexive
If Kalibry can identify stable patterns in markets, it demonstrates **unusual robustness**.
### **2\. Deep Historical Data**
Millions of trajectories spanning:
*   equities
*   FX
*   indices
*   commodities
*   crypto
This provides the perfect playground to validate:
*   analog matching
*   empirical distribution reasoning
*   uncertainty quantification
*   explainability
### **3\. Instant Feedback Loops**
Markets generate **real outcomes every day**, allowing Kalibry to be:
*   rapidly evaluated
*   continuously calibrated
*   instantly stress-tested across multiple regimes
### **4\. Extreme Diversity of Behaviors**
Assets differ by:
*   volatility
*   liquidity
*   trend structure
*   microstructure
*   sector dynamics
Success here proves the system can handle **heterogeneous evolving entities**.
### **5\. Strict Regulatory Environment**
By designing Kalibry to meet:
*   AI Act
*   ESMA guidelines
*   SEC guidance
…from day one, the system is built **future-proof** for regulated industries.
**Finance is the perfect crucible—**
**If Kalibry works here, it works anywhere.**
# **19.2 Expansion into Other Verticals**
Once validated in finance, Kalibry becomes a **horizontal technology** that applies to any evolving system.
Below are the four high-priority target verticals.
## **1\. Industry & Predictive Maintenance**
Industrial systems generate complex signals:
*   vibration
*   temperature
*   pressure
*   operational cycles
Kalibry can:
*   detect degradation trajectories
*   identify early failure analogs
*   forecast downtime probabilities
*   lower maintenance costs
**Outcome:** Predictive maintenance becomes explainable and empirically grounded.
## **2\. eCommerce & Customer Behaviour**
Customer, product, and demand patterns evolve over time.
Kalibry can predict:
*   purchasing trajectories
*   churn risk
*   revenue run rates
*   high-value customer patterns
*   inventory risks
This transforms eCommerce from reactive to **predictive**.
## **3\. Healthcare & Patient Evolution**
Medical signals such as:
*   biometrics
*   lab values
*   symptom trajectories
…are fundamentally temporal processes.
Kalibry can identify:
*   early warning patterns
*   treatment response trajectories
*   risk states and transitions
**Outcome:** Clinical decision support systems with _transparent reasoning_.
_Fully AI Act-aligned._
## **4\. Enterprise Risk & Insurance**
Risk evolves through:
*   operational incidents
*   supply chain instability
*   climate exposure
*   asset volatility
Kalibry enables:
*   trajectory-based risk scoring
*   explainable risk models
*   early detection of high-risk scenarios
This is highly valuable for insurers and large enterprises.
### **Vertical Expansion Summary**

| **Vertical**<br> | **Value**<br> | **Why Kalibry Fits**<br> |
| ---| ---| --- |
| Finance | forecasting, explainability | high-frequency validation |
| Industry | predictive maintenance | trajectory degradation |
| eCommerce | user & product behaviour | rich multivariate signals |
| Healthcare | patient evolution | regulated explainability |
| Risk | systemic risk prediction | pattern-based similarity |

# **\*\*19.3 Long-Term Vision:**
The Universal Engine for Evolving Entities\*\*
Kalibry’s long-term strategic vision is to become:
**The world’s first universal behavioral prediction engine.**
This engine works by:
1. **Embedding states** (R45 / R24 latent)
2. **Building trajectories** (60×45)
3. **Finding analogs** in historical data
4. **Aggregating empirical outcomes**
5. **Producing probabilistic forecasts**
6. **Explaining them with evidence**
This approach is:
*   model-agnostic
*   domain-agnostic
*   fully explainable
*   scalable
*   safe
*   compliant
### **Universal entity types Kalibry can model**
*   financial assets
*   customers
*   machines
*   organisms
*   supply chains
*   weather patterns
*   credit risk profiles
*   security threats
Any system that evolves through time can be formalized as:
**A sequence of embedded states** →
**forming a trajectory** →
**with historical analog families** →
**leading to learnable evolution probabilities.**
This is the core insight.
# **19.4 How the Finance MVP Accelerates Adoption & Trust**
The financial MVP is strategically designed as the **fastest path to adoption** and **strongest demonstration of credibility**.
### **1\. Immediate Quantitative Validation**
Investors get:
*   accuracy metrics
*   calibration graphs
*   regime-specific performance
*   analog explainability
This instantly builds trust.
### **2\. Strong Technical Moat**
If Kalibry can:
*   outperform classical econometrics
*   outperform LSTM/Transformer models
*   provide full explainability
…then adoption becomes inevitable.
### **3\. Regulatory Alignment from Day One**
By building a finance-ready, AI Act-compliant MVP:
*   enterprise buyers trust the system
*   healthcare & industrial clients see safety-first design
*   regulators can audit the logic
This removes barriers to scaling.
### **4\. High-Value Case Studies as Sales Assets**
The MVP produces:
*   real predictions
*   real reasoning
*   real outcomes
*   real evidence
A case study package transforms Kalibry from a concept → into a **validated technology**.
### **5\. Investors Trust What They Can Measure**
Markets allow instant feedback.
Positive results create a **self-validating momentum loop**:
1. Daily predictions
2. Daily outcomes
3. Daily evidence
4. Daily quantifiable value
This drives investor confidence faster than any other vertical.
# **Conclusion of Section 19**
Finance is not just the first vertical—it is the **strategic accelerator** that proves Kalibry’s trajectory intelligence in the most hostile, regulated, data-intensive environment.
Once validated in finance, Kalibry expands naturally into:
*   industry
*   eCommerce
*   healthcare
*   enterprise risk
becoming a **universal engine for predicting the evolution of any system**.
#   

#   

#   

#   

#   

#   

#   

# **20\. Appendix**
The appendix consolidates all technical artifacts, mathematical definitions, specifications, and compliance materials required to support scientific validation, investor due diligence, and internal engineering reproducibility.
# **20.1 Full 45-Dimensional Feature Specification**
The engineered 45-dimensional feature vector captures **price dynamics**, **volatility**, **momentum**, **liquidity**, and **regime structure** across multiple horizons.
Each feature is computed daily, rolling window unless noted otherwise.
## **(A) Returns & Trend Indicators — 12 features**

| **Feature**<br> | **Description**<br> |
| ---| --- |
| R1 | 1-day log return |
| R2 | 5-day log return |
| R3 | 10-day log return |
| R4 | 20-day log return |
| R5 | 30-day log return |
| R6 | 60-day log return |
| R7 | Rolling 10-day trend slope (linear regression) |
| R8 | Rolling 20-day trend slope |
| R9 | Rolling 30-day trend slope |
| R10 | Trend intensity ratio (abs(slope)/volatility) |
| R11 | Higher-order trend curvature (2nd derivative) |
| R12 | Drift indicator (sign of long-term slope) |

## **(B) Volatility Structure — 10 features**

| **Feature**<br> | **Description**<br> |
| ---| --- |
| V1 | 5-day realized volatility |
| V2 | 10-day realized volatility |
| V3 | 20-day realized volatility |
| V4 | 30-day realized volatility |
| V5 | 60-day realized volatility |
| V6 | Volatility compression ratio (10d/30d) |
| V7 | Volatility expansion ratio (20d/60d) |
| V8 | GARCH(1,1) conditional variance estimate |
| V9 | Volatility asymmetry (up- vs down-move variance) |
| V10 | Volatility of volatility (rolling std of V1–V5) |

## **(C) Momentum & Mean Reversion — 8 features**

| **Feature**<br> | **Description**<br> |
| ---| --- |
| M1 | 10-day momentum (close / SMA10 – 1) |
| M2 | 20-day momentum |
| M3 | 30-day momentum |
| M4 | RSI-14 |
| M5 | Z-score of price relative to 20-day mean |
| M6 | Distance from 10-day SMA |
| M7 | Distance from 30-day SMA |
| M8 | Mean-reversion strength (price vs Bollinger midline) |

##   

## **(D) Liquidity & Market Microstructure — 7 features**

| **Feature**<br> | **Description**<br> |
| ---| --- |
| L1 | 5-day volume Z-score |
| L2 | 20-day volume Z-score |
| L3 | Volume/volatility ratio |
| L4 | Turnover (volume × close) |
| L5 | Rolling Amihud illiquidity proxy |
| L6 | Range/close ratio (HL/Close) |
| L7 | ATR(14) normalized by closing price |

## **(E) Regime Indicators — 5 features**
(All binary/continuous)

| **Feature**<br> | **Description**<br> |
| ---| --- |
| G1 | Trend regime (up/down/flat encoded) |
| G2 | Volatility regime (compressing/expanding) |
| G3 | Liquidity regime (high/low) |
| G4 | Market phase (risk-on/risk-off encoded using SPX/VIX cross-regime) |
| G5 | Price location vs 60-day channel (0–1 normalized position) |

## **(F) Asset Identity Features — 3 features**

| **Feature**<br> | **Description**<br> |
| ---| --- |
| A1 | Volatility bucket (low/medium/high; 1-hot or ordinal) |
| A2 | Sector/asset class embedding |
| A3 | Cyclicality factor (β vs benchmark over 1yr) |

Total = **45 engineered features**
High-dimensional enough to capture structure but compact enough to remain interpretable.
# **20.2 Embedding Dimension Experiment Setup**
Purpose: determine the optimal embedding dimension for similarity search.
## **Dimensions tested**
*   R30
*   R45
*   R60
*   R90
*   R120
*   Learned latent embeddings: R16, R24, R32
## **Procedure**
### **1\. Build embeddings**
For each dimension:
*   compute or project features
*   normalize (z-score / rolling normalization)
### **2\. Produce 60-day trajectories**
*   create 60×D matrices
*   flatten or pool (max/mean/pca-pooled)
### **3\. Evaluate similarity fidelity**
Metrics:
*   analog cohesion
*   intra-cluster variance
*   retrieval stability
*   calibration correlation
*   probability smoothness
### **4\. Evaluate model performance**
*   directional accuracy (H1, H5)
*   Brier score
*   log-loss
*   entropy distribution
### **5\. Compute cost-performance ratio**
*   vector DB memory
*   query latency
*   indexing time
## **Expected Outcome**
R45 or R60 ≈ best tradeoff.
R24 latent ≈ best learned embedding.
# **20.3 Window-Length Experiment Setup**
Goal: determine optimal trajectory length (30, 60, 90 days).
## **Tested windows**
*   30×45
*   60×45
*   90×45
## **Metrics**
*   forecasting accuracy
*   calibration stability
*   analog diversity
*   overfitting risk
*   runtime and memory scaling
## **Experimental design**
*   keep target horizon fixed (H1, H5)
*   vary only window length
*   ensure no leakage
*   evaluate regime-specific performance
## **Expected results**
*   **60 days** = best signal/noise balance
*   30 days = too noisy, low coherence
*   90 days = higher cost and unnecessary noise accrual
# **20.4 Mathematical Definitions of All Metrics**
Below are the formal definitions used in evaluation.
## **Directional Accuracy**
![](https://t90182117410.p.clickup-attachments.com/t90182117410/7babcec1-1d92-4f17-9a7a-aa33815ed064/27ea19c8-f748-463b-b245-16a97b29a16a.png)
## **Brier Score**
![](https://t90182117410.p.clickup-attachments.com/t90182117410/f3dfbd38-364d-409d-a499-22491e2f7503/b6b91b8b-4fb7-4e22-9555-eeb06b24c691.png)
## **Log-Loss**
![](https://t90182117410.p.clickup-attachments.com/t90182117410/643fc2da-8dd6-4792-80eb-4ead21bd1ba1/12a17568-eaa7-4914-bc62-fd432c4126ba.png)
## **Entropy**
![](https://t90182117410.p.clickup-attachments.com/t90182117410/3a29ae6a-e501-4706-82ee-5875dc5a9fff/95d161fa-2c54-437e-8669-32c1b74d116e.png)
## **Calibration Curve**
Bin predictions into buckets and compute:
![](https://t90182117410.p.clickup-attachments.com/t90182117410/c814a8c1-bda0-4b43-bd8a-03b5dcc68a28/292f96b8-3974-4456-bb1e-a520692609ea.png)
## **Analog Cohesion Index**
![](https://t90182117410.p.clickup-attachments.com/t90182117410/2fc0f0dc-f115-474b-945f-e9a63b8855ca/d1446687-b5cb-4136-9e3b-b497e51a42f4.png)
(lower = tighter analog clusters)
## **Temporal Smoothness**
![](https://t90182117410.p.clickup-attachments.com/t90182117410/a70e7ccb-2ba7-4270-94c9-8cb534e485db/99132bf2-aefa-4fc2-9cdf-0a7e4c428f13.png)
# **20.5 Detailed Baseline Model Equations**
Complete list of baselines and their mathematical definitions.
## **(A) Random Walk**
![](https://t90182117410.p.clickup-attachments.com/t90182117410/472210dd-1e70-4ab1-a1f3-2a86d569ea37/2ea9ba40-c60f-4dd2-84fe-3e56dbc40990.png)
## **(B) Always Up**
![](https://t90182117410.p.clickup-attachments.com/t90182117410/d7f07692-5120-4f72-a266-2916c6f7eab6/f7b87935-e79f-4b81-a7e6-8915635f1844.png)
## **(C) Momentum 20-day**
![](https://t90182117410.p.clickup-attachments.com/t90182117410/2dcff84c-ea82-4aef-b21d-6e06577afceb/6cceb737-d6df-42eb-9317-704aa941e2af.png)
## **(D) AR(p)**
![](https://t90182117410.p.clickup-attachments.com/t90182117410/f924929c-901c-48bf-a8dd-d2d5bf750353/c9525110-d7fd-4d41-8501-48a7b1185a88.png)
## **(E) MA(q)**
![](https://t90182117410.p.clickup-attachments.com/t90182117410/33dba6b0-b318-49ec-856e-8cf84d821514/45ba9eb6-0e50-4c1a-a23a-2a1168dd56b5.png)
## **(F) ARMA(p,q)**
![](https://t90182117410.p.clickup-attachments.com/t90182117410/545fb233-d7d4-4c94-8115-2457b60c81ac/11020de5-2064-4c52-9c34-36a03f42eed5.png)
## **(G) ARIMA(p,d,q)**
![](https://t90182117410.p.clickup-attachments.com/t90182117410/958c5855-85c1-4528-8ba5-3c0069b09f9a/a6baed52-4be6-442f-9b64-472ce3266b3d.png)
## **(H) GARCH(1,1)**
​![](https://t90182117410.p.clickup-attachments.com/t90182117410/b44a84f6-473b-4390-a60a-8ab364ef96c2/b9d3dc7d-016a-45e7-87ac-3a547ef1586c.png)
## **(I) LSTM / GRU / Transformer**
Modern ML baselines defined by:
![](https://t90182117410.p.clickup-attachments.com/t90182117410/4d995a2e-4b45-4af0-9544-54d693b16d7b/0b01ad9a-48f4-4a32-bc52-4e7ecfd91016.png)
Full training regimen included in main report.
# **20.6 Additional Case Studies**
This section includes extra MVP examples for investor validation.
### **Case Study Types Included**
*   breakout continuation
*   momentum collapse
*   volatility compression → breakout
*   V-shaped recovery
*   bear rally exhaustion
*   sideways drift patterns
*   high-vol regime mean reversion
Each case study includes:
*   trajectory
*   analogs
*   forecast probabilities
*   observed outcome
*   reasoning narrative
*   interpretation
(Full catalog delivered as separate Case Study Pack.)
# **20.7 Compliance Templates**
This appendix includes pre-written compliance templates required for investors, enterprise buyers, and regulators.
## **Not Investment Advice Statement**
Standard disclaimer for all outputs.
## **Explainability Statement**
Describes how Kalibry generates empirical, analog-based explanations.
## **Traceability Documentation**
Defines:
*   analog list
*   input trajectory
*   evidence graph
*   prediction versioning
*   timestamped audit log
## **AI Act Alignment Summary**
Checklist covering:
*   transparency
*   human oversight
*   robustness
*   repeatability
*   fairness (non-applicable: no personal data)
*   data governance
*   risk controls
## **Risk Boundaries**
Formal statement of:
*   horizons
*   limitations
*   permitted usage
*   non-permitted usage
## **Security & Data Handling Template**
Procedure for:
*   data sourcing
*   encryption
*   access logging
*   role-based permissions