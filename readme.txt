This repository contains the accompanying code for the paper “Distinct Origins of Radial and Tangential MEG Components: Gyrus Side Area vs. Whole Gyrus" . The primary goal is to transform raw time series signals into a robust set of high-dimensional statistical features that characterize signal complexity, amplitude fluctuations, and frequency-domain dynamics. The pipeline moves beyond basic statistical measures (like mean and standard deviation) by incorporating advanced metrics such as Sample Entropy (SampEn), Skewness, Kurtosis, and Event-Related Desynchronization/Synchronization (ERD/ERS).
Mean_STD.py, Skew_Kurt.py, and Sample_Entropy.py are used to calculate low-dimensional statistical descriptors of the signal's shape and magnitude across multiple levels and measure the complexity and unpredictability of the signal time series (trial-level, session-level). ttest.py performs rigorous statistical hypothesis testing across all derived features (mean, std, skew, …) between orthogonal directions (radial vs. tangential). ERDERS.py computes Event-Related Desynchronization and Synchronization (ERD/ERS). This is crucial for analyzing oscillatory activity in specific frequency bands (e.g., μ-rhythm, β-band).

Before running the pipeline, ensure your environment is set up:
Python 3.8+
Libraries: Install all necessary scientific packages.
pip install numpy pandas scipy matplotlib seaborn numba joblib scikit-learn

Step-by-Step Execution
Due to the sequential nature of signal processing, the analysis must be run in the following order:
Step 1: Feature Extraction (SampEn,Skew,…)
Step 2: ERD/ERS Calculation & Visualization
Step 3: Statistical Inference (Hypothesis Testing)
Step 4: Visualization (Final Report Generation)