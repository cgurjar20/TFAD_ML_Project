# TFAD_ML_Project
Project on Time Frequency Anomaly Detection in Machine Learning
A Machine Learning project on Time Series Anomaly Detection using TFAD framework on SMAP, MSL and SWaT benchmark datasets.
This project Enhances and improves the TFAD (Time-Frequency Analysis based Time Series Anomaly Detection) framework proposed in the CIKM 2022 paper for anomaly detection.

Key enhancements implemented:
1. Explored FFT based frequency domain feature augmentation for explicit spectral anomaly representation and compared performance against the baseline TFAD Pipeline.
2. Developed channel wise importance weighting for improved sensor relevance learning.
3. Added Threshold Optimization using validation F1 Score sweeps.
4. Extended evaluation with Accuracy, Specificity, FPR, FNR like Metrics.
5. Created improved visualizations including confusion matrix, metric dashboards, and prediction distribution analysis.
6. Implemented checkpoint mechanisms using PyTorch Lightning for experiment tracking.

Project combines concepts involving - 
1. Temporal Learning using TCNs
2. Frequency Domain Analysis using FFT
3. Deep Learning Networks based Anomaly Detection

Technologies used - 
PyTorch, PyTorch Lightning, Temporal Convolutional Network, Fast Fourier Transform, and Time Series Analysis.

By performing experiments, I concluded that, TFAD is highly effective for multivariate anomaly detection, while the extensions as FFT enhance the performance in specific scenarios, which depend on the factors like dataset properties, feature representations, threshold selection like parameters while designing robust anomaly detection systems.
