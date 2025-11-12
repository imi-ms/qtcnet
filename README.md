# QTcNet

**QTcNet** is a deep-learning model for precise measurement of the heart-rate–corrected QT-interval (QTc) from 12-lead ECGs.
The model uses a regression variant of the InceptionTime architecture:
- Pre-Training: 120 300 algorithm-labelled ECGs
- 100 HZ sampling rate
- Vendor QTc bias (+15 ms) removed before training
- Fine-Tuning: 445 cardiologist-annotated ECGs (PTB-Diagnostic Database)

Performance (expert-labelled cohorts)

| Dataset            | PTB   | QTcMS | ECGRDVQ | Average |
|--------------------|-------|-------|---------|---------|
| **MAE&nbsp;(ms)**  | 18.84 | 13.88 | 7.42    | 13.38   |
| **RMSE&nbsp;(ms)** | 29.61 | 24.85 | 11.78   | 22.08   |


Without fine-tuning, QTcNet already cuts mean absolute error from 23.4 ms to 13.4 ms and nearly halves large (> 50 ms) outliers.

Why use QTcNet?
- State-of-the-art accuracy - ~50 % less error than typical commercial algorithms
- Rapidly adaptable - a few hundred expert labels are enough to match local clinical standards
- Openn Source - code & pretrained weights plus an online demo at https://qtcnet.uni-muenster.de
- Explainable - Integrated-Gradient maps show focus on QRS onset & T-wave offset

Lead order expected by the model
```
I, II, III, aVR, aVF, aVL, V1, V2, V3, V4, V5, V6
```

Citations:
```
@article{10.1093/europace/euaf274,
    author = {Plagwitz, Lucas and Doldi, Florian and Magerfleisch, Jannes and Zotov, Maxim and Bickmann, Lucas and Heider, Dominik and Varghese, Julian and Eckardt, Lars and Büscher, Antonius},
    title = {QTcNet: A Deep Learning Model for Direct Heart Rate Corrected QT Interval Estimation},
    journal = {EP Europace},
    pages = {euaf274},
    year = {2025},
    month = {10},
    issn = {1099-5129},
    doi = {10.1093/europace/euaf274},
    url = {https://doi.org/10.1093/europace/euaf274},
    eprint = {https://academic.oup.com/europace/advance-article-pdf/doi/10.1093/europace/euaf274/64953062/euaf274.pdf},
}
```