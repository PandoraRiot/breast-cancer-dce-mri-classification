# 📋 README — iter1

**Fecha:** 2026-04-16 00:00  
**Experimento:** BreastDCE_TransferLearning_iter1  
**Objetivo clínico:** Maximizar Recall (Sensibilidad ≥ 0.90) para minimizar Falsos Negativos

---

## 🔧 Configuración

| Parámetro | Valor |
|---|---|
| Modelos | ResNet50, EfficientNet-B3 |
| Epochs F1 (frozen) | 5 |
| Epochs F2 (finetune) | 25 |
| LR fase 1 | 0.001 |
| LR fase 2 | 5e-05 |
| Batch size | 16 |
| Dropout | 0.4 |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR |
| pos_weight | 0.3737 |
| Early stopping patience | 7 |
| Seed | 42 |

## 📊 Dataset

| Split | Benigno | Maligno | Total |
|---|---|---|---|
| Train | 327 | 875 | 1202 |
| Val   | 24  | 93  | 117  |
| Test  | 114 | 289 | 403  |

**Input:** (B, 3, 256, 256) — canales: VIBRANT (pre), VIBRANT+C2 (early), VIBRANT+C7 (late)

## 🏆 Resultados — Test Set

| Métrica | ResNet50 | EfficientNet-B3 |
|---|---|---|
| AUC-ROC | 0.8486 [0.799, 0.896] | 0.821 [0.771, 0.867] |
| Recall (Sens.) | 0.9204 [0.888, 0.948] | 0.91 [0.877, 0.939] |
| Especificidad | 0.6667 | 0.5263 |
| F1-Score | 0.8971 | 0.868 |
| Balanced Acc | 0.7935 | 0.7182 |
| PR-AUC | 0.9107 | 0.9042 |
| Brier Score | 0.1264 | 0.1774 |
| FN (críticos) | 23 | 26 |
| FP | 38 | 54 |
| Threshold | 0.43 | 0.1 |
| Train time (min) | 3.81 | 4.74 |
| Inf (ms/img) | 6.9 | 6.4 |

## 📁 Estructura de archivos

```
iter1/
├── configs/
│   └── config.yaml
├── evidencias/
│   ├── 00_distribucion_clases.png
│   ├── 01_curvas_entrenamiento.png
│   ├── 02_evaluacion_completa.png
│   ├── 03_gradcam_sidebyside.png
│   ├── resnet50_error_analysis.png
│   ├── resnet50_metrics_per_sample.csv
│   ├── efficientnet_b3_error_analysis.png
│   └── efficientnet_b3_metrics_per_sample.csv
├── modelos_guardados/
│   ├── Alexandra_ResNet50_iter1_20260416_0000.pth
│   ├── Alexandra_ResNet50_iter1_20260416_0000.pkl
│   ├── Alexandra_EfficientNetB3_iter1_20260416_0000.pth
│   └── Alexandra_EfficientNetB3_iter1_20260416_0000.pkl
└── logs/
    ├── tensorboard/
    ├── mlruns/
    └── results_summary_iter1.csv
```

## 🔍 MLflow — Cómo ver resultados

```bash
# En Colab o local:
mlflow ui --backend-store-uri file:///content/breast-cancer-dce-mri-classification/iter1/logs/mlruns --port 5000
```
Los runs quedan guardados en `iter1/logs/mlruns/`.
Para la siguiente iteración, cambia `ITER = "iter2"` al inicio del notebook.

## 💡 Observaciones y próximos pasos

- [ ] Comparar con iter2 (ajustar hiperparámetros si Recall < 0.90)
- [ ] Probar data augmentation más agresivo en clase benigna
- [ ] Evaluar WeightedRandomSampler adicional al pos_weight
- [ ] Probar ViT (Vision Transformer) en iter3

---
*Generado automáticamente por el notebook de iter1*
