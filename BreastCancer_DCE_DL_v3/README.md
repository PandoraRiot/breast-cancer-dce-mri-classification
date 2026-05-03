# BreastCancer_DCE_DL — v3

> Transfer Learning + Fine-Tuning Progresivo | 5-Fold CV | RadImageNet | MobileViT-S | Umbral clínico ≥0.95 sensibilidad

## Experimento
- **Fecha:** 20260503_2130
- **Dataset:** BreastDM DCE-MRI — 1319 train/val · 403 test
- **Normalización:** Z-score por estudio (no ImageNet)
- **Threshold:** Optimizado en VAL → aplicado fijo en TEST
- **Target clínico:** Sensibilidad ≥ 0.95

## Resultados Test Set
| Modelo (Pretrain) | AUC-ROC [CI95] | Recall [CI95] | Especif. | NPV [CI95] | FNR⚠️ | Threshold |
|---|---|---|---|---|---|---|
| resnet50 (RadImageNet) | 0.8471 [0.719,0.949] | 1.0 [1.000,1.000] | 0.0 | 0.0 [0.234,0.511] | 0.0 | 0.07 |
| efficientnet_b3 (ImageNet) | 0.8647 [0.747,0.959] | 1.0 [1.000,1.000] | 0.2941 | 1.0 [0.154,0.433] | 0.0 | 0.17 |
| mobilevit_s (ImageNet) | 0.8804 [0.750,0.976] | 1.0 [1.000,1.000] | 0.3529 | 1.0 [0.139,0.424] | 0.0 | 0.165 |

**Benchmark BreastDM:** ACC ≥ 88.20% · AUC ≥ 0.9154

## 5-Fold CV (train+val, estratificado por paciente)
| Modelo | AUC mean±std | Recall mean±std | Folds con Recall≥0.95 |
|---|---|---|---|
| resnet50 | 0.7928±0.0000 | 0.9615±0.0000 | 1/5 |
| efficientnet_b3 | 0.7537±0.0000 | 0.9615±0.0000 | 1/5 |
| mobilevit_s | 0.8251±0.0000 | 0.9615±0.0000 | 1/5 |

## Correcciones respecto a iter2
- ✅ Normalización Z-score por estudio (no ImageNet)
- ✅ Threshold tuning en VAL → aplicado fijo en TEST
- ✅ 5-Fold CV estratificado por paciente
- ✅ MobileViT-S (Transformer) como tercer modelo
- ✅ RadImageNet para ResNet50
- ✅ Target sensibilidad ≥ 0.95

## Seed: 42
