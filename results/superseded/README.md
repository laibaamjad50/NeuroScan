# Superseded results

These are the EfficientNet scores from the first evaluation pass, kept for the record.
They were produced by scoring the models **after** fine-tuning had damaged them.

Two bugs caused it (both explained in Section 5 of `Phase2_Track1_Kainat.ipynb`):

1. `base.trainable = True` left BatchNormalization layers trainable, so fine-tuning
   recomputed their moving statistics and degraded the pretrained features.
2. `EarlyStopping(restore_best_weights=True)` resets per `fit()` call, so after phase B
   it restored the best *phase B* epoch rather than phase A's much better weights.

`ModelCheckpoint` does not reset between calls, so the good phase-A weights survived in
`models/*.keras`. Reloading them recovered about 16 points:

| run | here (damaged) | corrected (`*_best` in the parent folder) |
|---|---|---|
| effnet_clean | 0.6943 | 0.8551 |
| effnet_leaky | 0.7469 | 0.8850 |
| effnet_leaky_on_clean_test | 0.7498 | 0.8858 |

**Do not quote these numbers.** Use the `*_best` files in `results/`.
