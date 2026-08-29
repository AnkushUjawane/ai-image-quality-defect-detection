# Model Evaluation Report

Train samples: 1120  |  Test samples: 280

| Issue | Accuracy | Precision | Recall | F1 | ROC-AUC | Test positive rate |
|---|---|---|---|---|---|---|
| blur | 0.875 | 0.677 | 0.759 | 0.715 | 0.925 | 0.207 |
| underexposure | 0.864 | 0.755 | 0.587 | 0.661 | 0.840 | 0.225 |
| overexposure | 0.911 | 0.849 | 0.726 | 0.783 | 0.920 | 0.221 |
| noise | 0.961 | 0.983 | 0.851 | 0.912 | 0.982 | 0.239 |
| corruption | 0.964 | 0.933 | 0.903 | 0.918 | 0.992 | 0.221 |

## Confusion matrices (rows=actual, cols=predicted; order [0,1])

**blur**: [[201, 21], [14, 44]]

**underexposure**: [[205, 12], [26, 37]]

**overexposure**: [[210, 8], [17, 45]]

**noise**: [[212, 1], [10, 57]]

**corruption**: [[214, 4], [6, 56]]

## Top contributing features per issue

- **blur**: laplacian_var (0.20), noise_estimate (0.15), edge_density (0.12), high_freq_energy (0.10)
- **underexposure**: mean_brightness (0.32), colorfulness (0.12), brightness_std (0.10), overexposed_frac (0.10)
- **overexposure**: overexposed_frac (0.33), mean_brightness (0.25), saturation_mean (0.10), entropy (0.04)
- **noise**: noise_estimate (0.27), high_freq_energy (0.20), laplacian_var (0.19), entropy (0.08)
- **corruption**: underexposed_frac (0.24), noise_estimate (0.18), block_artifact_score (0.16), laplacian_var (0.10)

## Failure cases & limitations

- Trained entirely on procedurally generated synthetic images with controlled degradations; real-world photographic distributions may differ and reduce generalization.
- Overexposure and noise can be confounded on low-detail bright images, reducing noise recall in those scenarios.
- Corruption detection emphasizes block-artifact and high-frequency cues; hard file corruption (undecodable uploads) is handled in API validation before model inference.
- Multi-issue images (2-3 simultaneous degradations) are harder to disentangle than single-issue images, so per-issue recall may drop.
