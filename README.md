# pore_pipeline

Clinical-grade facial pore analysis pipeline from a single unfiltered RGB smartphone selfie. Skincare tool, **not a medical device**. Zero-shot foundation models + classical image processing. No training.

## What it does

Given one selfie, produces:

- Per-pore records (location, diameter in mm, eccentricity, contrast index, visibility index, confidence)
- Per-zone aggregates across 12 anatomical zones (count, density/cm², diameter stats, Flament 0–9 score, plausibility flag)
- Whole-face composite (0–100 score, IBSA 1–5 pore grade, plausibility interval, dominant zone, L–R asymmetry)
- Full JSON output with every measurement's provenance
- Expert diagnostic dashboard

## Architecture (11 stages)

1. **Capture-quality gate** — face presence, pose, blur, exposure, capture distance, classical texture-energy floor (advisory beauty-filter indicator)
2. **Face detection + 478-landmark mesh** — MediaPipe FaceLandmarker (Tasks API)
3. **Skin isolation** — landmark-based masking + optional BiSeNet face parser + hair-edge shadow-band exclusion
4. **Illumination normalization** — Shades-of-Gray → L\*a\*b\* → ITA° → Fitzpatrick bucket → homomorphic filter → phototype-adaptive CLAHE
5. **Per-zone ROI extraction** — 12 anatomical zones from fixed MediaPipe landmark indices
6. **Classical pore candidate detection** — multi-scale DoG + h-basin/h-dome morphology + Loy-Zelinsky fast radial symmetry + 8-orientation hair-coherence bank + adaptive NMS
7. **Zero-shot false-positive suppression** — three-signal ensemble:
   - **DINOv2** within-zone Mahalanobis outlier scoring
   - **SAM 2** point-prompted segmentation → per-pore mask shape gates
   - **Depth Anything V2** pseudo-depth profile analysis for scar discrimination
8. **Per-pore metrology** — diameter mm, area mm², eccentricity, orientation, contrast index, visibility index, confidence
9. **Per-zone aggregation** — counts, densities, stats, Flament mapping, literature-plausibility flags
10. **Whole-face composite** — zone-weighted score, asymmetry index, IBSA 1–5 grade, plausibility interval
11. **Output packaging** — JSON + dashboard + rejection log

## Usage

### On Colab

```python
!pip install -q -r requirements.txt
from pore_pipeline import (
    run_pipeline, PipelineModels,
    FaceLandmarkDetector, BiSeNetONNXParser,
    DinoV2Encoder, Sam2Prompter, DepthAnythingV2Encoder,
    render_dashboard,
)

models = PipelineModels(
    face_detector=FaceLandmarkDetector(),
    face_parser=BiSeNetONNXParser(),
    dino=DinoV2Encoder(),
    sam=Sam2Prompter(),
    depth=DepthAnythingV2Encoder(),
)

result = run_pipeline(
    "/content/my_selfie.jpg",
    models,
    output_json_path="/content/result.json",
    debug_keep_intermediates=True,
)

print(f"Fitzpatrick: {result.fitzpatrick}")
print(f"Detected pores: {len(result.records)}")
print(f"Composite score: {result.whole_face.composite_score_0_100:.1f}/100")

fig = render_dashboard(
    bgr=cv2.imread("/content/my_selfie.jpg"),
    skin=result.debug["skin"],
    zones=result.debug["zones"],
    records=result.records,
    zone_aggs=result.zone_aggs,
    whole_face=result.whole_face,
    fitzpatrick=result.fitzpatrick,
    warnings=result.warnings,
)
fig.savefig("/content/dashboard.png", dpi=150, bbox_inches="tight")
```

### Graceful degradation

Every foundation model wrapper in `PipelineModels` is optional. Missing models cause the pipeline to fall back to classical signals with warnings in the output JSON:

- `dino=None` → within-zone outlier rejection disabled
- `sam=None` → per-pore metrology uses classical elliptic fit instead of SAM mask
- `depth=None` → scar-vs-pore depth-profile discrimination disabled
- `face_parser=None` → skin isolation uses landmarks only (hair strands may cause false positives)

The minimum viable run requires only `face_detector` (MediaPipe). Everything else optional.

## Calibration

**Heuristic mode** (default): Flament/IBSA grades are derived from density + visibility via piecewise-linear functions calibrated to published density ranges. Output JSON flags `calibration_basis_global: "heuristic_no_anchors"`.

**Anchor mode** (if you have the published Flament 2004 and/or IBSA 2024 anchor photographs):

```python
from pore_pipeline import Anchor, calibrate_flament_from_anchors

anchors = [Anchor(img_path, grade, "flament_10") for img_path, grade in [
    ("flament_0.jpg", 0.0), ("flament_1.jpg", 1.0), ...,  # all 10 anchors
]]

def extract(img_path):
    r = run_pipeline(img_path, base_models)
    return {
        "face_mean_visibility": np.mean([a.visibility_mean for a in r.zone_aggs.values()]),
        "whole_face_composite": r.whole_face.composite_score_0_100,
    }

calib = calibrate_flament_from_anchors(extract, anchors)
models.flament_calib = calib
```

Anchor images are **not** distributed with this code — licensing is the caller's responsibility.

## Honest limitations

- **No cohort validation.** Performance claims are principled-plausible, not cohort-measured.
- **Fitzpatrick V–VI extrapolation.** Flament anchors skew Fitzpatrick II–IV; V–VI outputs are flagged `extrapolated_calibration: true`.
- **Foundation-model bias.** DINOv2 was trained on Fitzpatrick-skewed web data; within-zone outlier scoring mitigates but does not eliminate this.
- **Beauty filters.** Classical texture-energy floor flags likely filter use but is not a learned retouching detector. Manual verification is advised.
- **Absolute accuracy not claimed.** The composite score is emitted with a *plausibility interval*, not a statistical confidence interval. Downstream consumers must not conflate the two.

## Output JSON top-level fields

```
schema_version, pipeline_version, timestamp_utc,
calibration_basis_global,
image_metadata,
capture_quality (pass, reasons, warnings, blur, exposure, pose, ppmm, texture_energy),
phototype (fitzpatrick_estimate, ita_deg, extrapolated_calibration),
skin_isolation (parser_backend),
zones[] (count, density_per_cm2, diameter stats, visibility, flament_score, plausibility),
whole_face (composite_score, plausibility_interval, ibsa_pore_grade, dominant_zone, asymmetry),
model_versions (pinned identifiers per stage),
warnings, limitations
```

See `pore_pipeline/pipeline.py::_build_json` for the full schema.

## Project layout

```
pore_pipeline/
├── pore_pipeline/
│   ├── __init__.py          # public API
│   ├── config.py            # thresholds, zones, phototype params
│   ├── utils.py             # color, geometry, scale calibration
│   ├── quality.py           # Stage 1
│   ├── face.py              # Stage 2 (MediaPipe)
│   ├── parsing.py           # Stage 3 (landmarks + BiSeNet)
│   ├── illumination.py      # Stage 4
│   ├── zones.py             # Stage 5
│   ├── candidates.py        # Stage 6 (classical)
│   ├── suppression.py       # Stage 7 (DINOv2 + SAM 2 + Depth Anything V2)
│   ├── metrology.py         # Stages 8 + 9 + 10
│   ├── mapping.py           # Flament / IBSA anchor calibration
│   ├── visualization.py     # dashboard
│   └── pipeline.py          # orchestrator
├── tests/
│   └── mock_e2e.py          # synthetic end-to-end sanity
├── requirements.txt
├── README.md
└── run_colab.ipynb          # Colab demo
```

## License & scope

Research/skincare use. Not a medical device. No disease diagnosis. See the planning document (`pore_pipeline_plan_v2.md`) for full design rationale and risk register.
