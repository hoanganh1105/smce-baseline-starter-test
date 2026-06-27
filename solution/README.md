# Team solution folder

**Replace the Python modules in this folder with your team's pipeline.**  
The Streamlit demo calls your code through **`shared/benchmark.py`** (template-owned — do not replace).

## Required API

Implement in [`pipeline.py`](./pipeline.py):

```python
def predict_from_image(img: PIL.Image.Image, min_conf: float = 0.35) -> dict[str, str]:
    """
    Returns at minimum:
        {
            "ocr_text": str,
            "brand_name": str,
            "product_name": str,
        }
    """
```

Optional `timing_ms` in the return dict — if omitted, the template benchmark layer measures wall-clock time for you.

Optional:

```python
def predict_from_text(ocr_text: str) -> tuple[str, str]:
    """Return (brand_name, product_name) from OCR text only."""
```

## Model footprint (template-owned)

Edit **`MODEL_PROFILE`** in [`team_config.py`](../team_config.py), not files here.

Benchmark / timing: [`shared/benchmark.py`](../shared/benchmark.py) (always runs on Streamlit deploy + Live test).

```bash
python scripts/benchmark_solution.py --limit 6
```

## Files in this folder

| File | Role | Team action |
|------|------|-------------|
| [`pipeline.py`](./pipeline.py) | OCR + brand/product orchestration | **Replace / enhance** |
| [`brand_rules.py`](./brand_rules.py) | Regex brand dictionary (baseline) | Extend or swap |
| [`product_model.py`](./product_model.py) | Sklearn product head (baseline) | Replace with your model |
| [`baseline_notebook.ipynb`](./baseline_notebook.ipynb) | Reference EDA + experiments | Replace with your notebook |

## Do not modify

- [`../shared/benchmark.py`](../shared/benchmark.py) — timing + deploy smoke benchmark
- [`../shared/`](../shared/) — data path helpers
- [`../private_test/metric.py`](../private_test/metric.py) — official scoring
- [`../streamlit_app.py`](../streamlit_app.py) core logic (About text optional)

## Quick test

```bash
python scripts/run_submission.py --limit 6
streamlit run streamlit_app.py
```

See [README.md](../README.md#team-setup) for the full fork checklist.
