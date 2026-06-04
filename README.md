# RainCast

**RainCast** is a high-resolution 72-hour short-term precipitation forecasting model for hourly precipitation prediction over China. It was published in **KDD '26**. [[Paper](https://doi.org/10.1145/3770855.3818880)]

[weight and example link](https://box.nju.edu.cn/d/b72deb1b858c4f66a3ff/)


## Input Data

### 1. Baguan global forecast

For each forecast initialization time `YYYYMMDDHH`, the Baguan global forecast file should be placed at:

```
input/1.Baguan_gloabl_forecast/YYYYMMDDHH.npy
```

The expected shape is:

```
(72, 71, 151, 245)
```

where:

- `72` denotes the forecast lead times from 1 h to 72 h;
- `71` denotes the meteorological variables;
- `151 × 245` denotes the Baguan forecast grid;
- the latitude range is **16.25°N–53.75°N**;
- the longitude range is **74.00°E–135.00°E**;
- the spatial resolution is **0.25° × 0.25°**.

### 2. CMPAS initial precipitation

CMPAS provides the initial precipitation condition for RainCast. For each forecast initialization time, the script uses the three precipitation frames immediately before the forecast time.

Each CMPAS file should have the shape:

```
(736, 1216)
```

where:

- `736 × 1216` denotes the high-resolution precipitation grid;
- the latitude range is **16.60°N–53.35°N**;
- the longitude range is **74.10°E–134.85°E**;
- the spatial resolution is **0.05° × 0.05°**.

After loading the three previous frames, the initial precipitation input has shape:

```
(3, 1, 736, 1216)
```

The ONNX output also uses the same high-resolution grid as CMPAS:

```
(72, 1, 736, 1216)
```

### 3. Internal ONNX inputs

The ONNX model expects the following input names:

```text
input_baguan_forecast
input_initial_raw
time_feature
leadtime_feature
lead_time
```

For each inference batch with batch size `B`, their shapes are:

```text
input_baguan_forecast : (B, 71, 151, 245)
input_initial_raw     : (B, 3, 1, 736, 1216)
time_feature          : (B, 4, 368, 608)
leadtime_feature      : (B, 4, 368, 608)
lead_time             : (B,)
```

## Run Inference

Please organize the files as follows:

```
.
├── infer_ref_onnx/
│   ├── infer_onnx.py
│   └── raincast_reg.onnx
└── input/
    ├── 1.Baguan_gloabl_forecast/
    │   └── YYYYMMDDHH.npy
    └── 2.CMAPS/
        └── YYYYMMDDHH.npy
```

Run inference from the repository root:

```bash
python infer_reg_onnx/infer_onnx.py \
  --start_time 2024062012 \
  --onnx_path infer_reg_onnx/raincast_reg.onnx \
  --input_path input \
  --output_path infer_reg_onnx/output \
  --lead_time 72
```

The script will use CUDA if available and fall back to CPU automatically. The output will be saved as:

```text
infer_reg_onnx/output/2024062012.npy
```

The output shape is:

```text
(72, 1, 736, 1216)
```

Each frame corresponds to the predicted hourly precipitation from lead time 1 h to 72 h.

## Notes

- The original input datasets used by RainCast may be difficult to obtain. The Baguan global forecast can be replaced by forecast fields from other large AI weather models or numerical weather prediction models, as long as the variables, order, spatial grid, and data format are made consistent with the expected input.
- The CMPAS precipitation input can also be replaced by other precipitation datasets with the same spatial resolution and grid definition.
- This release only includes the deterministic regression-head ONNX inference. The ensemble forecasting version will be released later.