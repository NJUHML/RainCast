import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent


def default_input_path():
    local_input = SCRIPT_DIR / "input"
    if local_input.exists():
        return local_input
    return REPO_DIR / "infer_reg_raw" / "input"


def resolve_path(path, base=SCRIPT_DIR):
    path = Path(path)
    if path.is_absolute():
        return path
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (base / path).resolve()


def is_leap_year(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def make_time_feature(start_times, lead_times, embed_dim=4, img_size=(368, 608)):
    start_times = np.asarray(start_times).reshape(-1)
    lead_times = np.asarray(lead_times).reshape(-1)
    if start_times.shape[0] != lead_times.shape[0]:
        raise ValueError("start_times and lead_times must have the same batch length.")
    if embed_dim % 4 != 0:
        raise ValueError("time_embed_dim must be a multiple of 4.")

    h, w = img_size
    d_split = embed_dim // 2
    vectors = []
    for start_time, lead_time in zip(start_times, lead_times):
        dt = datetime.strptime(str(int(start_time)), "%Y%m%d%H") + timedelta(hours=int(lead_time))

        year_start = datetime(dt.year, 1, 1, 0)
        hour_in_year = int((dt - year_start).total_seconds() // 3600)
        hours_per_year = (366 if is_leap_year(dt.year) else 365) * 24
        year_angle = 2 * np.pi * hour_in_year / hours_per_year
        pos_year = year_angle * np.arange(1, d_split // 2 + 1)

        day_angle = 2 * np.pi * dt.hour / 24
        pos_day = day_angle * np.arange(1, d_split // 2 + 1)

        vectors.append(np.concatenate([
            np.sin(pos_year),
            np.cos(pos_year),
            np.sin(pos_day),
            np.cos(pos_day),
        ]))

    features = np.asarray(vectors, dtype=np.float32)[:, :, None, None]
    return np.broadcast_to(features, (features.shape[0], embed_dim, h, w)).copy()


def make_leadtime_feature(lead_times, embed_dim=4, max_lead_time=72, img_size=(368, 608)):
    lead_times = np.asarray(lead_times).reshape(-1)
    if embed_dim % 2 != 0:
        raise ValueError("leadtime_embed_dim must be even.")

    h, w = img_size
    d = embed_dim // 2
    vectors = []
    for lead_time in lead_times:
        lead_angle = 2 * np.pi * int(lead_time) / max_lead_time
        pos_lead = lead_angle * np.arange(1, d + 1)
        vectors.append(np.concatenate([np.sin(pos_lead), np.cos(pos_lead)]))

    features = np.asarray(vectors, dtype=np.float32)[:, :, None, None]
    return np.broadcast_to(features, (features.shape[0], embed_dim, h, w)).copy()


class DataPreprocessor:
    def __init__(self, input_path, lead_time=72):
        self.input_path = Path(input_path).resolve()
        self.lead_time = int(lead_time)

    def load(self, start_time):
        start_dt = datetime.strptime(str(start_time), "%Y%m%d%H")

        forecast_path = self.input_path / "1.Baguan_gloabl_forecast" / f"{start_dt:%Y%m%d%H}.npy"
        forecast = np.load(forecast_path, mmap_mode="r")
        forecast = np.ascontiguousarray(forecast[:self.lead_time, :71, :, :], dtype=np.float32)

        initial_frames = []
        for idx in range(3):
            cmaps_dt = start_dt - timedelta(hours=idx) + timedelta(hours=8)
            cmaps_path = self.input_path / "2.CMAPS" / f"{cmaps_dt:%Y%m%d%H}.npy"
            initial_frames.append(np.asarray(np.load(cmaps_path), dtype=np.float32))
        initial_base = np.stack(initial_frames, axis=0)[::-1].copy()

        start_times = np.full((self.lead_time,), int(start_dt.strftime("%Y%m%d%H")), dtype=np.int64)
        lead_times = np.arange(1, self.lead_time + 1, dtype=np.int64)

        return {
            "input_baguan_forecast": forecast,
            "input_initial_base": np.ascontiguousarray(initial_base, dtype=np.float32),
            "input_start_time": start_times,
            "lead_time": lead_times,
        }


def pad_batch(array, batch_size):
    if array.shape[0] == batch_size:
        return array, array.shape[0]
    if array.shape[0] > batch_size:
        raise ValueError(f"Cannot pad array with batch {array.shape[0]} to smaller batch_size {batch_size}.")
    pad_count = batch_size - array.shape[0]
    pad_values = np.repeat(array[-1:], pad_count, axis=0)
    return np.concatenate([array, pad_values], axis=0), array.shape[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Run RainCast_Reg ONNX inference.")
    parser.add_argument("--start_time", type=str, default="2024062012", help="Forecast start time in YYYYMMDDHH format.")
    parser.add_argument("--lead_time", type=int, default=72)
    parser.add_argument("--onnx_path", type=str, default=str(SCRIPT_DIR / "raincast_reg.onnx"))
    parser.add_argument("--input_path", type=str, default=str(default_input_path()))
    parser.add_argument("--output_path", type=str, default=str(SCRIPT_DIR / "output"))
    parser.add_argument("--infer_batch_size", type=int, default=1, help="Used only when ONNX has a dynamic batch dimension.")
    parser.add_argument("--providers", type=str, default="CUDAExecutionProvider,CPUExecutionProvider")

    parser.add_argument("--time_embed_dim", type=int, default=4)
    parser.add_argument("--leadtime_embed_dim", type=int, default=4)
    parser.add_argument("--max_lead_time", type=int, default=72)
    return parser.parse_args()


def make_session(onnx_path, providers):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime or onnxruntime-gpu is required on the server for ONNX inference.") from exc

    requested = [provider.strip() for provider in providers.split(",") if provider.strip()]
    available = set(ort.get_available_providers())
    selected = [provider for provider in requested if provider in available]
    if not selected:
        selected = ["CPUExecutionProvider"]

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=selected)
    print(f"ONNXRuntime providers: {session.get_providers()}")
    return session


def static_batch_size(session):
    first_dim = session.get_inputs()[0].shape[0]
    return first_dim if isinstance(first_dim, int) else None


def validate_session_inputs(session):
    expected = {
        "input_baguan_forecast",
        "input_initial_raw",
        "time_feature",
        "leadtime_feature",
        "lead_time",
    }
    actual = {input_info.name for input_info in session.get_inputs()}
    missing = sorted(expected - actual)
    if missing:
        raise RuntimeError(
            "The ONNX model inputs do not match this pure inference script. "
            f"Missing inputs: {missing}. Actual inputs: {sorted(actual)}. "
            "Please use the ONNX exported by the latest export_onnx.py."
        )


def build_feed(inputs, start, end, batch_size, args):
    actual_batch = end - start
    lead_times = inputs["lead_time"][start:end]
    start_times = inputs["input_start_time"][start:end]

    input_baguan_forecast, _ = pad_batch(inputs["input_baguan_forecast"][start:end], batch_size)
    lead_time, _ = pad_batch(lead_times[:, None], batch_size)
    lead_time = lead_time.reshape(-1).astype(np.int64)
    start_times, _ = pad_batch(start_times[:, None], batch_size)
    start_times = start_times.reshape(-1).astype(np.int64)

    input_initial_raw = np.repeat(inputs["input_initial_base"][None, :, :, :], actual_batch, axis=0)
    input_initial_raw, _ = pad_batch(input_initial_raw, batch_size)
    input_initial_raw = np.ascontiguousarray(input_initial_raw, dtype=np.float32)

    img_size = (368, 608)
    time_feature = make_time_feature(
        start_times,
        lead_time,
        embed_dim=args.time_embed_dim,
        img_size=img_size,
    )
    leadtime_feature = make_leadtime_feature(
        lead_time,
        embed_dim=args.leadtime_embed_dim,
        max_lead_time=args.max_lead_time,
        img_size=img_size,
    )

    return {
        "input_baguan_forecast": np.ascontiguousarray(input_baguan_forecast, dtype=np.float32),
        "input_initial_raw": input_initial_raw,
        "time_feature": time_feature,
        "leadtime_feature": leadtime_feature,
        "lead_time": lead_time,
    }, actual_batch


def main():
    args = parse_args()
    onnx_path = resolve_path(args.onnx_path, Path.cwd())
    input_path = resolve_path(args.input_path, Path.cwd())
    output_path = resolve_path(args.output_path, Path.cwd())
    output_path.mkdir(parents=True, exist_ok=True)

    session = make_session(onnx_path, args.providers)
    validate_session_inputs(session)
    exported_batch = static_batch_size(session)
    batch_size = exported_batch if exported_batch is not None else args.infer_batch_size
    print(f"Inference batch size: {batch_size}")

    preprocessor = DataPreprocessor(input_path=input_path, lead_time=args.lead_time)
    inputs = preprocessor.load(args.start_time)
    print(f"input_baguan_forecast raw: {inputs['input_baguan_forecast'].shape}")
    print(f"input_initial_base raw: {inputs['input_initial_base'].shape}")

    outputs = []
    for start in range(0, args.lead_time, batch_size):
        end = min(start + batch_size, args.lead_time)
        feed, actual_batch = build_feed(inputs, start, end, batch_size, args)
        rain_norm = session.run(["rain_norm"], feed)[0]
        rain = rain_norm[:actual_batch] * 10.0
        outputs.append(rain.astype(np.float32, copy=False))
        print(f"finished lead {start + 1}-{end}: {rain.shape}")

    y_hat = np.concatenate(outputs, axis=0)
    save_path = output_path / f"{args.start_time}.npy"
    np.save(save_path, y_hat)
    print(f"Saved: {save_path}")
    print(f"Output shape: {y_hat.shape}")


if __name__ == "__main__":
    main()
