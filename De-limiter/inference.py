import os
import json
import argparse
import glob

import numpy as np
import torch
import tqdm
import soundfile as sf
import pyloudnorm as pyln
from dotmap import DotMap

from models import load_model_with_args
from separate_func import (
    conv_tasnet_separate,
)
from utils import str2bool, db2linear


tqdm.monitor_interval = 0


def separate_track_with_model(
    args, model, device, track_audio, track_name, meter, augmented_gain
):
    with torch.no_grad():
        if (
            args.model_loss_params.architecture == "conv_tasnet_mask_on_output"
            or args.model_loss_params.architecture == "conv_tasnet"
        ):
            estimates = conv_tasnet_separate(
                args,
                model,
                device,
                track_audio,
                track_name,
                meter=meter,
                augmented_gain=augmented_gain,
            )

        return estimates


def main():
    def float_or_none(val):
        if val is None or str(val).strip().lower() in ("none", "null", "false", ""):
            return None
        return float(val)

    parser = argparse.ArgumentParser(description="model test.py")
    parser.add_argument("--target", type=str, default="all")
    parser.add_argument("--data_root", type=str, default="./input_data")
    parser.add_argument("--weight_directory", type=str, default="./weight")
    parser.add_argument("--output_directory", type=str, default="./output")
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--use_mps", type=str2bool, default=False)
    parser.add_argument("--save_name_as_target", type=str2bool, default=False)
    parser.add_argument(
        "--loudnorm_input_lufs",
        type=float_or_none,
        default=None,
        help="If you want to use loudnorm for input",
    )
    parser.add_argument(
        "--save_output_loudnorm",
        type=float_or_none,
        default=-14.0,
        help="Save loudness normalized outputs or not. If you want to save, input target loudness",
    )
    parser.add_argument(
        "--save_mixed_output",
        type=float_or_none,
        default=None,
        help="Save original+delimited-estimation mixed output with a ratio of default 0.5 (orginal) and 1 - 0.5 (estimation)",
    )
    parser.add_argument(
        "--save_16k_mono",
        type=str2bool,
        default=False,
        help="Save 16k mono wav files for FAD evaluation.",
    )
    parser.add_argument(
        "--save_histogram",
        type=str2bool,
        default=False,
        help="Save histogram of the output. Only valid when the task is 'delimit'",
    )
    parser.add_argument(
        "--use_singletrackset",
        type=str2bool,
        default=False,
        help="Use SingleTrackSet if input data is too long.",
    )
    parser.add_argument(
        "--save_sgi_params",
        type=str2bool,
        default=False,
        help="Save SGI params for further analysis",
    )
    parser.add_argument(
        "--avoid_clipping",
        type=str2bool,
        default=True,
        help="Forcefully normalize an output by np.abs(wav).max() if there exist some values that exceed 0 dBFS",
    )

    args, _ = parser.parse_known_args()

    with open(f"{args.weight_directory}/{args.target}.json", "r") as f:
        args_dict = json.load(f)
        args_dict = DotMap(args_dict)

    for key, value in args_dict["args"].items():
        if key in list(vars(args).keys()):
            pass
        else:
            setattr(args, key, value)

    args.test_output_dir = f"{args.output_directory}"
    os.makedirs(args.test_output_dir, exist_ok=True)

    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device("cuda")
    elif getattr(args, "use_mps", False) and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    ###################### Define Models ######################
    our_model = load_model_with_args(args)
    our_model = our_model.to(device)

    target_model_path = f"{args.weight_directory}/{args.target}.pth"
    checkpoint = torch.load(target_model_path, map_location=device)
    our_model.load_state_dict(checkpoint)

    our_model.eval()

    meter = pyln.Meter(44100)

    if os.path.isfile(args.data_root):
        test_tracks = [f"{args.data_root}"]
    else:  # if data is folder
        test_tracks = sorted(
            glob.glob(f"{args.data_root}/*.wav")
            + glob.glob(f"{args.data_root}/*.mp3")
        )

    for track in tqdm.tqdm(test_tracks):
        track_name = os.path.basename(track).replace(".wav", "").replace(".mp3", "")
        track_data, sr = sf.read(track, dtype="float32", always_2d=True)
        track_audio = np.ascontiguousarray(track_data.T)

        orig_audio = track_audio.copy()

        if sr != 44100:
            raise ValueError("Sample rate should be 44100")
        augmented_gain = None
        print("Now De-limiting : ", track_name, flush=True)

        if args.loudnorm_input_lufs is not None:  # If you want to use loud-normalized input
            track_lufs = meter.integrated_loudness(track_audio.T)
            augmented_gain = args.loudnorm_input_lufs - track_lufs
            track_audio = track_audio * db2linear(augmented_gain, eps=0.0)

        track_audio = (
            torch.as_tensor(track_audio, dtype=torch.float32).unsqueeze(0).to(device)
        )

        estimates = separate_track_with_model(
            args, our_model, device, track_audio, track_name, meter, augmented_gain
        )

        if args.save_mixed_output is not None:
            track_lufs = meter.integrated_loudness(orig_audio.T)
            target_ln = args.save_output_loudnorm if args.save_output_loudnorm is not None else track_lufs
            augmented_gain = target_ln - track_lufs
            orig_audio = orig_audio * db2linear(augmented_gain, eps=0.0)

            mixed_output = orig_audio * args.save_mixed_output + estimates * (
                1 - args.save_mixed_output
            )

            os.makedirs(f"{args.test_output_dir}/{track_name}", exist_ok=True)
            sf.write(
                f"{args.test_output_dir}/{track_name}/{track_name}_mixed.wav",
                mixed_output.T,
                args.data_params.sample_rate,
                subtype="FLOAT",  # the mix has no clip protection, so keep overs instead of clipping
            )


if __name__ == "__main__":
    main()
