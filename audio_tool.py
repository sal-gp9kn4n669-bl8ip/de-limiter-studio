#!/usr/bin/env python3
"""Audio helper for De-limiter Studio: decoding, loudness metering and writing (WAV, FLAC, AAC).

It runs with the same Python as inference.py and replaces ffmpeg, so the app needs no external tools:

  stage SRC DST                  decode any supported input to the 44.1 kHz stereo float WAV the model expects
  measure FILE [FILE ...]        integrated & core loudness (LUFS), 4x-oversampled true peak (dBTP), and waveform
  write SRC DST --format wav|flac|m4a [--bitrate KBPS] [--gain DB] [--tags-from FILE]
                                 apply gain and write 24-bit WAV/FLAC or AAC (.m4a via afconvert), copying basic tags

Every command prints one line "RESULT <json>" on success (loudness values are null when not measurable)
and exits non-zero with a message on failure.

Decoding uses libsndfile (WAV, AIFF, FLAC, MP3, Ogg Vorbis/Opus) and falls back to macOS's built-in
afconvert/afinfo for formats it can't read (M4A: AAC/ALAC).
"""
import argparse
import json
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
from numpy.lib.stride_tricks import sliding_window_view

import numpy as np
import pyloudnorm
import scipy.signal
import soundfile as sf
import soxr

MODEL_RATE = 44100
WAVEFORM_BUCKETS = 480
TAG_FIELDS = ("title", "artist", "album", "tracknumber", "date", "genre")
# Keys in Core Audio's info dictionary (as printed by `afinfo -i`) -> libsndfile tag names
CORE_AUDIO_TAGS = {
    "title": "title",
    "artist": "artist",
    "album": "album",
    "track number": "tracknumber",
    "year": "date",
    "recorded date": "date",
    "genre": "genre",
}


def read_audio(path):
    """Returns (float32 array shaped frames x channels, sample rate)."""
    try:
        data, rate = sf.read(path, dtype="float32", always_2d=True)
        return data, rate
    except RuntimeError:  # libsndfile can't open it (e.g. M4A): let Core Audio decode it
        pass
    with tempfile.TemporaryDirectory() as tmp:
        decoded = os.path.join(tmp, "decoded.wav")
        result = subprocess.run(
            ["/usr/bin/afconvert", "-f", "WAVE", "-d", "LEF32", path, decoded],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"unsupported or unreadable audio file ({result.stderr.strip() or 'afconvert failed'})")
        data, rate = sf.read(decoded, dtype="float32", always_2d=True)
        return data, rate


def to_stereo(data):
    """The model is stereo-only: duplicate mono, fold multichannel down to L/R."""
    channels = data.shape[1]
    if channels == 2:
        return data
    if channels == 1:
        return np.repeat(data, 2, axis=1)
    left, right = data[:, 0].copy(), data[:, 1].copy()
    if channels >= 3:  # front centre at -3 dB into both sides
        left += 0.7071 * data[:, 2]
        right += 0.7071 * data[:, 2]
    if channels >= 6:  # 5.1 order FL FR FC LFE BL BR: surrounds at -3 dB
        left += 0.7071 * data[:, 4]
        right += 0.7071 * data[:, 5]
    return np.stack([left, right], axis=1)


def true_peak(data, rate, block_seconds=10):
    """Peak of the 4x-oversampled signal, streamed in blocks so long tracks don't need 4x the memory."""
    stream = soxr.ResampleStream(rate, rate * 4, data.shape[1], dtype="float32", quality="HQ")
    peak, block = 0.0, rate * block_seconds
    for start in range(0, len(data), block):
        chunk = np.ascontiguousarray(data[start:start + block])
        oversampled = stream.resample_chunk(chunk, last=start + block >= len(data))
        if oversampled.size:
            peak = max(peak, float(np.abs(oversampled).max()))
    return peak


def core_loudness(data, rate):
    """Sustained 'core' loudness (LUFS): 10th-60th percentile of active 10 ms K-weighted micro-frames
    inside each 400 ms window with ITU-R BS.1770-4 two-stage gating, ignoring transient spikes."""
    f0_1, gain_1, q_1 = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k1 = math.tan(math.pi * f0_1 / rate)
    vh = 10.0 ** (gain_1 / 20.0)
    vb = vh ** 0.4996667741545416
    k1_2 = k1 * k1
    a0_1 = 1.0 + k1 / q_1 + k1_2
    b1 = [(vh + vb * k1 / q_1 + k1_2) / a0_1, 2.0 * (k1_2 - vh) / a0_1, (vh - vb * k1 / q_1 + k1_2) / a0_1]
    a1 = [1.0, 2.0 * (k1_2 - 1.0) / a0_1, (1.0 - k1 / q_1 + k1_2) / a0_1]

    f0_2, q_2 = 38.13547087602444, 0.5003270373238773
    k2 = math.tan(math.pi * f0_2 / rate)
    k2_2 = k2 * k2
    a0_2 = 1.0 + k2 / q_2 + k2_2
    b2 = [1.0 / a0_2, -2.0 / a0_2, 1.0 / a0_2]
    a2 = [1.0, 2.0 * (k2_2 - 1.0) / a0_2, (1.0 - k2 / q_2 + k2_2) / a0_2]

    sos = np.array([b1 + a1, b2 + a2], dtype=np.float32)
    micro_step = max(1, round(0.010 * rate))
    n_micro = len(data) // micro_step
    if n_micro < 40:
        return -math.inf

    k_weighted = scipy.signal.sosfilt(sos, data[:n_micro * micro_step], axis=0)
    micro_energy = np.mean(
        k_weighted.reshape(n_micro, micro_step, data.shape[1]).astype(np.float64) ** 2,
        axis=1,
    ).sum(axis=1)
    windows = sliding_window_view(micro_energy, 40)[::10]
    win_mean = windows.mean(axis=1)
    sorted_w = np.sort(windows, axis=1)

    core_energies = np.empty(len(windows), dtype=np.float64)
    for i in range(len(windows)):
        m = win_mean[i]
        if m <= 1e-12:
            core_energies[i] = m
            continue
        row = sorted_w[i]
        idx0 = int(np.searchsorted(row, m * 0.0316))
        active_cnt = 40 - idx0
        if active_cnt < 4:
            core_energies[i] = m
        else:
            lo = idx0 + int(active_cnt * 0.10)
            hi = max(lo + 1, idx0 + math.ceil(active_cnt * 0.60))
            core_energies[i] = row[lo:hi].mean()

    abs_gate = 10.0 ** ((-70.0 + 0.691) / 10.0)
    abs_gated = core_energies[core_energies > abs_gate]
    if len(abs_gated) == 0:
        return -math.inf
    rel_lufs = -0.691 + 10.0 * math.log10(abs_gated.mean()) - 10.0
    rel_gate = 10.0 ** ((rel_lufs + 0.691) / 10.0)
    rel_gated = abs_gated[abs_gated > rel_gate]
    if len(rel_gated) == 0:
        return -0.691 + 10.0 * math.log10(abs_gated.mean())
    return -0.691 + 10.0 * math.log10(rel_gated.mean())


def waveform_peaks(data, num_buckets=WAVEFORM_BUCKETS):
    """Downsamples stereo float audio into `num_buckets` peak amplitude values in [0, 1]."""
    n = len(data)
    if n == 0 or num_buckets <= 0:
        return []
    actual = min(num_buckets, n)
    edges = np.linspace(0, n, actual + 1, dtype=np.int64)
    abs_max_ch = np.max(np.abs(data), axis=1)
    peaks = np.maximum.reduceat(abs_max_ch, edges[:-1])
    return [round(float(p), 4) for p in peaks]


def loudness(data, rate):
    try:
        lufs = pyloudnorm.Meter(rate).integrated_loudness(data)
    except ValueError:  # shorter than one 400 ms gating block
        lufs = -math.inf
    core = core_loudness(data, rate)
    peak = true_peak(data, rate)
    peak_db = 20 * math.log10(peak) if peak > 0 else -math.inf
    return {
        "lufs": round(lufs, 3) if math.isfinite(lufs) else None,
        "core_lufs": round(core, 3) if math.isfinite(core) else None,
        "peak": round(peak_db, 3) if math.isfinite(peak_db) else None,
        "peaks": waveform_peaks(data),
    }


def read_tags(path):
    """Basic tags (title, artist, album, track, date, genre) from the source file."""
    try:
        with sf.SoundFile(path) as source:
            metadata = source.copy_metadata()
        return {key: value for key, value in metadata.items() if key in TAG_FIELDS and value}
    except RuntimeError:
        pass
    # libsndfile can't open M4A; Core Audio's afinfo prints its tags as an "info dictionary"
    info = subprocess.run(["/usr/bin/afinfo", "-i", path], capture_output=True, text=True).stdout
    tags, in_dictionary = {}, False
    for line in info.splitlines():
        if line.strip() == "info dictionary:":
            in_dictionary = True
            continue
        if in_dictionary:
            match = re.match(r'^\s+(.+?)\s*:\s*"(.*)"\s*$', line)
            if not match:
                break
            key = CORE_AUDIO_TAGS.get(match.group(1).strip())
            if key and match.group(2) and key not in tags:
                tags[key] = match.group(2)
    return tags


def stamp_m4a_tags(path, tags):
    """Inserts basic iTunes metadata atoms into afconvert's moov.udta.meta.ilst box in-place,
    shrinking the adjacent padding `free` box so `mdat` offsets and `iTunSMPB` stay untouched."""
    if not tags:
        return
    atom_map = {
        "title": b"\xa9nam",
        "artist": b"\xa9ART",
        "album": b"\xa9alb",
        "date": b"\xa9day",
        "genre": b"\xa9gen",
    }
    extra = b""
    for key, fourcc in atom_map.items():
        val = tags.get(key)
        if not val:
            continue
        payload = str(val).encode("utf-8")
        data_box = struct.pack(">I4sII", 16 + len(payload), b"data", 1, 0) + payload
        atom = struct.pack(">I4s", 8 + len(data_box), fourcc) + data_box
        extra += atom

    if tags.get("tracknumber"):
        parts = str(tags["tracknumber"]).split("/")
        try:
            trk = int(parts[0].strip())
            tot = int(parts[1].strip()) if len(parts) > 1 else 0
            payload = struct.pack(">HHHH", 0, trk, tot, 0)
            data_box = struct.pack(">I4sII", 16 + len(payload), b"data", 0, 0) + payload
            extra += struct.pack(">I4s", 8 + len(data_box), b"trkn") + data_box
        except ValueError:
            pass

    if tags.get("albumversion"):
        mean_b = b"com.apple.iTunes"
        name_b = b"albumversion"
        val_b = str(tags["albumversion"]).encode("utf-8")
        mean_box = struct.pack(">I4sI", 12 + len(mean_b), b"mean", 0) + mean_b
        name_box = struct.pack(">I4sI", 12 + len(name_b), b"name", 0) + name_b
        data_box = struct.pack(">I4sII", 16 + len(val_b), b"data", 1, 0) + val_b
        inner = mean_box + name_box + data_box
        extra += struct.pack(">I4s", 8 + len(inner), b"----") + inner

    if not extra:
        return

    with open(path, "rb") as f:
        buf = bytearray(f.read())

    pos = 0
    moov_pos = None
    while pos + 8 <= len(buf):
        sz, fcc = struct.unpack(">I4s", buf[pos:pos + 8])
        if sz < 8:
            break
        if fcc == b"moov":
            moov_pos = pos
            break
        pos += sz
    if moov_pos is None:
        return

    moov_sz = struct.unpack(">I", buf[moov_pos:moov_pos + 4])[0]
    free_pos = moov_pos + moov_sz
    if free_pos + 8 > len(buf):
        return
    free_sz, free_fcc = struct.unpack(">I4s", buf[free_pos:free_pos + 8])
    if free_fcc != b"free" or free_sz - len(extra) < 8:
        return

    udta_pos = buf.find(b"udta", moov_pos + 8, free_pos) - 4
    meta_pos = buf.find(b"meta", udta_pos + 8, free_pos) - 4
    ilst_pos = buf.find(b"ilst", meta_pos + 12, free_pos) - 4
    if min(udta_pos, meta_pos, ilst_pos) < 0:
        return

    for box_pos in (moov_pos, udta_pos, meta_pos, ilst_pos):
        old_sz = struct.unpack(">I", buf[box_pos:box_pos + 4])[0]
        struct.pack_into(">I", buf, box_pos, old_sz + len(extra))

    new_free_sz = free_sz - len(extra)
    new_free = struct.pack(">I4s", new_free_sz, b"free") + b"\x00" * (new_free_sz - 8)
    buf[free_pos:free_pos + free_sz] = extra + new_free
    with open(path, "wb") as f:
        f.write(buf)


def stamp_flac_vorbis_comment(path, key, value):
    """Appends a key=value Vorbis comment to a FLAC file's VORBIS_COMMENT metadata block (type 4)."""
    with open(path, "rb") as f:
        buf = bytearray(f.read())
    if buf[:4] != b"fLaC":
        return
    pos = 4
    while pos + 4 <= len(buf):
        header = buf[pos]
        is_last = (header & 0x80) != 0
        btype = header & 0x7F
        length = int.from_bytes(buf[pos + 1:pos + 4], "big")
        block_start = pos + 4
        block_end = block_start + length
        if btype == 4:
            vlen = struct.unpack_from("<I", buf, block_start)[0]
            count_pos = block_start + 4 + vlen
            count = struct.unpack_from("<I", buf, count_pos)[0]
            entry = f"{key}={value}".encode("utf-8")
            packed_entry = struct.pack("<I", len(entry)) + entry
            struct.pack_into("<I", buf, count_pos, count + 1)
            new_len = length + len(packed_entry)
            if new_len < (1 << 24):
                buf[pos + 1:pos + 4] = new_len.to_bytes(3, "big")
                buf[block_end:block_end] = packed_entry
                with open(path, "wb") as f:
                    f.write(buf)
            return
        pos = block_end
        if is_last:
            break


def stage(args):
    data, rate = read_audio(args.src)
    data = to_stereo(data)
    if rate != MODEL_RATE:
        data = soxr.resample(data, rate, MODEL_RATE, quality="VHQ").astype(np.float32, copy=False)
    sf.write(args.dst, data, MODEL_RATE, subtype="FLOAT")
    return loudness(data, MODEL_RATE)


def measure(args):
    results = []
    for path in args.files:
        try:
            data, rate = read_audio(path)
            data = to_stereo(data)
            results.append(loudness(data, rate))
        except Exception as error:  # one unreadable file shouldn't hide the others' results
            print(f"measure failed for {path}: {error}", file=sys.stderr)
            results.append({"lufs": None, "core_lufs": None, "peak": None, "peaks": []})
    return results


def write(args):
    data, rate = sf.read(args.src, dtype="float32", always_2d=True)
    if args.gain:
        data = data * np.float32(10 ** (args.gain / 20))
    np.clip(data, -1.0, 1.0, out=data)  # integer PCM can't hold overs; clip rather than wrap around
    tags = read_tags(args.tags_from) if args.tags_from else {}

    if args.format == "m4a":
        with tempfile.TemporaryDirectory() as tmp:
            gained_wav = os.path.join(tmp, "gained.wav")
            sf.write(gained_wav, data, rate, subtype="PCM_24", format="WAV")
            bitrate_bps = str(int(args.bitrate) * 1000)
            res = subprocess.run(
                [
                    "/usr/bin/afconvert",
                    "-f", "m4af",
                    "-d", "aac",
                    "-b", bitrate_bps,
                    "-q", "127",
                    "-s", "2",
                    gained_wav,
                    args.dst,
                ],
                capture_output=True,
                text=True,
            )
            if res.returncode != 0:
                raise RuntimeError(f"afconvert failed: {res.stderr.strip() or res.stdout.strip()}")
        tags["albumversion"] = "Delimited"
        stamp_m4a_tags(args.dst, tags)
    else:
        file_format = {"wav": "WAV", "flac": "FLAC"}[args.format]
        with sf.SoundFile(args.dst, "w", samplerate=rate, channels=data.shape[1],
                          subtype="PCM_24", format=file_format) as out:
            for key, value in tags.items():
                setattr(out, key, value)  # string metadata has to be set before any audio is written
            out.write(data)
        if args.format == "flac":
            stamp_flac_vorbis_comment(args.dst, "albumversion", "Delimited")
    return loudness(data, rate)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    stage_cmd = commands.add_parser("stage", help="decode to a 44.1 kHz stereo float WAV and measure it")
    stage_cmd.add_argument("src")
    stage_cmd.add_argument("dst")
    stage_cmd.set_defaults(run=stage)

    measure_cmd = commands.add_parser("measure", help="measure loudness and true peak")
    measure_cmd.add_argument("files", nargs="+")
    measure_cmd.set_defaults(run=measure)

    write_cmd = commands.add_parser("write", help="apply gain and write 24-bit WAV/FLAC or AAC (.m4a)")
    write_cmd.add_argument("src")
    write_cmd.add_argument("dst")
    write_cmd.add_argument("--format", choices=("wav", "flac", "m4a"), required=True)
    write_cmd.add_argument("--bitrate", type=int, default=256, help="AAC bitrate in kbps (default: 256)")
    write_cmd.add_argument("--gain", type=float, default=0.0, help="gain in dB")
    write_cmd.add_argument("--tags-from", help="copy basic tags from this file")
    write_cmd.set_defaults(run=write)

    args = parser.parse_args()
    try:
        result = args.run(args)
    except Exception as error:
        sys.exit(f"{args.command} failed: {error}")
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
