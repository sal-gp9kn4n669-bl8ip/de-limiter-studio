# De-limiter Studio

A native macOS (SwiftUI) application for restoring dynamic range and transient headroom to brickwall-limited audio, built on top of the [**De-limiter**](https://github.com/jeonchangbin49/De-limiter) Sample-wise Gain Inversion (SGI) network (*WASPAA 2023*).

---

## Features

- **Album / Batch Relative Loudness Rebalancing**: Automatically groups queued tracks by folder, anchors the loudest track in each album to your target LUFS, preserves exact track-to-track relative volume differences, and shifts the album down if any restored transient exceeds your True Peak Ceiling (`dBTP`).
- **Core-Volume-Matched Waveform Visualization**: Overlays the original waveform on top of the de-limited waveform matched by sustained **Core Loudness** (ITU-R BS.1770-4 K-weighted 10th–60th percentile energy), visually highlighting restored peaks rising above the original brickwall limiter ceiling and reporting restored peak headroom (`+dB`).
- **Synchronized Gapless A/B Player**: Instant switching between **A: Original** and **B: De-limited** with **Relative Volume Match** so you hear pure transient restoration without loudness bias.
- **Lossless & Apple AAC Output**:
  - **FLAC** (24-bit / 44.1 kHz lossless)
  - **WAV** (24-bit / 44.1 kHz lossless)
  - **AAC (`.m4a`)** via macOS Core Audio `afconvert` Constrained VBR (`-q 127 -s 2`, 96–320 kbps) with gapless `iTunSMPB` metadata and iTunes tag preservation.
- **Hardware Acceleration**: Supports both **CPU** and **Apple Silicon GPU (MPS)** inference.

---

## Download & Install (Pre-Built Release)

The pre-built release (`De-limiter-Studio-v1.0.0-macOS-arm64.zip`) is **100% self-contained**—it includes a standalone Python 3.11 runtime, PyTorch, audio libraries, and the pretrained model weights inside `De-limiter.app`. No Python, Homebrew, or `ffmpeg` installation is required.

1. Download **`De-limiter-Studio-v1.0.0-macOS-arm64.zip`** from the [**Releases**](../../releases) page.
2. Extract the archive and drag **`De-limiter.app`** to `/Applications` (or anywhere you like).
3. Because the app is not notarized with an Apple Developer ID, run this once in Terminal to clear the macOS quarantine flag:
   ```bash
   xattr -dr com.apple.quarantine /Applications/De-limiter.app
   ```
4. Launch **`De-limiter.app`**.

*System Requirements: Apple Silicon Mac (M1 / M2 / M3 / M4) running macOS 14.0 (Sonoma) or later.*

---

## Building from Source

### Prerequisites
- **macOS 14.0+** on Apple Silicon (`arm64`)
- **Xcode Command Line Tools** (`xcode-select --install`) for `swiftc`
- [**uv**](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`) to provision the standalone Python 3.11 runtime

### Build Commands
```bash
# Full build (provisions standalone Python 3.11 + packages into De-limiter.app and compiles SwiftUI app):
./build_app.sh --full

# Fast incremental build (recompiles DelimiterApp.swift and syncs audio_tool.py + De-limiter/):
./build_app.sh

# Create a distributable zip archive (preserves bundle symlinks via ditto):
./build_app.sh --zip
```

---

## Repository Structure

```text
├── DelimiterApp.swift      # Native macOS SwiftUI interface, A/B player & batch pipeline
├── audio_tool.py           # Decoding, ITU-R BS.1770-4 + Core Loudness metering & WAV/FLAC/AAC writer
├── build_app.sh            # Self-contained De-limiter.app builder and packager
├── requirements-app.in     # Top-level Python dependencies for the bundled runtime
├── requirements-app.txt    # Pinned Python lockfile for reproducible builds
└── De-limiter/             # Trimmed inference engine & pretrained weights (all.pth)
```

---

## Attribution & Citation

All credit for the underlying **De-limiter** neural network architecture, Sample-wise Gain Inversion (SGI) methodology, and pretrained model weights (`De-limiter/weight/all.pth`) belongs to **Chang-Bin Jeon** and **Kyogu Lee**:

- **Original Repository**: [https://github.com/jeonchangbin49/De-limiter](https://github.com/jeonchangbin49/De-limiter)
- **Paper**: *"Music De-Limiter Networks via Sample-wise Gain Inversion"*, IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), 2023 ([arXiv:2308.01187](https://arxiv.org/abs/2308.01187)).

```bibtex
@inproceedings{jeon2023music,
  title={Music De-Limiter Networks via Sample-wise Gain Inversion},
  author={Jeon, Chang-Bin and Lee, Kyogu},
  booktitle={2023 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)},
  year={2023},
  organization={IEEE}
}
```

The `Conv-TasNet` filterbank and temporal convolutional masking layers in `De-limiter/models/` are adapted from [**Asteroid**](https://github.com/asteroid-team/asteroid) (MIT License, *Pariente et al., Interspeech 2020*).

---

## License

Released under the [MIT License](LICENSE). See [LICENSE](LICENSE) for full copyright and third-party license notices.

