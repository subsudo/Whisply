# Whisply Guide

This guide is for development, local testing, packaging, and debugging.

## 1. Project layout

Key files:
- `main.py` - app entry point and runtime wiring
- `settings_dialog.py` - main settings window
- `tray.py` - tray UI and tray menus
- `overlay.py` - recording / loading / transcribing overlay
- `transcriber.py` - backend orchestration and model lifecycle
- `model_store.py` - model download, validation, and install tracking
- `cuda_downloader.py` - on-demand CUDA runtime download
- `installer.iss` - Inno Setup installer
- `release.ps1` - preferred build entry point

## 2. Development setup

Create a virtual environment:

```powershell
py -m venv .venv
```

Install base dependencies:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Install CUDA-related dependencies:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-cuda.txt
```

Optional dependencies:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-openvino.txt
.venv\Scripts\python.exe -m pip install -r requirements-optional.txt
```

## 3. Running Whisply locally

```powershell
.venv\Scripts\python.exe main.py
```

## 4. Core runtime behavior

### Hotkey

Default hotkey:
- `win+ctrl`

Supported input modes:
- hold to speak
- press to speak

### Recording flow

1. Hotkey starts recording.
2. Audio is captured from the selected input device.
3. VAD precheck can reject silence.
4. Model is loaded if needed.
5. Audio is transcribed locally.
6. Result is inserted via clipboard paste.
7. Optional rescue memory keeps the last dictation available for manual copy.

## 5. Storage locations

### Config

- `%APPDATA%\Whisply\config.yaml`

### Models

- `%LOCALAPPDATA%\Whisply\models`

### CUDA runtime downloaded by Whisply

- `%LOCALAPPDATA%\Whisply\cuda_runtime`

### Logs

- `%LOCALAPPDATA%\Whisply\logs`

Typical log files:
- `whisply-YYYYMMDD.log`
- `whisply-fault.log`

## 6. Configuration overview

Important config areas:

### `hotkey`
- `combination`
- `mode`
- `debounce_ms`
- `debug_trace`
- `debug_global`

### `audio`
- `device`
- `sample_rate`
- `channels`
- `vad_enabled`
- `vad_threshold`
- `vad_min_speech_ms`

### `whisper`
- `model`
- `language`
- `backend`
- `compute_type`
- `beam_size`
- `unload_after_idle_sec`
- `download_root`

### `overlay`
- `width`
- `height`
- `monitor_index`
- `opacity`
- `bottom_offset`
- `waveform_style`
- `waveform_gradient_start`
- `waveform_gradient_end`
- `transcribing_delay_ms`
- `transcription_timeout_ms`

### `insertion`
- `restore_clipboard`
- `paste_delay_ms`
- `append_trailing_space`
- `rescue_enabled`
- `rescue_timeout_sec`
- `rescue_never_expire`

### `general`
- `autostart`
- `language_ui`
- `debug_logging`
- `log_to_file`
- `log_dir`
- `log_retention_days`

## 7. Model handling

Supported product models:
- `small`
- `medium`
- `large-v3`
- `large-v3-turbo`

Models can be installed:
- during first-run setup
- from the tray
- by selecting an uninstalled model in settings
- via CLI prefetch

## 8. CUDA handling

Whisply does not bundle all CUDA runtime DLLs in the main installer.

Instead:
- NVIDIA hardware is detected
- CUDA availability is checked
- if required runtime DLLs are missing, Whisply can offer a download
- runtime files are stored in the app-owned `cuda_runtime` folder

This keeps the main installer smaller while still allowing GPU acceleration.

## 9. Logging and debug mode

Standard mode:
- normal operational logging at `INFO`

Debug toggle in settings:
- enables broader `DEBUG` logging for troubleshooting
- intended for temporary use while diagnosing issues

Logs are technical and should stay relatively compact in normal use.

## 10. Building

### Preferred build path

Use `release.ps1`.

Installer build:

```powershell
powershell -ExecutionPolicy Bypass -File .\release.ps1 -Target installer
```

Full release build:

```powershell
powershell -ExecutionPolicy Bypass -File .\release.ps1 -Target all -Clean
```

### Build behavior

- compile-check runs first
- PyInstaller builds `Whisply.exe`
- Inno Setup builds `Whisply-Installer.exe`
- `release.ps1` now includes an automatic fallback via `dist_installer` if Inno cannot write directly to `dist`

## 11. Releases

Recommended GitHub release assets:
- `Whisply-Installer.exe` as the main download

Before publishing a release:
- verify `cuda_manifest.json` contains the real runtime asset URL
- test installer on a clean Windows machine
- test CPU path
- test CUDA path if NVIDIA hardware is available
- confirm first-run setup still behaves correctly

## 12. Scope boundaries

Current product-facing backend support:
- `cpu`
- `cuda`
- `auto`

OpenVINO code remains in the repository for future work but is currently not presented as a primary product path.

