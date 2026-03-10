from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
import logging
import os
import shutil
import sys
import urllib.request
import uuid
import zipfile
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QDialog, QLabel, QProgressBar, QPushButton, QVBoxLayout

from i18n import normalize_ui_language, tr
from paths import get_cuda_runtime_dir


log = logging.getLogger(__name__)
MANIFEST_FILENAME = "cuda_manifest.json"
DOWNLOAD_URL_ENV = "WHISPLY_CUDA_RUNTIME_URL"


class DownloadCancelledError(RuntimeError):
    pass


def _manifest_candidates() -> list[Path]:
    candidates: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / MANIFEST_FILENAME)
    # Installed onefile exe location as an additional robust fallback.
    try:
        exe_dir = Path(sys.executable).resolve().parent
        candidates.append(exe_dir / MANIFEST_FILENAME)
    except Exception:
        pass
    module_dir = Path(__file__).resolve().parent
    candidates.append(module_dir / MANIFEST_FILENAME)
    candidates.append(Path.cwd() / MANIFEST_FILENAME)
    return candidates


def load_cuda_manifest() -> dict[str, Any]:
    for candidate in _manifest_candidates():
        if not candidate.exists():
            continue
        with candidate.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid CUDA manifest format: {candidate}")
        if "files" not in payload or not isinstance(payload["files"], dict):
            raise ValueError(f"Missing 'files' map in CUDA manifest: {candidate}")
        return payload
    raise FileNotFoundError("cuda_manifest.json not found")


def _normalize_expected_hash(value: str) -> str:
    raw = value.strip().lower()
    if raw.startswith("sha256:"):
        raw = raw.split(":", 1)[1]
    if len(raw) != 64 or any(ch not in "0123456789abcdef" for ch in raw):
        raise ValueError(f"Invalid SHA256 hash value '{value}'")
    return raw


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_file_case_insensitive(root: Path, file_name: str) -> Path | None:
    target = file_name.lower()
    for candidate in root.rglob("*"):
        if candidate.is_file() and candidate.name.lower() == target:
            return candidate
    return None


def get_manifest_download_url(manifest: dict[str, Any]) -> str:
    env_override = os.environ.get(DOWNLOAD_URL_ENV, "").strip()
    if env_override:
        return env_override
    return str(manifest.get("download_url", "")).strip()


def verify_cuda_runtime(
    runtime_dir: Path | None = None,
    manifest: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    runtime_dir = runtime_dir or get_cuda_runtime_dir()
    manifest = manifest or load_cuda_manifest()
    files = manifest.get("files", {})
    if not isinstance(files, dict) or not files:
        return False, "manifest_invalid_files"
    if not runtime_dir.is_dir():
        return False, "runtime_missing"

    for file_name, expected_hash_raw in files.items():
        if not isinstance(file_name, str) or not isinstance(expected_hash_raw, str):
            return False, "manifest_invalid_entry"
        try:
            expected_hash = _normalize_expected_hash(expected_hash_raw)
        except ValueError as exc:
            return False, f"manifest_invalid_hash:{file_name}:{exc}"

        target_path = runtime_dir / file_name
        if not target_path.exists():
            return False, f"missing:{file_name}"

        actual_hash = _sha256(target_path)
        if actual_hash != expected_hash:
            return False, f"hash_mismatch:{file_name}"

    return True, "ok"


def is_cuda_runtime_downloaded() -> bool:
    try:
        ok, _ = verify_cuda_runtime()
        return ok
    except Exception:
        return False


def install_cuda_runtime_headless(
    progress_cb: Callable[[int], None] | None = None,
) -> tuple[bool, str]:
    try:
        manifest = load_cuda_manifest()
    except Exception as exc:
        return False, str(exc)

    url = get_manifest_download_url(manifest)
    if not url or "USER/" in url:
        return False, "download_url_not_configured"

    runtime_dir = get_cuda_runtime_dir()
    work_dir = runtime_dir.parent / f"_cuda_runtime_download_{uuid.uuid4().hex[:8]}"
    zip_path = work_dir / "cuda_runtime.zip"
    extract_dir = work_dir / "extracted"
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Whisply/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            total = int(response.headers.get("Content-Length", "0") or "0")
            downloaded = 0
            chunk_size = 1024 * 256
            with zip_path.open("wb") as zip_file:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    zip_file.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb and total > 0:
                        progress_cb(max(1, min(80, int(downloaded * 80 / total))))

        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_dir)
        if progress_cb:
            progress_cb(90)

        _install_runtime_from_extracted(extract_dir, runtime_dir, manifest)
        if progress_cb:
            progress_cb(100)
        return True, "ok"
    except Exception as exc:
        log.exception("Headless CUDA runtime install failed")
        return False, str(exc)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _install_runtime_from_extracted(
    extracted_dir: Path,
    runtime_dir: Path,
    manifest: dict[str, Any],
) -> None:
    files = manifest.get("files", {})
    if not isinstance(files, dict) or not files:
        raise RuntimeError("manifest_invalid_files")

    runtime_tmp = runtime_dir.parent / f"_cuda_runtime_new_{uuid.uuid4().hex[:8]}"
    runtime_backup = runtime_dir.parent / f"_cuda_runtime_backup_{uuid.uuid4().hex[:8]}"
    shutil.rmtree(runtime_tmp, ignore_errors=True)
    runtime_tmp.mkdir(parents=True, exist_ok=True)
    backup_created = False

    try:
        for file_name, expected_hash_raw in files.items():
            if not isinstance(file_name, str) or not isinstance(expected_hash_raw, str):
                raise RuntimeError("manifest_invalid_entry")
            expected_hash = _normalize_expected_hash(expected_hash_raw)

            source = _find_file_case_insensitive(extracted_dir, file_name)
            if source is None:
                raise RuntimeError(f"missing:{file_name}")

            destination = runtime_tmp / file_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

            actual_hash = _sha256(destination)
            if actual_hash != expected_hash:
                raise RuntimeError(f"hash_mismatch:{file_name}")

        ok, reason = verify_cuda_runtime(runtime_tmp, manifest)
        if not ok:
            raise RuntimeError(reason)

        if runtime_dir.exists():
            shutil.rmtree(runtime_backup, ignore_errors=True)
            runtime_dir.replace(runtime_backup)
            backup_created = True

        try:
            runtime_tmp.replace(runtime_dir)
        except Exception:
            if backup_created and runtime_backup.exists() and not runtime_dir.exists():
                runtime_backup.replace(runtime_dir)
            raise

        if backup_created and runtime_backup.exists():
            shutil.rmtree(runtime_backup, ignore_errors=True)
    finally:
        shutil.rmtree(runtime_tmp, ignore_errors=True)
        if runtime_backup.exists():
            shutil.rmtree(runtime_backup, ignore_errors=True)


class DownloadThread(QThread):
    progress = Signal(int)
    finished = Signal(bool, str)

    def __init__(self, manifest: dict[str, Any], runtime_dir: Path) -> None:
        super().__init__()
        self._manifest = manifest
        self._runtime_dir = runtime_dir
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def _check_cancelled(self) -> None:
        if self._cancelled:
            raise DownloadCancelledError("cancelled")

    def run(self) -> None:
        url = get_manifest_download_url(self._manifest)
        if not url or "USER/" in url:
            self.finished.emit(False, "download_url_not_configured")
            return

        work_dir = self._runtime_dir.parent / f"_cuda_runtime_download_{uuid.uuid4().hex[:8]}"
        zip_path = work_dir / "cuda_runtime.zip"
        extract_dir = work_dir / "extracted"

        shutil.rmtree(work_dir, ignore_errors=True)
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._check_cancelled()
            request = urllib.request.Request(url, headers={"User-Agent": "Whisply/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                total = int(response.headers.get("Content-Length", "0") or "0")
                downloaded = 0
                chunk_size = 1024 * 256
                with zip_path.open("wb") as zip_file:
                    while True:
                        self._check_cancelled()
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        zip_file.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            self.progress.emit(max(1, min(80, int(downloaded * 80 / total))))

            self._check_cancelled()
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(extract_dir)
            self.progress.emit(90)

            self._check_cancelled()
            _install_runtime_from_extracted(extract_dir, self._runtime_dir, self._manifest)
            self.progress.emit(100)
            self.finished.emit(True, "ok")
        except DownloadCancelledError:
            self.finished.emit(False, "cancelled")
        except Exception as exc:
            log.exception("CUDA runtime download/install failed")
            self.finished.emit(False, str(exc))
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)


class CudaDownloadDialog(QDialog):
    def __init__(self, parent=None, language: str = "de") -> None:
        super().__init__(parent)
        self._lang = normalize_ui_language(language)
        self._manifest = load_cuda_manifest()
        self._thread: DownloadThread | None = None
        self._success = False
        self._last_message = ""
        self._size_mb = int(self._manifest.get("size_mb", 0) or 0)

        self.setWindowTitle(self._t("cuda_download_title"))
        self.setModal(True)
        self.resize(460, 210)

        layout = QVBoxLayout(self)
        self._label = QLabel(self._t("cuda_download_prompt", size=self._size_mb))
        self._label.setWordWrap(True)
        layout.addWidget(self._label)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.hide()
        layout.addWidget(self._progress)

        self._download_button = QPushButton(self._t("cuda_download_btn"))
        self._download_button.clicked.connect(self._start_download)
        layout.addWidget(self._download_button)

        self._cancel_button = QPushButton(self._t("cuda_skip_btn"))
        self._cancel_button.clicked.connect(self.reject)
        layout.addWidget(self._cancel_button)

    def _t(self, key: str, **kwargs: object) -> str:
        return tr(self._lang, key, **kwargs)

    def _start_download(self) -> None:
        self._download_button.setEnabled(False)
        self._cancel_button.setText(self._t("cuda_cancel_btn"))
        self._progress.setValue(0)
        self._progress.show()
        self._label.setText(self._t("cuda_download_in_progress"))

        self._thread = DownloadThread(self._manifest, get_cuda_runtime_dir())
        self._thread.progress.connect(self._progress.setValue)
        self._thread.finished.connect(self._on_finished)
        self._thread.start()

    def _on_finished(self, success: bool, message: str) -> None:
        self._success = success
        self._last_message = message
        self._progress.hide()
        self._download_button.hide()

        if success:
            self._label.setText(self._t("cuda_download_success"))
            self._cancel_button.setText("OK")
            try:
                self._cancel_button.clicked.disconnect()
            except Exception:
                pass
            self._cancel_button.clicked.connect(self.accept)
            return

        if message == "cancelled":
            self._label.setText(self._t("cuda_download_cancelled"))
        elif message == "download_url_not_configured":
            self._label.setText(self._t("cuda_download_url_missing"))
        elif "hash_mismatch" in message:
            self._label.setText(self._t("cuda_hash_mismatch"))
        else:
            self._label.setText(self._t("cuda_download_failed", error=message))

        self._cancel_button.setText("OK")
        try:
            self._cancel_button.clicked.disconnect()
        except Exception:
            pass
        self._cancel_button.clicked.connect(self.reject)

    def was_successful(self) -> bool:
        return self._success

    def last_message(self) -> str:
        return self._last_message

    def closeEvent(self, event) -> None:  # noqa: ANN001
        if self._thread and self._thread.isRunning():
            self._thread.cancel()
            self._thread.wait(3000)
        super().closeEvent(event)
