"""Google Drive integration — sync files between shared Drive folders and local cache.

Uses a service-account credential for unattended access.
Requires ``google-api-python-client`` and ``google-auth``.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from src.config import Settings
from src.logging_utils import get_logger
from src.utils.files import IMAGE_EXTENSIONS

log = get_logger("gdrive")

# Google Drive MIME types for folders
_FOLDER_MIME = "application/vnd.google-apps.folder"

# Expected sub-folders inside the shared root
DRIVE_SUBFOLDERS = ("raw", "parsed", "styled", "display", "archive", "logs")


def _build_service(settings: Settings) -> Any:
    """Authenticate with a service-account key and return a Drive service object."""
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    key_path = settings.resolve_path(Path(settings.gdrive_service_account_key))
    if not key_path.exists():
        raise FileNotFoundError(
            f"Service-account key not found at {key_path}. "
            "Set GDRIVE_SERVICE_ACCOUNT_KEY in your .env file."
        )

    creds = service_account.Credentials.from_service_account_file(
        str(key_path),
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _find_subfolder_id(service: Any, parent_id: str, name: str) -> str | None:
    """Return the folder ID of *name* inside *parent_id*, or ``None``."""
    q = (
        f"'{parent_id}' in parents and mimeType='{_FOLDER_MIME}' "
        f"and name='{name}' and trashed=false"
    )
    resp = service.files().list(q=q, fields="files(id, name)", pageSize=1).execute()
    files = resp.get("files", [])
    return files[0]["id"] if files else None


def _list_image_files(service: Any, folder_id: str) -> list[dict[str, str]]:
    """List image files in a Drive folder (id, name, modifiedTime)."""
    q = f"'{folder_id}' in parents and trashed=false"
    results: list[dict[str, str]] = []
    page_token: str | None = None

    while True:
        resp = (
            service.files()
            .list(
                q=q,
                fields="nextPageToken, files(id, name, modifiedTime, mimeType)",
                pageSize=100,
                pageToken=page_token,
            )
            .execute()
        )
        for f in resp.get("files", []):
            ext = Path(f["name"]).suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                results.append(f)
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return results


def _download_file(service: Any, file_id: str, dest: Path) -> None:
    """Download a Drive file to *dest*."""
    from googleapiclient.http import MediaIoBaseDownload

    request = service.files().get_media(fileId=file_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def _upload_file(service: Any, local_path: Path, folder_id: str) -> str:
    """Upload *local_path* into *folder_id*. Returns the new file ID."""
    from googleapiclient.http import MediaFileUpload

    media = MediaFileUpload(str(local_path), resumable=True)
    metadata: dict[str, Any] = {"name": local_path.name, "parents": [folder_id]}
    created = (
        service.files().create(body=metadata, media_body=media, fields="id").execute()
    )
    return created["id"]


# ── Public API ───────────────────────────────────────────────────────────────


def sync_folder(
    settings: Settings,
    subfolder: str = "raw",
) -> list[Path]:
    """Download new/updated images from ``<root>/<subfolder>`` into local cache.

    Returns list of newly downloaded local paths.
    """
    if not settings.gdrive_root_folder_id:
        log.warning("GDRIVE_ROOT_FOLDER_ID not set — skipping sync")
        return []

    service = _build_service(settings)
    folder_id = _find_subfolder_id(service, settings.gdrive_root_folder_id, subfolder)
    if not folder_id:
        log.warning("Drive subfolder '%s' not found under root", subfolder)
        return []

    remote_files = _list_image_files(service, folder_id)
    cache_dir = settings.resolve_path(settings.local_cache_dir) / subfolder
    cache_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []
    existing_names = {p.name for p in cache_dir.iterdir()}

    for rf in remote_files:
        if rf["name"] not in existing_names:
            dest = cache_dir / rf["name"]
            log.info("Downloading %s", rf["name"])
            _download_file(service, rf["id"], dest)
            downloaded.append(dest)

    log.info("Sync complete for '%s' — %d new file(s)", subfolder, len(downloaded))
    return downloaded


def sync_all(settings: Settings) -> dict[str, list[Path]]:
    """Sync ``raw`` and ``parsed`` folders."""
    return {
        "raw": sync_folder(settings, "raw"),
        "parsed": sync_folder(settings, "parsed"),
    }


def upload_to_drive(
    settings: Settings,
    local_path: Path,
    subfolder: str = "styled",
) -> str | None:
    """Upload a local file to the given Drive subfolder. Returns file ID or None."""
    if not settings.gdrive_root_folder_id:
        log.warning("GDRIVE_ROOT_FOLDER_ID not set — skipping upload")
        return None

    service = _build_service(settings)
    folder_id = _find_subfolder_id(service, settings.gdrive_root_folder_id, subfolder)
    if not folder_id:
        log.warning("Drive subfolder '%s' not found", subfolder)
        return None

    file_id = _upload_file(service, local_path, folder_id)
    log.info("Uploaded %s → Drive/%s (id=%s)", local_path.name, subfolder, file_id)
    return file_id
