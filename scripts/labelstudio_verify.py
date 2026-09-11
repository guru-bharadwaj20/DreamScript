"""Phase 2.2.1 - prove that Label Studio actually loads this project.

A labelling config that merely parses is not evidence of anything. This script starts a real
Label Studio server against a throwaway database, creates a project from
`labeling/label_studio/config.xml`, registers the repository as a local-files source, syncs it,
imports the pre-annotated tasks built by `src.ir.labelstudio`, and finally fetches one of the
images over HTTP. If that returns JPEG bytes, the project loads images - which is what
contributing.md 2.2.1 asks for.

Three things were learned making this work, and they are why the script is shaped this way:

1. **First start is slow.** Label Studio runs its migrations on first boot; on this machine
   that took over three minutes, so the readiness timeout is generous.
2. **Legacy API tokens are off by default** in 1.23. The script enables them through the
   Django ORM once the database exists, because there is no API to do it without a token.
3. **`/data/local-files/` serves nothing by default.** The document-root environment variable
   is not enough: a file is only served if a `LocalFilesImportStorage` covers it. Registering
   one is the supported route and produces exactly the URL form `src.ir.labelstudio.image_url`
   already generates.

Label Studio pins its own dependency versions, so it lives in a separate virtualenv and this
runs under *that* interpreter:

    .venv-labelstudio/Scripts/python scripts/labelstudio_verify.py

Everything it creates goes in a temporary directory and is deleted afterwards.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "labeling" / "label_studio" / "config.xml"
TASKS = ROOT / "labeling" / "tasks" / "hdbpmn.json"
IMAGE_DIR = ROOT / "data" / "raw" / "hdbpmn" / "data" / "images" / "ex00"

USERNAME = "verify@dreamscript.local"
PASSWORD = "verify-dreamscript-2026"
TOKEN = "dreamscript-verify-token-0000000000000000"

#: Migrations on a cold database took 200s here; three times that is the timeout.
BOOT_TIMEOUT = 600.0

ENABLE_TOKENS = """
import django
django.setup()
from organizations.models import Organization
from jwt_auth.models import JWTSettings
from users.models import User
from rest_framework.authtoken.models import Token
for org in Organization.objects.all():
    settings, _ = JWTSettings.objects.get_or_create(organization=org)
    settings.legacy_api_tokens_enabled = True
    settings.api_tokens_enabled = True
    settings.save()
for user in User.objects.all():
    Token.objects.get_or_create(user=user)
print("tokens enabled")
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def request(url, *, data=None, method="GET", raw=False):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Authorization", f"Token {TOKEN}")
    if body:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            payload = resp.read()
            return resp.status, (payload if raw else json.loads(payload or b"null"))
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()[:400]


def wait_for(url: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5):
                return True
        except urllib.error.HTTPError:
            return True  # answering at all means it is up
        except Exception:
            time.sleep(3)
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep", action="store_true", help="do not delete the temporary instance")
    args = ap.parse_args(argv)

    if not TASKS.is_file():
        print("build the tasks first:")
        print("  python -m src.ir.labelstudio --tasks data/processed/ir/hdbpmn")
        return 1

    workdir = Path(tempfile.mkdtemp(prefix="ls-verify-"))
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    site_packages = Path(sys.executable).parents[1] / "Lib" / "site-packages" / "label_studio"

    env = dict(os.environ)
    env.update(
        {
            "LABEL_STUDIO_BASE_DATA_DIR": str(workdir),
            "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED": "true",
            "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT": str(ROOT),
            "LABEL_STUDIO_USERNAME": USERNAME,
            "LABEL_STUDIO_PASSWORD": PASSWORD,
            "LABEL_STUDIO_USER_TOKEN": TOKEN,
            "COLLECT_ANALYTICS": "false",
            "DO_NOT_TRACK": "1",
        }
    )

    checks: dict[str, bool] = {}
    detail: dict[str, object] = {}
    log_path = workdir / "server.log"
    log = log_path.open("w", encoding="utf-8")
    server = subprocess.Popen(
        [sys.executable, "-m", "label_studio.server", "start", "--port", str(port), "--no-browser"],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        cwd=str(workdir),
    )

    try:
        checks["server_started"] = wait_for(f"{base}/health", BOOT_TIMEOUT)
        if not checks["server_started"]:
            raise RuntimeError(f"server never answered; see {log_path}")

        ormenv = dict(env)
        ormenv["DJANGO_SETTINGS_MODULE"] = "label_studio.core.settings.label_studio"
        ormenv["PYTHONPATH"] = str(site_packages)
        orm = subprocess.run(
            [sys.executable, "-c", ENABLE_TOKENS],
            env=ormenv,
            capture_output=True,
            text=True,
            cwd=str(workdir),
        )
        checks["legacy_tokens_enabled"] = "tokens enabled" in orm.stdout

        status, project = request(
            f"{base}/api/projects/",
            data={
                "title": "DreamScript IR verification",
                "label_config": CONFIG.read_text(encoding="utf-8"),
            },
            method="POST",
        )
        checks["project_created_with_our_config"] = status in (200, 201)
        if not checks["project_created_with_our_config"]:
            raise RuntimeError(f"project creation failed: {status} {project}")
        project_id = project["id"]

        status, storage = request(
            f"{base}/api/storages/localfiles/",
            data={
                "project": project_id,
                "title": "repository images",
                "path": str(IMAGE_DIR),
                "regex_filter": r".*\.(jpg|jpeg|png)",
                "use_blob_urls": True,
            },
            method="POST",
        )
        checks["local_files_storage_registered"] = status in (200, 201)
        if checks["local_files_storage_registered"]:
            code, _ = request(
                f"{base}/api/storages/localfiles/{storage['id']}/sync", data={}, method="POST"
            )
            checks["storage_synced"] = code == 200

        payload = json.loads(TASKS.read_text(encoding="utf-8"))
        status, imported = request(
            f"{base}/api/projects/{project_id}/import", data=payload, method="POST"
        )
        checks["pre_annotated_tasks_imported"] = (
            status in (200, 201) and imported.get("task_count", 0) > 0
        )
        checks["predictions_survived_the_import"] = imported.get("prediction_count", 0) > 0
        detail["import"] = imported

        status, listed = request(f"{base}/api/tasks/?project={project_id}&page_size=500")
        rows = listed.get("tasks", listed) if isinstance(listed, dict) else listed
        checks["tasks_listed"] = bool(rows)
        detail["tasks"] = len(rows)

        refs = [
            row["data"]["image"]
            for row in rows
            if str(row.get("data", {}).get("image", "")).startswith("/data/local-files")
        ]
        checks["tasks_reference_local_images"] = bool(refs)
        status, blob = request(f"{base}{refs[0]}", raw=True)
        checks["image_served_over_http"] = status == 200 and len(blob) > 1000
        checks["image_bytes_are_an_image"] = isinstance(blob, bytes) and blob[:3] in (
            b"\xff\xd8\xff",
            b"\x89PN",
        )
        detail["image"] = {"url": refs[0], "bytes": len(blob)}
    except Exception as exc:  # noqa: BLE001 - the message is the result
        print(f"  ERROR  {exc}", file=sys.stderr)
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()
        if args.keep:
            print(f"instance kept at {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)

    print(json.dumps(detail, indent=2))
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if checks and all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
