"""Local browser interface for the room-reconstruction CLI."""
from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import HTMLResponse
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Install web dependencies with: python -m pip install -e '.[web]'") from exc

ALLOWED_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
RESULTS_ROOT = Path(os.environ.get("ROOM_RECONSTRUCTION_RESULTS", "/home/allen/results"))
REPO_ROOT = Path(__file__).resolve().parents[2]

@dataclass
class Job:
    job_id: str
    output: Path
    process: subprocess.Popen[str] | None = None

JOBS: dict[str, Job] = {}
app = FastAPI(title="Room Render", version="0.1.0")

INDEX_HTML = '''
<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Room Render</title><style>
body{font:16px system-ui,sans-serif;max-width:760px;margin:3rem auto;padding:0 1rem;color:#18212b}
main{border:1px solid #d8dee6;border-radius:14px;padding:2rem;box-shadow:0 8px 30px #18212b12}
button{background:#2563eb;color:#fff;border:0;border-radius:8px;padding:.65rem 1rem;cursor:pointer}
button:disabled{opacity:.5}.status{margin-top:1.5rem;padding:1rem;background:#f5f7fa;border-radius:8px;white-space:pre-wrap}
</style></head><body><main><h1>Room Render</h1>
<p>Upload an original-quality room video. Processing runs on your computer's GPU.</p>
<p>Move slowly with physical translation and substantial overlap between views.</p>
<form id="form"><input id="video" type="file" accept=".mov,.mp4,.mkv,.avi,.m4v" required>
<button id="submit">Reconstruct room</button></form><div id="status" class="status" hidden></div></main>
<script>
const form=document.querySelector('#form'),input=document.querySelector('#video'),button=document.querySelector('#submit'),status=document.querySelector('#status');
const show=t=>{status.hidden=false;status.textContent=t};
form.onsubmit=async e=>{e.preventDefault();button.disabled=true;show('Uploading video…');const d=new FormData();d.append('video',input.files[0]);
const r=await fetch('/api/jobs',{method:'POST',body:d});if(!r.ok){show(await r.text());button.disabled=false;return}const j=await r.json();
const poll=async()=>{const s=await (await fetch('/api/jobs/'+j.id)).json();show(s.message);if(s.status==='running')setTimeout(poll,2000);else button.disabled=false};poll()};
</script></body></html>
'''

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML

def _status(job: Job) -> dict[str, str]:
    metadata_path = job.output / "metadata.json"
    metadata: dict[str, object] = {}
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    state = str(metadata.get("status", "running"))
    if job.process and job.process.poll() is not None and state == "running":
        state = "failed"
    if state == "completed":
        message = "Reconstruction completed. Open it with: python reconstruct.py --output " + str(job.output) + " --open"
    elif state == "failed":
        message = "Reconstruction stopped. Check logs/processing.log for details."
    else:
        message = "Reconstruction is running on the local GPU. Camera recovery runs before training."
    return {"id": job.job_id, "status": state, "message": message}

@app.post("/api/jobs")
async def create_job(video: Annotated[UploadFile, File(...)]) -> dict[str, str]:
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Unsupported video format.")
    job_id = secrets.token_hex(6)
    output = RESULTS_ROOT / ("web-" + job_id)
    output.mkdir(parents=True, exist_ok=False)
    input_path = output / ("input" + suffix)
    input_path.write_bytes(await video.read())
    job = Job(job_id=job_id, output=output)
    command = [sys.executable, str(REPO_ROOT / "reconstruct.py"), str(input_path),
               "--output", str(output), "--config", str(REPO_ROOT / "configs" / "low-memory.yml")]
    job.process = subprocess.Popen(command, cwd=REPO_ROOT, text=True)  # noqa: ASYNC220
    JOBS[job_id] = job
    return {"id": job_id}

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, str]:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _status(job)

def main() -> None:
    import uvicorn
    uvicorn.run("room_reconstruction.webapp:app", host="127.0.0.1", port=8000, reload=False)

if __name__ == "__main__":
    main()
