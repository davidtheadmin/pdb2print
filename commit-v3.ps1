# One-shot: clear stale git locks, commit the v3 session, push.
# Run from the repo root in PowerShell.  Delete this file afterwards.

$ErrorActionPreference = "Stop"

function Assert-Git($what) {
    # git failures are exit codes, not PowerShell exceptions, so they have to be
    # checked explicitly or the script cheerfully reports success after failing.
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n$what failed (exit $LASTEXITCODE). Stopping." -ForegroundColor Red
        exit 1
    }
}

# --- 1. Refuse to touch locks if a git process is genuinely running ----------
$running = Get-Process git, git-remote-https -ErrorAction SilentlyContinue
if ($running) {
    Write-Host "A git process is still running:" -ForegroundColor Red
    $running | Format-Table Id, ProcessName, StartTime -AutoSize
    Write-Host "Close it (or: Stop-Process -Id <id>) and re-run this script." -ForegroundColor Red
    exit 1
}

# --- 2. Clear every stale lock, not just index.lock -------------------------
# A crashed git leaves locks in several places: .git\index.lock, .git\HEAD.lock,
# .git\config.lock, .git\refs\heads\<branch>.lock, .git\logs\...
$locks = Get-ChildItem -Path .git -Filter *.lock -Recurse -Force -ErrorAction SilentlyContinue
if ($locks) {
    Write-Host "Removing $($locks.Count) stale lock file(s):" -ForegroundColor Yellow
    foreach ($l in $locks) {
        Write-Host "  $($l.FullName.Substring($PWD.Path.Length + 1))"
        Remove-Item $l.FullName -Force
    }
} else {
    Write-Host "No stale locks found." -ForegroundColor Green
}

# --- 3. Keep the reverted experiments out of this commit --------------------
# See HANDOVER.md section 5 for what is in the patch and what is worth reusing.
if (Test-Path experiments-to-reapply.patch) {
    New-Item -ItemType Directory -Force -Path ..\pdb2print-archive | Out-Null
    Move-Item experiments-to-reapply.patch ..\pdb2print-archive\ -Force
    Write-Host "Moved experiments-to-reapply.patch to ..\pdb2print-archive\" -ForegroundColor Yellow
}

# --- 4. Commit --------------------------------------------------------------
$msg = @'
v3: scored magnet placement, flush socket, press-fit bores

Joinery rework - magnets and bridges now share one geometry path:
- Flush socket (default on): a flat-ended collar on each part so the two
  halves meet on one clean disc instead of two ragged surfaces.
- Press-fit bores: oversize diameter and depth with a 45 deg lead-in, since
  an FDM hole cut to a magnet's nominal size will not accept it.
- Two-stage seat scoring: point-cloud shortlist, then ranked against the real
  solids, so a second magnet lands on the second-best patch.
- Axis chosen by testing three candidates (contact / mass / mass-flat) against
  what would collide on assembly, fixing the 90 deg-wrong magnets on DNA. The
  centre of mass of a rod slides along the rod; see NOTES.md.
- Joint path cleared: material reaching past the mating face is cut away so
  the parts can actually close.
- Bridge is the same joint minus the bore - a true cylinder, not a capsule.

UI: sticky Generate, larger top-left download buttons, structure-named
exports, cartoon representation withdrawn (design for the rework in NOTES.md).

Docs: README rewritten, HANDOVER.md added.
'@

git add -A
Assert-Git "git add"

# Nothing staged means it was already committed on a previous run.
git diff --cached --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "Nothing staged - already committed. Pushing." -ForegroundColor Yellow
} else {
    $msg | git commit -F -
    Assert-Git "git commit"
    Write-Host "`nCommitted:" -ForegroundColor Green
    git log --oneline -1
}

# --- 5. Push ----------------------------------------------------------------
git push
Assert-Git "git push"

Write-Host "`nPushed. See HANDOVER.md for tomorrow." -ForegroundColor Green
git log --oneline -3
