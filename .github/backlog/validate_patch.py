"""Validate an untrusted patch before materializing it; run from the base revision."""
import pathlib
import subprocess
import sys


def git(*args):
    return subprocess.check_output(["git", *args])


patch = pathlib.Path(sys.argv[1]).resolve()
if not 0 < patch.stat().st_size <= 100_000:
    raise SystemExit("Patch must contain 1–100000 bytes")
subprocess.run(["git", "apply", "--cached", "--check", str(patch)], check=True)
subprocess.run(["git", "apply", "--cached", str(patch)], check=True)
names = git("diff", "--cached", "--name-only", "-z").split(b"\0")[:-1]
if not 0 < len(names) <= 20:
    raise SystemExit("Patch must change 1–20 files")
for raw in names:
    name = raw.decode("utf-8", errors="strict")
    path = pathlib.PurePosixPath(name)
    if (path.is_absolute() or ".." in path.parts or "\\" in name
            or any(part.startswith(".") for part in path.parts)
            or not (name == "README.md" or path.parts[0] in {"src", "tests", "scripts", "configs"})):
        raise SystemExit(f"Disallowed path: {name}")
for record in git("diff", "--cached", "--raw", "--no-renames", "-z").split(b"\0"):
    if record.startswith(b":"):
        old, new = record.split()[:2]
        if old[1:] not in {b"000000", b"100644"} or new not in {b"000000", b"100644"}:
            raise SystemExit("Only regular non-executable files are allowed")
subprocess.run(["git", "diff", "--cached", "--check"], check=True)
# Validation is complete. Apply exactly the validated patch to the worktree.
subprocess.run(["git", "apply", str(patch)], check=True)
