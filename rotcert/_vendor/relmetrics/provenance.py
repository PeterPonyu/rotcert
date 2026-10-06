"""Provenance stamping for result dicts.

Portfolio rule: results only change by re-running analysis code, and every
result JSON must be traceable to the exact code, seed(s), and environment
that produced it. :func:`stamp_result` embeds that trace under the
``"provenance"`` key so downstream figure scripts and manuscripts can verify
lineage.
"""

from __future__ import annotations

import datetime
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

__all__ = ["stamp_result"]


def _git_info(directory: Path) -> Dict[str, Optional[Union[str, bool]]]:
    """Return {'git_sha', 'git_dirty'} for the repo containing ``directory``,
    or Nones if not inside a git repository (or git is unavailable)."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if sha.returncode != 0:
            return {"git_sha": None, "git_dirty": None}
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=10,
        )
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
        return {"git_sha": sha.stdout.strip(), "git_dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"git_sha": None, "git_dirty": None}


def stamp_result(
    result: Dict[str, object],
    script_path: Union[str, Path],
    seeds: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    """Embed provenance metadata into a result dict (in place; also returned).

    Adds/overwrites ``result["provenance"]`` with:

    - ``script``: absolute path of the producing script,
    - ``git_sha`` / ``git_dirty``: HEAD SHA and dirty flag of the repo
      containing the script (``None`` if not a repo),
    - ``timestamp_utc``: ISO-8601 UTC time of stamping,
    - ``seeds``: list of RNG seeds used (or ``None``),
    - ``python_version`` and ``numpy_version``,
    - ``platform``: OS/arch string.

    Parameters
    ----------
    result:
        The result dict about to be serialized (e.g. to JSON).
    script_path:
        Path of the analysis script, normally ``__file__``.
    seeds:
        RNG seed(s) that produced the result.

    Returns
    -------
    dict
        The same ``result`` object, with ``"provenance"`` filled in.
    """
    script = Path(script_path).resolve()
    directory = script.parent if script.parent.exists() else Path.cwd()

    try:
        import numpy

        numpy_version: Optional[str] = numpy.__version__
    except ImportError:  # pragma: no cover - numpy is a hard dep elsewhere
        numpy_version = None

    provenance: Dict[str, object] = {
        "script": str(script),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seeds": list(seeds) if seeds is not None else None,
        "python_version": sys.version.split()[0],
        "numpy_version": numpy_version,
        "platform": platform.platform(),
    }
    provenance.update(_git_info(directory))
    result["provenance"] = provenance
    return result
