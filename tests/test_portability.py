"""Portability checks the Linux build cannot make by compiling: MSVC is the
Windows compiler (build.bat), and its /openmp is OpenMP 2.0. GCC accepts
newer constructs that MSVC refuses — v0.22.0's first Windows build failed on
`#pragma omp atomic write` (3.1). This scans the C sources so the Linux suite
fails first.

Run from the repo root:  python -m pytest tests/test_portability.py -q
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = sorted((ROOT / "src").glob("*.c")) + sorted((ROOT / "src").glob("*.h"))

# OpenMP clauses / directives beyond 2.0 (MSVC's /openmp)
NEWER = [
    (r"#\s*pragma\s+omp\s+atomic\s+(read|write|capture|update|seq_cst)", "atomic read/write/capture/update (3.1)"),
    (r"#\s*pragma\s+omp\s+(task|taskwait|taskyield|taskgroup|taskloop)\b", "tasks (3.0+)"),
    (r"#\s*pragma\s+omp\s+(simd|declare|target|teams|distribute|cancel)\b", "simd/declare/target/teams/distribute/cancel (4.0+)"),
    (r"\bcollapse\s*\(", "collapse (3.0)"),
    (r"schedule\s*\(\s*auto\b", "schedule(auto) (3.0)"),
    (r"\breduction\s*\(\s*(min|max)\s*:", "min/max reduction (3.1)"),
    (r"\bomp_(get_max_active_levels|set_max_active_levels|get_level|get_ancestor_thread_num|get_team_size|"
     r"get_active_level|in_final|get_cancellation|get_proc_bind|get_num_places)\b", "an OpenMP 3.0+ runtime call"),
]


def test_openmp_is_msvc_2_0_compatible():
    bad = []
    for f in SOURCES:
        for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            for pat, what in NEWER:
                if re.search(pat, line):
                    bad.append(f"{f.name}:{i}: {what}: {line.strip()}")
    assert not bad, "OpenMP newer than MSVC's 2.0:\n" + "\n".join(bad)


def test_parallel_for_loop_variables_are_signed_int():
    """OpenMP 2.0 requires a signed integer loop variable in a parallel for;
    the next `for (` after each `parallel for` pragma must use one."""
    bad = []
    for f in SOURCES:
        lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, line in enumerate(lines):
            if re.search(r"#\s*pragma\s+omp\s+(parallel\s+)?for\b", line):
                nxt = next((l for l in lines[i + 1:i + 4] if l.strip().startswith("for")), "")
                m = re.match(r"\s*for\s*\(\s*(?:(\w[\w\s]*?)\s+)?(\w+)\s*=", nxt)
                if not m:
                    bad.append(f"{f.name}:{i + 2}: cannot read the loop: {nxt.strip()}")
                    continue
                decl = (m.group(1) or "").strip()
                if decl and decl not in ("int", "long", "long long", "ptrdiff_t"):
                    bad.append(f"{f.name}:{i + 2}: loop variable declared {decl!r}: {nxt.strip()}")
                if not decl:
                    # declared earlier: find "int <name>;" before the pragma
                    name = m.group(2)
                    before = "\n".join(lines[max(0, i - 40):i])
                    if not re.search(r"\b(int|long|ptrdiff_t)\s+([\w\s,=*]*\b)?" + re.escape(name) + r"\b", before):
                        bad.append(f"{f.name}:{i + 2}: loop variable {name!r} not declared a signed int nearby")
    assert not bad, "\n".join(bad)
