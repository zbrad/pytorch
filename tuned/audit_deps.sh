#!/bin/bash
# Sourced by tuned/package.sh. Audit the Python packages that `pip install <wheel>` pulls in.
#
# gpu_tuned_audit_deps <wheel> <log-file>
#   1. pip resolves the wheel's Requires-Dist into the exact versions it would install today
#      (`pip install --dry-run --report`, nothing installed) in a throwaway venv.
#   2. pip-audit checks those versions (not the wheel itself, whose local version is not on PyPI)
#      against the PyPI advisory data.
# A vulnerable package that has a fixed version fails the gate: the resolver chose the vulnerable
# one, so a pin or bound in the wheel metadata is keeping it there. A vulnerable package with no
# fixed version is printed as a warning and does not fail, since nothing we ship can change it.
# An advisory can be accepted in tuned/audit_allow.txt ("<advisory id>  <reason>"); each accepted
# id is printed. A check that cannot run (no network, resolver or pip-audit failure) is a failure
# (H4), not a skip. The report goes to stdout and <log-file>. GPU_TUNED_AUDIT_ALLOW_FILE overrides the
# allow file path (the tests use it).
#
# This audits what we ship as runtime dependencies. It does not look at upstream's CI, test or UI
# manifests, which is what Dependabot alerts on the default branch mostly report.
GPU_TUNED_PIP_AUDIT_VERSION="${GPU_TUNED_PIP_AUDIT_VERSION:-2.10.1}"

gpu_tuned_audit_deps() {
    local wheel="${1:?usage: gpu_tuned_audit_deps <wheel> <log-file>}" log="${2:?usage: gpu_tuned_audit_deps <wheel> <log-file>}"
    local here work rc=0
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    command -v python3 >/dev/null || { echo "ERROR: python3 is needed for the dependency audit." >&2; return 1; }
    work="$(mktemp -d)"
    {
        echo "=== dependency audit: runtime dependencies of $(basename "${wheel}") ==="
        echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ); pip-audit ${GPU_TUNED_PIP_AUDIT_VERSION}"
        python3 -m venv "${work}/venv" \
            && "${work}/venv/bin/python" -m pip install --quiet --disable-pip-version-check "pip-audit==${GPU_TUNED_PIP_AUDIT_VERSION}" \
            && "${work}/venv/bin/python" -m pip install --quiet --disable-pip-version-check --dry-run --ignore-installed \
                   --report "${work}/report.json" "${wheel}" \
            || { echo "ERROR: could not set up the audit venv or resolve the wheel's dependencies (network?)." >&2; rc=1; }
        if [[ ${rc} -eq 0 ]]; then
            "${work}/venv/bin/python" -I - "${work}/report.json" "${wheel}" > "${work}/reqs.txt" <<'PY' || rc=1
import json, re, sys
report, wheel = sys.argv[1], sys.argv[2]
own = re.sub(r"[-_.]+", "-", wheel.rsplit("/", 1)[-1].split("-", 1)[0]).lower()
for item in json.load(open(report))["install"]:
    meta = item["metadata"]
    if re.sub(r"[-_.]+", "-", meta["name"]).lower() != own:
        sys.stdout.write(f'{meta["name"]}=={meta["version"]}\n')
PY
        fi
        if [[ ${rc} -eq 0 ]]; then
            [[ -s "${work}/reqs.txt" ]] || { echo "ERROR: the wheel resolved to no dependencies; the audit has nothing to check." >&2; rc=1; }
        fi
        if [[ ${rc} -eq 0 ]]; then
            "${work}/venv/bin/pip-audit" --no-deps --disable-pip -r "${work}/reqs.txt" --format json \
                > "${work}/audit.json" 2> "${work}/audit.err"
            [[ -s "${work}/audit.json" ]] || { sed 's/^/  /' "${work}/audit.err" >&2; echo "ERROR: pip-audit produced no result." >&2; rc=1; }
        fi
        if [[ ${rc} -eq 0 ]]; then
            python3 -I - "${work}/audit.json" "${GPU_TUNED_AUDIT_ALLOW_FILE:-${here}/audit_allow.txt}" <<'PY' || rc=1
import json, sys
audit_path, allow_path = sys.argv[1], sys.argv[2]
allowed = {}
try:
    for line in open(allow_path):
        line = line.split("#", 1)[0].strip()
        if line:
            ident, _, reason = line.partition(" ")
            if not reason.strip():
                sys.stdout.write(f"ERROR: {allow_path}: '{ident}' has no reason.\n")
                sys.exit(1)
            allowed[ident] = reason.strip()
except FileNotFoundError:
    pass
deps = json.load(open(audit_path))["dependencies"]
failed = False
seen = set()
sys.stdout.write(f"audited {len(deps)} packages: " + " ".join(f'{d["name"]}=={d["version"]}' for d in deps) + "\n")
for dep in deps:
    for vuln in dep.get("vulns", []):
        if (dep["name"], dep["version"], vuln["id"]) in seen:
            continue
        seen.add((dep["name"], dep["version"], vuln["id"]))
        ids = {vuln["id"], *vuln.get("aliases", [])}
        accepted = next((allowed[i] for i in ids if i in allowed), None)
        fixes = ", ".join(vuln.get("fix_versions", [])) or "none"
        tag = "ACCEPTED" if accepted else ("VIOLATION" if vuln.get("fix_versions") else "WARNING")
        sys.stdout.write(f'{tag}: {dep["name"]}=={dep["version"]} {vuln["id"]} fixed in: {fixes}'
                         + (f" ({accepted})" if accepted else "") + "\n")
        failed = failed or tag == "VIOLATION"
sys.stdout.write("RESULT: " + ("FAILED" if failed else "no vulnerable package has a fix available") + "\n")
sys.exit(1 if failed else 0)
PY
        fi
    } 2>&1 | tee "${log}"
    # The block above runs in the pipeline's subshell, so its rc is lost; the RESULT line is the verdict.
    grep -q '^RESULT: no vulnerable package has a fix available$' "${log}" || rc=1
    rm -rf "${work}"
    return "${rc}"
}
