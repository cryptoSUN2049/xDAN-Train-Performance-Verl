#!/usr/bin/env bash
# Build the task-independent DSH runtime (/opt/dsh) once and pack it as an injectable payload.
#
# Same steps as the per-task image Dockerfile (portable CPython 3.12 + pinned wheelhouse + hashed
# runner sources + identity probe), but the result is a tarball that DshSdkAgent can unpack into
# any task sandbox at run time, so tasks and harnesses stay orthogonal (N images instead of N x H).
#
# usage: build_runtime_payload.sh <image-context-dir> <output-dir>
# Must run as root on a Linux x86_64 host where /opt/dsh does not exist yet.
set -euo pipefail
CONTEXT="${1:?image context dir (python.tar.gz, wheelhouse/, source/, install.py, build-inputs.json)}"
OUT="${2:?output dir}"
test ! -e /opt/dsh && test ! -e /opt/dsh-build
mkdir -p "$OUT"

mkdir /opt/dsh /opt/dsh-build
tar -xzf "$CONTEXT/python.tar.gz" -C /opt/dsh --strip-components=1
cp -r "$CONTEXT/wheelhouse" "$CONTEXT/source" "$CONTEXT/install.py" "$CONTEXT/build-inputs.json" /opt/dsh-build/
# install.py verifies wheel/source hashes, installs offline, writes /opt/dsh/bin/python, runs the identity
# probe and removes /opt/dsh-build.
/opt/dsh/bin/python3.12 -I /opt/dsh-build/install.py
mkdir -p /opt/dsh/checks && cp "$CONTEXT/smoke.py" /opt/dsh/checks/smoke.py

version=$(/opt/dsh/bin/python -c 'import importlib.metadata as m; print(m.version("deepseek-harness-sdk"))')
payload="$OUT/dsh-runtime-$version-linux-x86_64.tar.gz"
tar -czf "$payload" -C /opt dsh
sha=$(sha256sum "$payload" | cut -d' ' -f1)
echo "$sha" > "$payload.sha256"
rm -rf /opt/dsh
printf '{"payload": "%s", "sha256": "%s", "bytes": %s, "dsh_version": "%s"}\n' \
  "$payload" "$sha" "$(stat -c %s "$payload")" "$version" | tee "$OUT/payload.json"
