#!/usr/bin/env bash
# Fetch the MediaPipe Tasks Vision runtime and the pose model into vendor/ and models/.
# Run from the web/ folder. Idempotent: existing files are kept.
set -euo pipefail
cd "$(dirname "$0")"
VERSION="${MEDIAPIPE_VERSION:-1.0.1}"
BASE="https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}"
MODEL="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
mkdir -p vendor/wasm models
get() { [ -s "$2" ] && { echo "have  $2"; return; }; echo "fetch $2"; curl -fsSL -o "$2" "$1"; }
get "$BASE/vision_bundle.mjs" vendor/vision_bundle.mjs
for f in vision_wasm_internal.js vision_wasm_internal.wasm vision_wasm_nosimd_internal.js vision_wasm_nosimd_internal.wasm; do
  get "$BASE/wasm/$f" "vendor/wasm/$f"
done
get "$MODEL" models/pose_landmarker_lite.task
echo "$VERSION" > vendor/VERSION
du -sh vendor models
