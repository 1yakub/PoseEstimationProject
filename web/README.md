# Browser demo (in progress)

The same rep counting pipeline as the Python app, moved into the browser so it can be
hosted as static files. Frames are processed on the visitor's device with MediaPipe Tasks
Vision compiled to WebAssembly, so no video is uploaded and the server does no work.

## State

Done:

- `index.html`, the full page
- `styles.css`, the full design system
- `fonts/` plus `fonts.css`, latin subsets, self hosted, 140 KB
- `vendor/`, MediaPipe tasks-vision 1.0.1 bundle and the wasm pair
- `models/pose_landmarker_lite.task`

Not done, so the demo does not run yet:

1. `trainer.js`. Needs `FilesetResolver.forVisionTasks('./vendor/wasm')`, a
   `PoseLandmarker` in `VIDEO` running mode, `getUserMedia` with a file input fallback,
   the canvas draw (skeleton, plus an angle arc at the working joint, mirrored for the
   camera), and a JS port of `RepCounter` from `../AITrainer.py`. Keep the constants
   identical to the Python side: curl 160 to 50 degrees at the elbow, squat 165 to 70 at
   the knee, visibility cutoff 0.5.
2. Asset handling. `vendor/` and `models/` are about 28 MB, so gitignore them, add a
   `fetch-assets.sh` that the Docker build runs, and let the JS fall back to the jsDelivr
   CDN when a local asset is missing.
3. `Dockerfile` and an nginx config: gzip the wasm and the model, long cache headers on
   everything under `vendor/`, `models/` and `fonts/`. Use an arm64 capable base image.

## Running it locally

Any static server works. The wasm needs to be served over http, not opened from disk.

    python -m http.server 8000 --directory web
