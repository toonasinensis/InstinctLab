PGMT offline WM rollout inspector

Capture with the verified Isaac Lab / Newton environment:
  python scripts/pgmt/export_wm_rollout.py --checkpoint /path/to/model_2000.pt --output /persistent/path/new-capture

The checkpoint's sibling params/env.yaml and params/agent.yaml must exist.
No training, optimizer step, or training-resume contract relaxation is performed.
The deterministic PPO actor, WM weights, and normalizers are frozen and checked.
All regular H32 windows (stride 16) are exported, including valid terminal prefixes.
Predictions receive one real initial state and the same recorded actions as the simulator.
Contact is autoregressive after the initial state. Heightmaps use predicted poses.
No root/yaw alignment, foot IK, collision correction, or predicted-state physics is applied.
Native terminal successors are captured before automatic resets.

Default cases: flat L0 center; stairs-up L5 center; stairs-up L9 off-platform;
stairs-down L9 off-platform; rough L9 center; rough L9 off-platform.
Off-platform cases shift the spawn +1.6 m along world X, preserving root clearance.
They are explicitly labeled and are not native training reset samples.
The terrain ranges are unchanged. The terrain curriculum and observation noise
are disabled during this short diagnostic, not during training.

Complete the portable output by copying a pinned Three.js 0.180.0 vendor folder
containing three.module.js, three.core.js, OrbitControls.js and its LICENSE.
The existing supermimic_vibed/.cache/play/three-0.180.0 cache is compatible.
No CDN or Internet access is used by the completed viewer.
Serve the exported directory, for example:
  python -m http.server 16011 --bind 127.0.0.1 --directory /path/to/capture
Open http://127.0.0.1:16011/ in a WebGL-capable browser.

Blue = native simulator; amber = open-loop WM. Split views use the SAME camera.
The camera is centered once at the window start. Dragging the camera affects both.
Root error is Euclidean position distance in cm; foot error is the maximum
Euclidean error over the two ankle-roll link origins (not the sole surface).
Joint RMSE is the raw 29-joint angle RMSE in degrees.
Plots and metrics use original 50 Hz samples; no temporal interpolation is applied.
H8 is 0.16 s of training horizon; H32 is 0.64 s, including extrapolation.
The loop control repeats a single explicitly labeled window, not a long rollout.
The worst-window button searches the selected scene at the selected horizon.
The fixed camera frames both full window trajectories, including the feet;
it never follows or realigns a prediction at each time step. Orbit/side presets
use the initial true heading. The export-view button saves the current render
with its legend, checkpoint identity, anchor step, horizon and position errors.

Heightmaps are the SAME baked terrain sampled at true vs predicted pelvis pose,
with shared color limits and signed difference (WM minus actual). They are not
learned depth images or native raycast-vs-raster comparisons. Grid order is X slow,
Y fast; display X right / Y up. The H0 map is intentionally blank.
Rewards are weighted per control step, in the checkpoint term order:
native simulator, analytic true state/exact mesh, and analytic predicted state/raster.
Only supported predicted reward terms are selectable; totals use that same subset.
This distinguishes formula parity from compounded model/raster/event errors.

The full mesh is cropped only for display; WM queries retain the original map.
Robot visuals are exported from the SAME URDF using the native joint ordering.
Capture verifies URDF/native link FK (<0.3 mm), and the browser also checks both
native ankle positions (<0.3 mm). World quaternions are XYZW throughout.

manifest.json records hashes, seeds, validity, reset boundaries, and capture checks.
data.bin and meshes.bin contain little-endian float32/uint32 arrays, with explicit
offsets and shapes in the manifest. Keep the entire directory when copying.
These short diagnostic clips are not a tracking benchmark or proof of gradients.
