# Demo Runbook

Use this sequence for a short, reproducible project demo. The complete version
takes about two minutes when a reconstruction already exists.

## Story

1. Show 5–10 seconds of the original handheld room video.
2. Explain the pipeline in one sentence: sampled frames are registered into camera
   poses, then a Gaussian-splat scene is optimized to reproduce those views.
3. Run the environment check and a dry run to show the one-command interface.
4. Open the completed result and navigate around the room.
5. Show `metadata.json`, the evaluation table, and the exported PLY.
6. Close with the honest boundary: one room is proven; additional-room validation
   is the next milestone.

## Commands

```bash
conda activate nerfstudio
cd "/mnt/c/Users/allen/Documents/ChatGPT/Room Reconstruction"

python reconstruct.py --check-environment

python reconstruct.py "/mnt/c/Users/allen/Downloads/test1.mp4" \
  --output "/home/allen/results/demo-dry-run" \
  --config configs/default.yml \
  --dry-run

python reconstruct.py \
  --output "/home/allen/results/my-room" \
  --open
```

## Recording checklist

- Close unrelated windows and hide personal paths or notifications.
- Use a 1080p recording at 30 fps.
- Keep terminal text large enough to read.
- In the viewer, move slowly and pause at a recognizable input viewpoint.
- Do not imply that the dry run reconstructs the room; state that it validates the
  input and previews commands.
- Do not imply a reusable neural model was trained; state that the scene itself was
  optimized from this room's images.
- Stop the viewer with Ctrl+C and confirm GPU activity returns to idle before ending.

The repository should link the finished video once recorded. A screen recording is
the only presentation artifact here that still requires a manual capture step.
