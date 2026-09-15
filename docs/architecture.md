# Architecture

Room Render is a Python orchestration layer around established
computer-vision tools. It validates inputs, runs each stage, checks outputs,
records metadata, and stops unsafe runs before GPU training.

```text
video -> FFprobe -> FFmpeg frames -> optional OpenCV blur filter
      -> Nerfstudio/COLMAP camera poses -> registration gate
      -> Nerfstudio splatfacto -> checkpoint -> viewer or PLY export
```

The project owns validation, frame selection, configuration validation, pipeline
state, logging, resume behavior, quality gates, and export orchestration.
FFmpeg, COLMAP, Nerfstudio, and PyTorch perform media decoding, pose estimation,
Gaussian-splat optimization, and rendering.

Each run writes `metadata.json` and `logs/processing.log`. `--resume` explicitly
reuses completed stages and checkpoints. Sequential feature matching is the
default for ordered video frames, and blur filtering is opt-in.
