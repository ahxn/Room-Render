# Evaluation

## Questions

The version 1 evaluation asks:

1. Does the full pipeline complete on a personally captured room video?
2. Are enough frames registered to support a coherent scene?
3. Does optional blur filtering improve registration, speed, or held-out image
   quality?
4. Do the same defaults generalize to additional rooms?

## Method

- Use a unique output directory for every run.
- Keep the source video and frame target fixed when comparing filtering on/off.
- Record extracted, selected, rejected, and registered frame counts from
  `metadata.json`.
- Record wall-clock pipeline duration from a non-resumed run's `metadata.json`.
  A resumed invocation records only the resumed command's duration.
- Run `ns-eval --load-config <config.yml> --output-path <output.json>` for PSNR,
  SSIM, and LPIPS after training.
- Visually inspect the same held-out views for floating artifacts, missing surfaces,
  edge stability, and recognizable room structure.
- Treat differences on one video as directional, not proof of generalization.

## Results to date

Both runs below use the same first-room video and full 30,000-iteration
`splatfacto` optimization.

| Run | Extracted | Selected | Registered | Registration | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline, filtering off | 250 | 250 | 250 | 100.00% | 27.719 | 0.93306 | 0.12713 |
| Blur filtering, threshold 2.5 | 250 | 221 | 215 | 97.29% | 27.428 | 0.92552 | 0.12905 |

The baseline export contains 266,201 Gaussians and is approximately 66 MB.
Side-by-side held-out renders appeared nearly identical. The filtered run was
slightly worse on every reported quantitative metric, so blur filtering remains
available for experiments but is disabled by default.

This does not show that blur filtering is universally harmful. The first capture
was already sharp and registered perfectly. Filtering may still help a genuinely
blurred capture, but that hypothesis needs another controlled test.

## Remaining experiment matrix

| Capture | Default | Low-memory preview | Filtering comparison | Status |
| --- | --- | --- | --- | --- |
| Room 1 | Complete | Not required for initial proof | Complete | Complete |
| Room 2 | Failed registration gate (2/180, 1.1%) | Attempted | Not run | Recapture needed: insufficient overlap |
| Room 3 | Pending | Pending | Pending if blur is present | Waiting for capture |

Room 2 was a 47.9-second portrait capture (1080×1920) and is accepted by the
orientation-aware validator. Both sequential and exhaustive COLMAP matching found
poses for only 2/180 frames. The sampled views jump between room areas with too
little overlap, so the safety gate stopped the run before GPU training. This is a
useful failure result, not a successful reconstruction. Recapture Room 2 with
slower movement and overlapping intermediate views. A third capture can be
intentionally difficult—blank walls, reflections, or mild motion blur—to test
failure guidance and whether filtering ever helps.

## Generate a metadata table

```bash
python scripts/summarize_results.py \
  /home/allen/results/my-room \
  /home/allen/results/my-room-filtered-v2
```

This reports pipeline metadata. `ns-eval` quality metrics are kept separate because
they are outputs of the trained Nerfstudio configuration rather than the
orchestration metadata.
