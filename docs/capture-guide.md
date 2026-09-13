# Room Capture Guide

Good input matters more than aggressive tuning. COLMAP needs the same textured
surfaces to remain visible across nearby frames while the camera changes position.

## Before recording

- Use the main rear camera at a fixed zoom; avoid ultrawide lens switching.
- Clean the lens and record at 1080p or higher in steady, even lighting.
- Open curtains only if the window will not dominate the exposure.
- Remove moving people, pets, fans, and television content where practical.
- Plan a continuous walking route around the room rather than rotating in place.

## While recording

- Record roughly 30–60 seconds.
- Walk slowly and keep the phone motion smooth.
- Move sideways and around furniture to create parallax; do not only pan from one
  position.
- Keep 60–80% visual overlap between nearby views.
- Include walls and objects with texture. Approach blank walls obliquely while
  retaining textured edges or furniture in frame.
- Cover corners, doorways, and important objects from more than one angle.
- Avoid sudden turns, autofocus hunting, zooming, and exposure changes.

## Quick review before a full run

Reject and recapture if the video has long blurred sections, rapid turns, major
lighting changes, or large moving subjects. Then run a no-GPU command check:

```bash
python reconstruct.py "/path/to/room.mp4" \
  --output "/home/allen/results/room-dry-run" \
  --config configs/default.yml \
  --dry-run
```

For a new room, use the low-memory preset first. It is a cheaper way to confirm
camera registration and recognizable geometry before committing to the default
30,000-iteration run:

```bash
python reconstruct.py "/path/to/room.mp4" \
  --output "/home/allen/results/room-preview" \
  --config configs/low-memory.yml
```

If registration is weak, recapture before tuning training. More optimization
cannot repair incorrect or missing camera poses.

## Consistent experiment naming

Use separate directories so runs never overwrite each other:

```text
/home/allen/results/office-preview
/home/allen/results/office-default
/home/allen/results/office-filtered
```

Record the room, capture date, phone/camera, preset, filtering choice, and any
unusual conditions in the evaluation notes.
