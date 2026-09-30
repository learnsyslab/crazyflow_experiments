# Reconstructing a scene

Once, before the first run:

```bash
pixi run -e vision colmap-build   # compiles COLMAP with CASPAR, about half an hour
```

## The sequence

```bash
# 0. Put the recording in recordings/. Its stem becomes the scene name.
cp <your video>.mp4 recordings/<scene>.mp4

# 1. Select frames.
pixi run -e vision python crazyflow_experiments/vision/reconstruct/frames.py <scene>.mp4

# 2. Reconstruct. This might take some time.
pixi run -e vision python crazyflow_experiments/vision/reconstruct/sfm.py <scene>

# 3. Export the model to the archive the trainer reads.
pixi run -e vision python crazyflow_experiments/vision/reconstruct/export.py \
    saves/scenes/<scene>/sparse/0

# 4. Fit the splat.
pixi run -e vision python crazyflow_experiments/vision/reconstruct/train.py \
    --scene <scene> --variant 30k --n_gaussians 30000 --downscale 2

# 5. Align the scene to the room, writing saves/scenes/<scene>/frame.json.
pixi run -e vision python crazyflow_experiments/vision/reconstruct/align.py refine \
    saves/scenes/<scene>/raw/30k.ply

# 6. Carry a fit into the room's frame with the transform found in the refinement.
pixi run -e vision python crazyflow_experiments/vision/reconstruct/align.py apply \
    saves/scenes/<scene>/raw/30k.ply saves/scenes/<scene>/aligned/30k.ply
```

Everything of one scene sits under `saves/scenes/<scene>/`: the frames in `images/`, the COLMAP
model in `sparse/`, the cameras and points in `scene.npz` and the transform to scale the scene to
its real size in `frame.json`.

## Aligning

A reconstruction is solved only up to a similarity, so it comes out tilted and arbitrarily scaled.
Stage 5 opens a viewer with the world axes, two reference lines and a marker, and you move the scene
by hand.

The transform belongs to the scene, not to the fit you happened to align against, so it is written
once to `saves/scenes/<scene>/frame.json` as a rotation, a scale and a translation. Stage 6 applies
it to any fit of the same reconstruction.
