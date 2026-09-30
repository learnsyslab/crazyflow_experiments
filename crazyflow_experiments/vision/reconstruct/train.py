"""Fit a gaussian splat to a scene and save it.

Ensure that you are tuning on the same resolution as used during training for the best results. We
train on multiple reconstructions of varying convergence, so the policy does not overfit. The pool
is a grid of 10k, 20k and 30k gaussians at 1k, 3k, 10k and 30k optimizer steps.

Run with
    pixi run -e vision python crazyflow_experiments/vision/reconstruct/train.py \
        --scene lab --variant 100k --n_gaussians 100000 --steps 30000
"""

from __future__ import annotations

import os

os.environ["SCIPY_ARRAY_API"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import dm_pix
import fire
import jax
import jax.numpy as jnp
import numpy as np
import optax
import splax
from scipy.spatial import KDTree
from scipy.special import logit

from crazyflow_experiments.utils.paths import SCENES_DIR

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Args:
    """Configuration of a splat fitting run."""

    variant: str
    """name of this fit"""
    scene: str = "lab"
    """scene to fit"""
    images: str | None = None
    """explicit directory of the source frames, overriding scene"""
    masks: str | None = None
    """directory of one mask per camera if required"""
    out: str | None = None
    """explicit save directory, overriding scene"""
    seed: int = 0

    # Splat settings
    n_gaussians: int = 100_000
    """size of the splat"""
    sh_degree: int = 3
    """degree of the harmonics carrying the colour in [0, 1, 2, 3]"""
    downscale: int = 2
    """downscale factor for the training"""
    init_opacity: float = 0.1
    """initial gaussian opacity"""
    scale_cap: float = 0.3
    """largest gaussian allowed for the nearest neighbour spacing in normalized units"""
    max_anisotropy: float = 100.0
    """largest ratio between the longest and the shortest axis of a gaussian"""

    # Optimizer settings
    steps: int = 30_000
    """optimizer steps"""
    batch: int = 4
    """batch size"""
    means_lr: float = 1.5e-3
    scales_lr: float = 5e-3
    quats_lr: float = 1e-3
    colors_lr: float = 1e-2
    sh_rest_lr_scale: float = 0.05
    opacity_lr: float = 5e-2
    exposure_lr: float = 1e-3

    # Loss settings
    ssim_lambda: float = 0.2
    """structural similarity loss weight"""
    opacity_reg: float = 0.003
    """relocation regularization weight"""
    scale_reg: float = 0.01
    """roundness regularization weight"""

    # Relocation settings
    min_opacity: float = 0.005
    """opacity below which a gaussian is recycled"""
    refine_start: int = 200
    """views before the first relocation"""
    relocate_every: int = 100
    """views between relocations"""
    noise_lr: float = 5e4
    """scale of the noise injected into the positions, annealed with the schedule"""

    # Evaluation settings
    test_every: int = 8
    """one held out view in this many"""
    log_every: int = 1000
    """steps between evaluations of the held out views"""


def load_views(images: Path, names: list[str], downscale: int) -> np.ndarray:
    """Load the training images at the requested resolution."""
    views = []
    for name in names:
        image = cv2.imread(str(images / name))
        assert image is not None, f"{images / name} is missing"
        height, width = image.shape[0] // downscale, image.shape[1] // downscale
        views.append(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)[..., ::-1])
    return np.stack(views)


def load_mask(masks: Path, name: str, shape: tuple[int, int]) -> np.ndarray:
    """Load the overlay mask at the training resolution."""
    mask = cv2.imread(str(masks / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
    assert mask is not None, f"{masks / name}.png is missing"
    return (cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_AREA) > 200)[..., None]


def knn_log_scales(xyz: np.ndarray, cap: float, k: int = 3) -> np.ndarray:
    """Size each gaussian by how far its nearest neighbours are."""
    distance, _ = KDTree(xyz).query(xyz, k=k + 1)  # the first neighbour is the point itself
    return np.log(np.clip(distance[:, 1:].mean(axis=1), 1e-4, cap)).astype(np.float32)


def init_params(
    xyz: np.ndarray, rgb: np.ndarray, track: np.ndarray, args: Args
) -> dict[str, jax.Array]:
    """Seed a fixed budget of gaussians from the sparse cloud, subsampling or padding to it."""
    n = args.n_gaussians
    rng = np.random.default_rng(args.seed)
    weights = np.log1p(track)
    if len(xyz) >= n:
        keep = rng.choice(len(xyz), n, replace=False, p=weights / weights.sum())
        xyz, rgb = xyz[keep], rgb[keep]
        log_scales = knn_log_scales(xyz, args.scale_cap)
    else:
        # Padding a cloud thinner than the budget splits each point into a cluster, so the
        # gaussians shrink by the cube root of how many now share the space.
        log_scales = knn_log_scales(xyz, args.scale_cap) - np.log(n / len(xyz)) / 3.0
        source = rng.choice(len(xyz), n - len(xyz), replace=True, p=weights / weights.sum())
        jitter = rng.normal(size=(len(source), 3)) * np.exp(log_scales[source])[:, None]
        xyz = np.concatenate([xyz, xyz[source] + jitter])
        rgb = np.concatenate([rgb, rgb[source]])
        log_scales = np.concatenate([log_scales, log_scales[source]])
    return {
        "means": jnp.asarray(xyz, jnp.float32),
        "log_scales": jnp.asarray(np.repeat(log_scales[:, None], 3, axis=1)),
        "quats": jnp.asarray(rng.normal(size=(n, 4)), jnp.float32),
        "sh_base": splax.io.rgb_to_sh(jnp.asarray(rgb / 255.0, jnp.float32))[:, None],
        "sh_rest": jnp.zeros((n, (args.sh_degree + 1) ** 2 - 1, 3), jnp.float32),
        "opacity_logit": jnp.full((n,), logit(args.init_opacity), jnp.float32),
    }


def normalize(
    viewmats: np.ndarray, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Rescale so cameras sit about a unit out, returning the scene, the scale and the centre."""
    rotations, translations = viewmats[:, :3, :3], viewmats[:, :3, 3]
    centers = -np.einsum("nji,nj->ni", rotations, translations)
    centre = np.median(centers, axis=0)
    scale = 1.0 / np.linalg.norm(centers - centre, axis=1).mean()
    normalized = viewmats.copy()
    normalized[:, :3, 3] = scale * (translations + np.einsum("nij,j->ni", rotations, centre))
    return normalized, scale * (points - centre), float(scale), centre


def to_metres(
    params: dict[str, jax.Array], scale: float, centre: np.ndarray
) -> dict[str, jax.Array]:
    """Carry a splat trained on a normalized scene back into metres."""
    return {
        **params,
        "means": params["means"] / scale + jnp.asarray(centre, jnp.float32),
        "log_scales": params["log_scales"] - np.log(scale),
    }


def cap_anisotropy(log_scales: jax.Array, max_ratio: float) -> jax.Array:
    """Raise the thinnest axis of each gaussian so its axis ratio stays bounded."""
    return jnp.maximum(log_scales, jnp.max(log_scales, axis=-1, keepdims=True) - np.log(max_ratio))


def render(
    params: dict[str, jax.Array],
    viewmat: jax.Array,
    shape: tuple[int, int],
    intrinsics: np.ndarray,
    dist: np.ndarray,
) -> jax.Array:
    """Render one view."""
    fx, fy, cx, cy = intrinsics
    image, _ = splax.render(
        params["means"],
        params["log_scales"],
        params["quats"],
        jnp.concatenate([params["sh_base"], params["sh_rest"]], axis=1),
        params["opacity_logit"],
        viewmat=viewmat,
        background=jnp.ones(3),
        img_shape=shape,
        f=(fx, fy),
        c=(cx, cy),
        dist=tuple(dist),
    )
    return image


def build_loss_fn(
    shape: tuple[int, int], intrinsics: np.ndarray, dist: np.ndarray, mask: jax.Array, args: Args
) -> Callable[..., tuple[jax.Array, jax.Array]]:
    """Build the photometric loss over a batch of views."""

    def per_view(
        params: dict[str, jax.Array], target: jax.Array, viewmat: jax.Array, exposure: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        image = render(params, viewmat, shape, intrinsics, dist)
        image = jnp.einsum("ij,hwj->hwi", exposure[:, :3], image) + exposure[:, 3]
        # Compositing the target into the masked pixels leaves them with no loss and no gradient.
        image = jnp.where(mask, image, target)
        return jnp.abs(image - target).mean(), 1.0 - dm_pix.ssim(target, jnp.clip(image, 0.0, 1.0))

    def loss_fn(
        params: dict[str, jax.Array], targets: jax.Array, viewmats: jax.Array, exposures: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        l1, dssim = jax.vmap(per_view, in_axes=(None, 0, 0, 0))(
            params, targets, viewmats, exposures
        )
        loss = (1.0 - args.ssim_lambda) * l1.mean() + args.ssim_lambda * dssim.mean()
        loss = loss + args.opacity_reg * jax.nn.sigmoid(params["opacity_logit"]).mean()
        # Elongated gaussians fit the training views and float in novel ones, so pull them round.
        return loss + args.scale_reg * jnp.exp(params["log_scales"]).mean(), l1.mean()

    return loss_fn


def reset_moments(opt_state: optax.OptState, reset: jax.Array) -> optax.OptState:
    """Zero the Adam moments of the gaussians that were relocated."""
    keep = (~reset).astype(jnp.float32)

    def zero(x: jax.Array) -> jax.Array:
        if isinstance(x, jnp.ndarray) and x.ndim >= 1 and x.shape[0] == reset.shape[0]:
            return x * keep.reshape((-1,) + (1,) * (x.ndim - 1))
        return x

    return jax.tree.map(zero, opt_state)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Compute PSNR between two images in the unit range."""
    mse = float(np.mean((np.clip(a, 0.0, 1.0) - b) ** 2))
    return -10.0 * np.log10(mse) if mse > 0 else float("inf")


def learning_schedule(args: Args) -> optax.Schedule:
    """Anneal the rate the positions follow down to a hundredth of it over the run."""
    return optax.exponential_decay(args.means_lr * np.sqrt(args.batch), args.steps, 0.01)


def exposure_table(n_views: int) -> jax.Array:
    """Give every training view an identity initialized affine colour correction."""
    identity = jnp.concatenate([jnp.eye(3), jnp.zeros((3, 1))], axis=1)
    return jnp.repeat(identity[None], n_views, axis=0)


def build_optimizer(args: Args, schedule: optax.Schedule) -> optax.GradientTransformation:
    """Give every parameter group its own Adam rate."""
    rate = np.sqrt(args.batch)
    transforms = {
        "means": optax.adam(schedule),
        "log_scales": optax.adam(args.scales_lr * rate),
        "quats": optax.adam(args.quats_lr * rate),
        "sh_base": optax.adam(args.colors_lr * rate),
        "sh_rest": optax.adam(args.colors_lr * args.sh_rest_lr_scale * rate),
        "opacity_logit": optax.adam(args.opacity_lr * rate),
        "exposure": optax.adam(args.exposure_lr * rate),
    }
    return optax.multi_transform(transforms, {key: key for key in transforms})


def build_step_fn(
    args: Args,
    loss_fn: Callable[..., tuple[jax.Array, jax.Array]],
    optimizer: optax.GradientTransformation,
) -> Callable[..., tuple[dict[str, jax.Array], optax.OptState, jax.Array]]:
    """Build the jitted optimizer step."""

    @jax.jit
    def step(
        params: dict[str, jax.Array],
        opt_state: optax.OptState,
        targets: jax.Array,
        viewmats: jax.Array,
        rows: jax.Array,
    ) -> tuple[dict[str, jax.Array], optax.OptState, jax.Array]:
        (_, l1), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, targets, viewmats, params["exposure"][rows]
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        stepped = optax.apply_updates(params, updates)
        capped = cap_anisotropy(stepped["log_scales"], args.max_anisotropy)
        return {**stepped, "log_scales": capped}, opt_state, l1

    return step


def build_relocate_fn(args: Args) -> Callable[..., tuple[dict[str, jax.Array], optax.OptState]]:
    """Build the jitted relocation of the gaussians that went transparent."""
    binoms = splax.mcmc.make_binoms(51)

    @jax.jit
    def relocate(
        params: dict[str, jax.Array], opt_state: optax.OptState, key: jax.Array
    ) -> tuple[dict[str, jax.Array], optax.OptState]:
        moved, reset = splax.mcmc.relocate(
            key,
            params["means"],
            params["log_scales"],
            params["quats"],
            jnp.concatenate([params["sh_base"], params["sh_rest"]], axis=1),
            params["opacity_logit"],
            binoms,
            min_opacity=args.min_opacity,
        )
        means, log_scales, quats, sh_colors, opacity_logit = moved
        relocated = {
            "means": means,
            "log_scales": log_scales,
            "quats": quats,
            "sh_base": sh_colors[:, :1],
            "sh_rest": sh_colors[:, 1:],
            "opacity_logit": opacity_logit,
        }
        return {**params, **relocated}, reset_moments(opt_state, reset)

    return relocate


def build_inject_noise_fn(args: Args) -> Callable[..., dict[str, jax.Array]]:
    """Build the jitted noise injection that keeps the gaussians exploring."""

    @jax.jit
    def inject_noise(
        params: dict[str, jax.Array], key: jax.Array, scaler: jax.Array
    ) -> dict[str, jax.Array]:
        means = splax.mcmc.inject_noise(
            key,
            params["means"],
            params["log_scales"],
            params["quats"],
            params["opacity_logit"],
            scaler,
            min_opacity=args.min_opacity,
        )
        return {**params, "means": means}

    return inject_noise


def train(
    intrinsics: np.ndarray,
    dist: np.ndarray,
    viewmats: np.ndarray,
    points_xyz: np.ndarray,
    points_rgb: np.ndarray,
    points_track: np.ndarray,
    views: np.ndarray,
    mask: np.ndarray,
    args: Args,
) -> tuple[dict[str, jax.Array], dict[str, Any]]:
    """Fit the splat to the views.

    Args:
        intrinsics: Flat (fx, fy, cx, cy) of the camera the frames were taken with.
        dist: Brown-Conrady coefficients of that camera.
        viewmats: World to camera matrices, shape (views, 4, 4).
        points_xyz: Reconstructed points the gaussians are seeded from.
        points_rgb: Their colours as uint8.
        points_track: How many views observed each point.
        views: Training images as uint8.
        mask: Pixels that carry loss, with a trailing axis.
        args: Run configuration.

    Returns:
        The gaussians in metres, and a report of what they scored.
    """
    height, width = views.shape[1:3]
    # The intrinsics are flat (fx, fy, cx, cy), so downscaling the images divides all four of them.
    # A 3x3 matrix would have its homogeneous row divided as well.
    intrinsics = intrinsics / args.downscale
    normalized_viewmats, normalized_points, scale, centre = normalize(viewmats, points_xyz)
    about = np.array2string(centre, precision=3, floatmode="fixed")
    logger.info(f"normalized by {scale:.4f} units per metre about {about}")
    test = np.arange(len(views)) % args.test_every == 0
    train_idx, test_idx = np.flatnonzero(~test), np.flatnonzero(test)
    assert len(train_idx) and len(test_idx), "the hold out left nothing to train or score on"
    logger.info(f"{len(train_idx)} training views, {len(test_idx)} held out, {width}x{height} px")

    params = init_params(normalized_points, points_rgb, points_track, args)
    params["exposure"] = exposure_table(len(train_idx))
    device_viewmats = jnp.asarray(normalized_viewmats, jnp.float32)
    # The whole view set is well under a gigabyte as bytes, so it lives on the device and the loop
    # never touches the host. Indexing it per step on the host instead forces a synchronisation and
    # promotes the images to float64 on the way, which leaves the device idle for most of the step.
    device_views = jnp.asarray(np.ascontiguousarray(views))
    jax_mask = jnp.asarray(mask)

    schedule = learning_schedule(args)
    optimizer = build_optimizer(args, schedule)
    opt_state = optimizer.init(params)
    # Training views index the exposure table, held out views never do.
    exposure_row = np.zeros(len(views), np.int32)
    exposure_row[train_idx] = np.arange(len(train_idx))
    device_exposure_row = jnp.asarray(exposure_row)
    device_train_idx = jnp.asarray(train_idx)
    loss_fn = build_loss_fn((height, width), intrinsics, dist, jax_mask, args)
    step = build_step_fn(args, loss_fn, optimizer)
    relocate = build_relocate_fn(args)
    inject_noise = build_inject_noise_fn(args)

    def evaluate(current: dict[str, jax.Array], indices: np.ndarray) -> tuple[float, float]:
        """Score views by pixel error and by structure."""
        scores, structure = [], []
        for i in indices:
            image = render(current, device_viewmats[i], (height, width), intrinsics, dist)
            target = views[i] / 255.0
            rendered = np.where(mask, np.asarray(image), target)
            scores.append(psnr(rendered, target))
            # At a small gaussian budget a splat holds its pixel error down while losing the detail
            # a person actually looks at, which only the structural score catches.
            structure.append(
                float(dm_pix.ssim(jnp.asarray(target), jnp.clip(jnp.asarray(rendered), 0.0, 1.0)))
            )
        return float(np.mean(scores)), float(np.mean(structure))

    # Score held out views spread over the recording, not the first few, which are all one pose.
    probe = test_idx[:: max(1, len(test_idx) // 8)]
    key = jax.random.key(args.seed)
    history, start = [], time.time()
    refine_stop = int(0.9 * args.steps)  # the last tenth settles the gaussians where they landed
    relocate_every = max(1, round(args.relocate_every / args.batch))
    refine_start = round(args.refine_start / args.batch)
    for iteration in range(1, args.steps + 1):
        key, batch_key, relocate_key, noise_key = jax.random.split(key, 4)
        batch = jax.random.choice(batch_key, device_train_idx, (args.batch,), replace=False)
        params, opt_state, l1 = step(
            params,
            opt_state,
            device_views[batch].astype(jnp.float32) / 255.0,
            device_viewmats[batch],
            device_exposure_row[batch],
        )
        if refine_start < iteration < refine_stop and iteration % relocate_every == 0:
            params, opt_state = relocate(params, opt_state, relocate_key)
        params = inject_noise(params, noise_key, schedule(iteration) * args.noise_lr)

        if iteration % args.log_every == 0 or iteration == args.steps:
            score, _ = evaluate(params, probe)
            history.append({"step": iteration, "l1": float(l1), "test_psnr": score})
            logger.info(
                f"step {iteration:6d}  l1 {float(l1):.4f}  test psnr {score:.2f} dB  "
                f"{time.time() - start:.0f} s"
            )

    # A sample of the training views is enough to read the gap to the held out ones from
    train_psnr, train_ssim = evaluate(params, train_idx[:32])
    test_psnr, test_ssim = evaluate(params, test_idx)
    metric = to_metres(params, scale, centre)
    del metric["exposure"]
    return metric, {
        "settings": asdict(args),
        "history": history,
        "normalization_scale": scale,
        "normalization_centre": centre.tolist(),
        "train_psnr": train_psnr,
        "test_psnr": test_psnr,
        "train_ssim": train_ssim,
        "test_ssim": test_ssim,
        "n_train_views": len(train_idx),
        "n_test_views": len(test_idx),
        "resolution": [int(height), int(width)],
        "wall_time_s": time.time() - start,
    }


def main(**kwargs: Any):
    """Train a splat on a reconstruction and write it to disk.

    Args:
        kwargs: Fields of Args. Must contain the variant.
    """
    args = Args(**kwargs)
    assert 0 <= args.sh_degree <= 3, f"sh_degree must be between 0 and 3, got {args.sh_degree}"
    folder = SCENES_DIR / args.scene
    archive = folder / "scene.npz"
    assert archive.exists(), f"no scene archive at {archive}, run the export stage first"
    frames = Path(args.images).expanduser() if args.images else folder / "images"
    target = Path(args.out).expanduser() if args.out else folder / "raw"

    reconstruction = np.load(archive)
    names = reconstruction["names"]
    logger.info(
        f"training on {args.scene}: {len(names)} views, "
        f"{len(reconstruction['points_xyz'])} points, {args.n_gaussians} gaussians, "
        f"downscale {args.downscale}"
    )
    views = load_views(frames, names, args.downscale)
    if args.masks is None:
        mask = np.ones((*views.shape[1:3], 1), dtype=bool)
    else:
        mask = load_mask(Path(args.masks).expanduser(), names[0], views.shape[1:3])
    params, report = train(
        reconstruction["intrinsics"],
        reconstruction["dist"],
        reconstruction["viewmats"],
        reconstruction["points_xyz"],
        reconstruction["points_rgb"],
        reconstruction["points_track"],
        views,
        mask,
        args,
    )

    target.mkdir(parents=True, exist_ok=True)
    ply = target / f"{args.variant}.ply"
    splax.io.write_ply(
        ply,
        params["means"],
        params["log_scales"],
        params["quats"],
        jnp.concatenate([params["sh_base"], params["sh_rest"]], axis=1),
        params["opacity_logit"],
    )
    (target / f"{args.variant}.json").write_text(json.dumps(report, indent=1))
    logger.info(
        f"test psnr {report['test_psnr']:.2f} dB, ssim {report['test_ssim']:.4f}, "
        f"train psnr {report['train_psnr']:.2f} dB"
    )
    logger.info(
        f"wrote {ply} ({ply.stat().st_size / 1e6:.1f} MB) in {report['wall_time_s'] / 60:.1f} min"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fire.Fire(main)
