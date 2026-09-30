r"""Run the CrazySim sim2real conditions inside the CrazySimDocker container.

Build the image variants from the repository root (patches are baked in at build time, see
`sim2real/CrazySimDocker`):

    docker build -f crazyflow_experiments/sim2real/CrazySimDocker -t crazysim:base .
    docker build -f crazyflow_experiments/sim2real/CrazySimDocker --build-arg GZ_FIX=1 \\
        -t crazysim:gz-fix .
    docker build -f crazyflow_experiments/sim2real/CrazySimDocker --build-arg MJ_FIX=1 \\
        -t crazysim:mj-fix .

Then run the four paper conditions for one simulation variant, headless (no GPU/X required). The
crazyflow repository (branch exp) is mounted read-only, the runner pip-installs it on startup:

    docker run --rm -v <path/to/crazyflow_experiments>:/workspace:rw \\
        -v <path/to/crazyflow>:/crazyflow:ro crazysim:base \\
        python3 /workspace/crazyflow_experiments/sim2real/run_crazysim.py crazysim-gz
    ... crazysim:gz-fix python3 ... crazysim-gz-fix
    ... crazysim:base   python3 ... crazysim-mj
    ... crazysim:mj-fix python3 ... crazysim-mj-fix

The physics name selects the backend ("mj" component -> MuJoCo, otherwise Gazebo) and becomes the
recording suffix, so the results land in `results/sim2real/*_<physics>_NN.pkl`, numbered like the
deployments. Every condition is repeated `n_rollouts` times (from `config.toml`, override with
`--rollouts`), since CrazySim is not deterministic and the sim-to-real analysis pairs the k-th
rollout with the k-th real flight. Rollouts that fail the takeoff or crash during the trajectory
are discarded and repeated (at most `MAX_ATTEMPTS` attempts per condition), and the attempt counts
are written to `results/sim2real/<physics>_attempts.toml`. Each rollout runs in
a fresh child process; for the Gazebo backend this script also restarts the gz server, the model
spawn and the SITL firmware process in between. `--single <condition>` runs one rollout in-process instead (for the Gazebo backend the sim
and firmware must then already be running, e.g. via `sitl_singleagent.sh`).
"""

import argparse
import importlib.util
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("SCIPY_ARRAY_API", "1")  # must be set before any scipy import

logging.basicConfig(level=logging.INFO, format="%(asctime)s [run_crazysim] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CRAZYSIM_ROOT = Path(os.environ.get("CRAZYSIM_ROOT", "/root/CrazySim"))
FIRMWARE_DIR = CRAZYSIM_ROOT / "crazyflie-firmware"
GZ_DIR = FIRMWARE_DIR / "tools/crazyflie-simulation/simulator_files/gazebo"
PLUGIN_BUILD_DIR = FIRMWARE_DIR / "sitl_make/build/build_crazysim_gz"
CF2_BINARY = FIRMWARE_DIR / "sitl_make/build/cf2"
WORLD = GZ_DIR / "worlds/crazysim_default.sdf"
MODEL_SDF = Path("/tmp/crazyflie_0.sdf")
CF2_DIR = Path("/tmp/cf0")
CONDITION_TIMEOUT = 600  # s per condition, generous upper bound for takeoff + three cycles
TAKEOFF_FAILED_EXIT_CODE = 3  # child exit code for a rollout whose takeoff failed
CRASHED_EXIT_CODE = 4  # child exit code for a rollout that crashed during the trajectory
MAX_ATTEMPTS = 10  # attempts per condition before giving up on reaching n_rollouts

# The four sim2real conditions of the crazyflow paper (trajectory, plane, cycle time, scaling)
MODEL = "cf2x_L250"  # drone model of the CrazySim rollouts, names their results folder
CONDITIONS: dict[str, tuple[str, str, float, list[float]]] = {
    "ellipse_9s": ("ellipse", "xy", 9.0, [3.0, 3.0, 0.0]),
    "ellipse_3p5s": ("ellipse", "xy", 3.5, [3.0, 3.0, 0.0]),
    "figure8_10s": ("figure8", "xz", 10.0, [2.0, 0.0, 1.0]),
    "figure8_5p5s": ("figure8", "xz", 5.5, [2.0, 0.0, 1.0]),
}


def bootstrap() -> None:
    """Make crazyflow and this repository importable inside the pristine container."""
    if importlib.util.find_spec("crazyflow") is None:
        candidates = [Path("/crazyflow"), REPO_ROOT / "submodules" / "crazyflow"]
        crazyflow_dir = next((path for path in candidates if path.is_dir()), None)
        if crazyflow_dir is None:
            raise FileNotFoundError(
                f"crazyflow not found, mount it into the container at one of: {candidates}"
            )
        logger.info(f"Installing crazyflow from {crazyflow_dir}")
        # The build backend writes egg-info into the source tree, which is typically a read-only
        # mount here, so install from a writable copy.
        build_dir = Path("/tmp/crazyflow-src")
        if build_dir.exists():
            shutil.rmtree(build_dir)
        # Without the checkout's own environments and caches, whose symlinks point outside the mount
        shutil.copytree(
            crazyflow_dir,
            build_dir,
            symlinks=True,
            ignore=shutil.ignore_patterns(".git", ".venv", ".pixi", "__pycache__", "*.egg-info"),
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", str(build_dir)], check=True
        )
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    python_path = os.environ.get("PYTHONPATH", "")
    if str(REPO_ROOT) not in python_path.split(":"):
        os.environ["PYTHONPATH"] = f"{REPO_ROOT}:{python_path}"


def condition_config(physics: str, condition: str) -> dict:
    """Build the experiment config for one paper condition."""
    import toml

    trajectory, plane, cycle_time, scaling = CONDITIONS[condition]
    config = toml.load(Path(__file__).parent / "config.toml")
    config["drone"]["model"] = MODEL
    config["controller"]["type"] = "mellinger"
    config["controller"]["ctrl_freq"] = 100
    config["experiment"].update(
        trajectory=trajectory,
        plane=plane,
        time=cycle_time,
        scaling=scaling,
        origin=[0.0, 0.0, 1.25],
        n_runs=3,
        soft_start=True,
        yaw_mode="constant",
        yaw_setting=0,
    )
    config["sim"]["physics"] = physics
    config["sim"]["render"] = False
    return config


def recording_paths(physics: str, condition: str) -> list[Path]:
    """Numbered recordings of one condition, sorted by index."""
    from crazyflow_experiments.sim2real import RECORDINGS_DIR
    from crazyflow_experiments.sim2real.utils import generate_filename_from_config

    stem = generate_filename_from_config(condition_config(physics, condition))
    return sorted((RECORDINGS_DIR / MODEL).glob(f"{stem}_[0-9][0-9].pkl"))


def setup_gz_environment() -> None:
    """Extend the environment like CrazySim's setup_gz.bash does."""
    os.environ["GZ_SIM_SYSTEM_PLUGIN_PATH"] = (
        f"{os.environ.get('GZ_SIM_SYSTEM_PLUGIN_PATH', '')}:{PLUGIN_BUILD_DIR}"
    )
    os.environ["GZ_SIM_RESOURCE_PATH"] = (
        f"{os.environ.get('GZ_SIM_RESOURCE_PATH', '')}:{GZ_DIR / 'models'}:{GZ_DIR / 'worlds'}"
    )
    os.environ["LD_LIBRARY_PATH"] = f"{os.environ.get('LD_LIBRARY_PATH', '')}:{PLUGIN_BUILD_DIR}"


def generate_model_sdf() -> None:
    """Render the drone model SDF from the jinja template with the paper's UDP ports."""
    jinja = GZ_DIR / "models/crazyflie/model.sdf.jinja"
    subprocess.run(
        [
            "python3",
            str(GZ_DIR / "launch/jinja_gen.py"),
            str(jinja),
            str(GZ_DIR),
            "--cffirm_udp_port",
            "19950",
            "--cflib_udp_port",
            "19850",
            "--cf_id",
            "0",
            "--cf_name",
            "cf",
            "--output-file",
            str(MODEL_SDF),
        ],
        check=True,
    )


def start_gz_sim() -> subprocess.Popen:
    """Start the headless gz server and spawn the drone model."""
    gz_proc = subprocess.Popen(["gz", "sim", "-s", "-r", str(WORLD), "-v", "1"])
    time.sleep(6)
    request = (
        f'sdf_filename: "{MODEL_SDF}", pose: {{position: {{x:0, y:0, z: 0.1}}}}, '
        'name: "crazyflie_0", allow_renaming: 1'
    )
    subprocess.run(
        [
            "gz",
            "service",
            "-s",
            "/world/crazysim_default/create",
            "--reqtype",
            "gz.msgs.EntityFactory",
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            "5000",
            "--req",
            request,
        ],
        check=True,
    )
    time.sleep(2)
    return gz_proc


def start_cf2() -> subprocess.Popen:
    """Start the SITL firmware process."""
    CF2_DIR.mkdir(parents=True, exist_ok=True)
    with open(CF2_DIR / "cf2_out.log", "w") as cf2_log:
        cf2_proc = subprocess.Popen(
            [str(CF2_BINARY), "19950"], cwd=CF2_DIR, stdout=cf2_log, stderr=subprocess.STDOUT
        )
    time.sleep(4)
    return cf2_proc


def stop_gz_processes(gz_proc: subprocess.Popen, cf2_proc: subprocess.Popen) -> None:
    """Tear down the firmware and the gz server between conditions."""
    cf2_proc.terminate()
    subprocess.run(["pkill", "-x", "cf2"], check=False)
    gz_proc.terminate()
    subprocess.run(["pkill", "-f", "gz sim"], check=False)
    subprocess.run(["pkill", "-9", "ruby"], check=False)  # gz sim's launcher lingers otherwise
    time.sleep(3)


def run_condition(physics: str, condition: str, gz_backend: bool) -> str:
    """Run one rollout of a condition in a child process.

    A fresh child process per rollout keeps the UDP ports, cflib state and (for the MuJoCo
    backend) the known harmless segfault on interpreter shutdown from leaking into the next
    rollout.

    Returns:
        ``"completed"`` if the rollout wrote a recording, ``"takeoff_failed"`` if the drone did not
        reach the start pose (see `sim_crazysim.takeoff_succeeded`), ``"crashed"`` if it tilted
        over during the trajectory, ``"failed"`` otherwise.
    """
    logger.info(f"=== condition {condition} ({physics}) ===")
    CF2_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    gz_proc = cf2_proc = None
    if gz_backend:
        gz_proc = start_gz_sim()
        cf2_proc = start_cf2()

    child = subprocess.run(
        [
            "timeout",
            str(CONDITION_TIMEOUT),
            sys.executable,
            __file__,
            physics,
            "--single",
            condition,
        ],
        cwd=CF2_DIR,
        check=False,
    )

    if gz_backend:
        stop_gz_processes(gz_proc, cf2_proc)
    else:
        subprocess.run(["pkill", "-x", "cf2"], check=False)
        time.sleep(1)

    written = [p for p in recording_paths(physics, condition) if p.stat().st_mtime >= t_start]
    # The MuJoCo backend segfaults on interpreter shutdown after the recording is saved, so the
    # recording itself, not the exit code, decides success.
    if written:
        outcome = "completed"
    elif child.returncode == TAKEOFF_FAILED_EXIT_CODE:
        outcome = "takeoff_failed"
    elif child.returncode == CRASHED_EXIT_CODE:
        outcome = "crashed"
    else:
        outcome = "failed"
    logger.info(
        f"condition {condition}: exit code {child.returncode}, "
        f"recording {written[-1].name if written else 'MISSING'} ({outcome})"
    )
    return outcome


def run_single(physics: str, condition: str) -> None:
    """Run one rollout in the current process (the sim must already run for the gz backend).

    Exits with `TAKEOFF_FAILED_EXIT_CODE` or `CRASHED_EXIT_CODE` for a discarded rollout, so that
    the parent can tell the outcomes apart.
    """
    import crazyflow_experiments.sim2real.sim_crazysim as sim_crazysim

    outcome = sim_crazysim.rollout(condition_config(physics, condition))
    if outcome == "takeoff_failed":
        sys.exit(TAKEOFF_FAILED_EXIT_CODE)
    if outcome == "crashed":
        sys.exit(CRASHED_EXIT_CODE)


def record_condition(
    physics: str, condition: str, gz_backend: bool, n_rollouts: int
) -> dict[str, int]:
    """Repeat a condition until `n_rollouts` completed rollouts exist or `MAX_ATTEMPTS` are used.

    The CrazySim chain tips the drone over during the takeoff from the floor or during the
    trajectory in some rollouts (see the investigation handoff). Those attempts leave no recording
    and are counted, so that the recorded rollouts are the completed flights while the crash rate
    stays visible.

    Returns:
        Counts of completed rollouts, discarded takeoffs, in-flight crashes and other failures.
    """
    counts = {"completed": 0, "takeoff_failed": 0, "crashed": 0, "failed": 0}
    while counts["completed"] < n_rollouts and sum(counts.values()) < MAX_ATTEMPTS:
        counts[run_condition(physics, condition, gz_backend)] += 1
    return counts


def write_attempts(physics: str, attempts: dict[str, dict[str, int]]) -> None:
    """Store the attempt counts next to the recordings, one file per simulator variant."""
    import toml

    from crazyflow_experiments.sim2real import RECORDINGS_DIR

    path = RECORDINGS_DIR / MODEL / f"{physics}_attempts.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Rollout attempts per condition: completed rollouts, attempts discarded because the\n"
        "# drone did not take off or crashed during the trajectory, and other failures.\n\n"
        + toml.dumps(attempts),
        encoding="utf-8",
    )


def chown_results_to_workspace_owner() -> None:
    """Return ownership of the recordings to the host user (the container runs as root)."""
    from crazyflow_experiments.sim2real import RECORDINGS_DIR

    if os.geteuid() != 0:
        return
    owner = os.stat(REPO_ROOT)
    subprocess.run(
        ["chown", "-R", f"{owner.st_uid}:{owner.st_gid}", str(RECORDINGS_DIR)], check=False
    )


def main() -> None:
    """Parse the CLI arguments and run the requested conditions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("physics", help="Simulation name, e.g. crazysim-gz or crazysim-mj-fix")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--single", choices=CONDITIONS, help="Run one rollout in-process")
    parser.add_argument(
        "--rollouts", type=int, help="Rollouts per condition, defaults to n_rollouts in config.toml"
    )
    args = parser.parse_args()

    assert args.physics.split("-")[0] == "crazysim", f"Not a CrazySim name: {args.physics}"
    bootstrap()

    if args.single is not None:
        run_single(args.physics, args.single)
        return

    gz_backend = "mj" not in args.physics.split("-")
    if gz_backend:
        setup_gz_environment()
        generate_model_sdf()

    n_rollouts = args.rollouts
    if n_rollouts is None:
        n_rollouts = condition_config(args.physics, args.conditions[0])["sim"]["n_rollouts"]
    attempts = {}
    for condition in args.conditions:
        attempts[condition] = record_condition(args.physics, condition, gz_backend, n_rollouts)
    write_attempts(args.physics, attempts)
    chown_results_to_workspace_owner()

    logger.info(f"=== {args.physics} finished: {attempts} ===")
    if any(counts["completed"] < n_rollouts for counts in attempts.values()):
        logger.warning(f"Some conditions did not reach {n_rollouts} completed rollouts: {attempts}")


if __name__ == "__main__":
    main()
