# Vision
We train a policy for vision-based flight using crazyflow's Gaussian splatting support. The policy
sees a grayscale camera frame, the IMU and its goal, and uses attitude or relative state control. We train it with DAgger on a splat of the real lab. Alongside the action, the network predicts the drone's pose and motion, which we can use for VIO.

To build a scene from your own recording, follow `crazyflow_experiments/vision/reconstruct/README.md`.

```bash
pixi run -e vision python crazyflow_experiments/vision/train.py --control attitude
pixi run -e vision python crazyflow_experiments/vision/sim.py --policy <run>
```

To deploy, launch the tracker and the estimator as described in the [main readme](../../README.md), publish the camera feed, and
then use
```bash
pixi run -e vision python crazyflow_experiments/vision/ros/cam_publisher.py
pixi run -e vision python crazyflow_experiments/vision/deploy.py --policy <run>
```
