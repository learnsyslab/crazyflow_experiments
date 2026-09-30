"""Make the SITL firmware's clock follow the simulator instead of the wall clock.

Applied inside the CrazySim image when `CrazySimDocker` is built with ``--build-arg SIM_TICK=1``
(``python3 crazysim_sim_tick.py <firmware dir>``).

The FreeRTOS POSIX port generates the tick from ``setitimer(ITIMER_REAL)``, i.e. from wall time,
while the stabilizer runs once per simulator IMU packet. Any real-time factor other than one then
mismatches the firmware's dt (EKF prediction, watchdogs, timestamps) against the physics: at a
real-time factor of 0.91 the Kalman filter's attitude estimate is inflated by 10 %. With this patch
the wall-clock timer signal keeps delivering the ticks, but once the first IMU packet has arrived
the tick count advances by the number of IMU packets received since the previous signal instead of
by one. One tick then equals one simulator step (1 ms at the 1 kHz sensor rate of the MuJoCo
backend) regardless of how fast the simulator runs, with at most one wall-clock millisecond of
latency. The firmware boots on the wall clock until the simulator connects. If the sensor stream
pauses for more than two timer periods, the tick falls back to the wall clock for the duration of
the gap: the SITL socket task sleeps one tick whenever the timer signal interrupts its ``poll()``,
and it is the only task that can receive the packet that would otherwise produce that tick.
"""

import sys
from pathlib import Path

PORT_MARKER = "static void vPortSystemTickHandler( int sig )\n{"
PORT_PATCH = """/* SIM_TICK patch: count scheduler ticks from the simulator's IMU packets. The wall-clock
 * timer signal still delivers the ticks, but the tick count advances by the number of IMU packets
 * received since the last signal, i.e. by simulation time. */
static volatile BaseType_t xSimTickEnabled = pdFALSE;
static volatile uint32_t ulSimTicksPending = 0;
static uint32_t ulSimIdleSignals = 0;

void vPortSimTick( void )
{
	xSimTickEnabled = pdTRUE;
	__atomic_add_fetch( &ulSimTicksPending, 1, __ATOMIC_SEQ_CST );
}
/*-----------------------------------------------------------*/

static void vPortSystemTickHandler( int sig )
{"""
PORT_INCREMENT_MARKER = "\t\txTaskIncrementTick();\n"
PORT_INCREMENT_PATCH = """\t\tif( xSimTickEnabled == pdFALSE )
\t\t{
\t\t\txTaskIncrementTick();
\t\t}
\t\telse
\t\t{
\t\t\tuint32_t ulTicks = __atomic_exchange_n( &ulSimTicksPending, 0, __ATOMIC_SEQ_CST );
\t\t\tif( ulTicks == 0 )
\t\t\t{
\t\t\t\t/* No IMU packet since the last signal. After two such signals fall back to a
\t\t\t\t * wall-clock tick so that a gap in the sensor stream cannot deadlock the firmware:
\t\t\t\t * the socket task sleeps one tick after an interrupted poll() and is the only
\t\t\t\t * task that can receive the packet that would produce that tick. */
\t\t\t\tif( ++ulSimIdleSignals >= 2 )
\t\t\t\t{
\t\t\t\t\txTaskIncrementTick();
\t\t\t\t}
\t\t\t}
\t\t\telse
\t\t\t{
\t\t\t\tulSimIdleSignals = 0;
\t\t\t\twhile( ulTicks-- > 0 )
\t\t\t\t{
\t\t\t\t\txTaskIncrementTick();
\t\t\t\t}
\t\t\t}
\t\t}
"""

SENSORS_MARKER = (
    "      case SENSOR_GYRO_ACC_SIM:\n        processAccGyroMeasurements(&(p.data[1]));\n"
)
SENSORS_PATCH = (
    "      case SENSOR_GYRO_ACC_SIM:\n"
    "        processAccGyroMeasurements(&(p.data[1]));\n"
    "        vPortSimTick();  // SIM_TICK patch: one scheduler tick per simulator IMU packet\n"
)
SENSORS_INCLUDE_MARKER = '#include "semphr.h"\n'
SENSORS_INCLUDE_PATCH = (
    '#include "semphr.h"\n\n// SIM_TICK patch, see crazysim_sim_tick.py\nvoid vPortSimTick(void);\n'
)


def patch(path: Path, marker: str, replacement: str) -> None:
    """Replace the single occurrence of ``marker`` in the file, failing loudly otherwise."""
    text = path.read_text()
    if text.count(marker) != 1:
        raise RuntimeError(f"{path}: expected exactly one occurrence of the patch marker")
    path.write_text(text.replace(marker, replacement))


def main(firmware_dir: Path) -> None:
    """Patch the FreeRTOS POSIX port and the SITL sensor driver."""
    port = firmware_dir / "vendor/FreeRTOS/portable/ThirdParty/GCC/Posix/port.c"
    patch(port, PORT_MARKER, PORT_PATCH)
    patch(port, PORT_INCREMENT_MARKER, PORT_INCREMENT_PATCH)
    sensors = firmware_dir / "src/hal/src/sensors_sitl.c"
    patch(sensors, SENSORS_INCLUDE_MARKER, SENSORS_INCLUDE_PATCH)
    patch(sensors, SENSORS_MARKER, SENSORS_PATCH)
    print("SIM_TICK patch applied")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
