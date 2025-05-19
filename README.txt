Reverse engineering nsight compute

essentially a ptrace version of https://github.com/geohot/cuda_ioctl_sniffer

use make to build sniffer and saxpy (target program)

cp -r /opt/nvidia/nsight-compute/2024.3.2 ncu-local
ncu-local/ncu ./saxpy
./sniffer <pid>

/usr/local/cuda-12.6/bin/ncu -- bash script
 \-> exec /opt/nvidia/nsight-compute/2024.3.2/target/linux-desktop-glibc_2_11_3-x64/ncu
     \-> clone/exec /opt/nvidia/nsight-compute/2024.3.2/target/linux-desktop-glibc_2_11_3-x64/./TreeLauncherSubreaper
         \-> clone/exec ncu target (ie. ./saxpy)

TreeLauncherSubreaper called with envs:
NV_CUDA_START_SUSPENDED=1
NV_COMPUTE_PROFILER_PERFWORKS_DIR=/home/chris/fun/ncurev/ncu-target/target/linux-desktop-glibc_2_11_3-x64/.
LOGNAME=chris
NV_NSIGHT_INJECTION_PORT_RANGE_BEGIN=0
NVIDIA_PROCESS_INJECTION_CRASH_REPORTING=0
PWD=/home/chris/fun/ncurev/ncu-target
TERM_PROGRAM_VERSION=3.4
USER=chris
NV_TPS_LAUNCH_TOKEN=Interactive Profile
HOME=/home/chris
NV_NSIGHT_INJECTION_PORT_RANGE_END=63
_=/home/chris/fun/ncurev/ncu-target/./ncu
SHLVL=2
LD_LIBRARY_PATH=/home/chris/fun/ncurev/ncu-target/target/linux-desktop-glibc_2_11_3-x64/.
LANG=C.UTF-8
OLDPWD=/home/chris/fun/ncurev/ncu-target/target
NVIDIA_PROCESS_INJECTION_XML_TARGET_SETTINGS=0
NV_NSIGHT_INJECTION_PORT_BASE=49152
NV_NSIGHT_INJECTION_TRANSPORT_TYPE=uds
NVIDIA-PROCESS-TRACKING-CONFIGURATION=blocking-rpc-events:
  - BeforePosixSpawn
  - AfterExit
  - BeforeFork
registered-rpc-events:
  - BeforePosixSpawn
  - PidOfChildFound
  - BeforeFork
  - AfterExit
  - AfterExitCodeFound
connection-name: /tmp/NVIDIA-treetracker-xD9H3k/pipe-1
uid: 1
tree-tracker-id: /tmp/NVIDIA-treetracker-xD9H3k
is-tracking-root-only: false
launcher-preload-prepend: /home/chris/fun/ncurev/ncu-target/target/linux-desktop-glibc_2_11_3-x64/./libInterceptorInjectionTarget.so:/home/chris/fun/ncurev/ncu-target/target/linux-desktop-glibc_2_11_3-x64/./libTreeLauncherTargetInjection.so
not-launcher-preload-prepend: /home/chris/fun/ncurev/ncu-target/target/linux-desktop-glibc_2_11_3-x64/./libcuda-injection.so
preload-update-preload-prepend: /home/chris/fun/ncurev/ncu-target/target/linux-desktop-glibc_2_11_3-x64/./libTreeLauncherTargetUpdatePreloadInjection.so
pipe-descriptor:
  - 9
  - 10
  - ""
  - 0
  - false
