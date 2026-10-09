#!/usr/bin/env bash
# strata-native.sh: the SYCL-built engine as a drop-in `exe` for serve/server.py and sycl/serve/server_intel.py
# (--engine strata) on a host that has oneAPI installed and no Docker. The binary runs directly, with stdin/stdout
# attached (the serve protocol is lines on those pipes) and stderr going to the server's log (the config's "log";
# without that key the server discards it). Paths in the config's args are this host's: there is no /work mount here.
#   INTEL_ONEAPI_ROOT       where oneAPI is installed, its setvars.sh is sourced   (default: /opt/intel/oneapi)
#   STRATA_SYCL_BIN         the engine binary, relative to the repo   (default: build-sycl-aot/strata, else build-sycl/strata)
#   ONEAPI_DEVICE_SELECTOR  the SYCL device   (default: level_zero:gpu; a caller's value wins. The VM has an NVIDIA
#                           card too, so the Arc is named; level_zero:* for two Arcs and --layer-split, #423)
#   SYCL_CACHE_PERSISTENT   always 0: the persistent JIT cache crashes on Xe2 during the first compile (docs/INTEL.md)
#   ZES_ENABLE_SYSMAN       always 1: the engine's free-VRAM query needs sysman
# Every variable of this script's environment reaches the engine as it is (the server puts the config's "env" block
# there), so STRATA_MIRROR_MIB, IGC_*, UR_*, NEOReadDebugKeys, ... need no forwarding. The port's own defaults below
# are overridden the same way, e.g. "env": {"STRATA_VERIFY_NO_HOST": "0"}: the engine tests its STRATA_* switches by
# presence, so a value of 0 (or empty) means "not set" and such a variable is unset here, not passed on.
set -eo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)                 # the repo
oneapi=${INTEL_ONEAPI_ROOT:-/opt/intel/oneapi}
if [ ! -r "$oneapi/setvars.sh" ]; then
    echo "strata-native.sh: $oneapi/setvars.sh not found (install oneAPI or set INTEL_ONEAPI_ROOT)" >&2
    exit 1
fi
set +u                                                    # setvars.sh reads unset variables (OCL_ICD_FILENAMES, 2026.1.1)
# shellcheck disable=SC1091
source "$oneapi/setvars.sh" --force >/dev/null            # its chatter must not reach the serve pipe on stdout
set -u
export SYCL_CACHE_PERSISTENT=0 ZES_ENABLE_SYSMAN=1
export ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}
# the port's run-time switches: the device-built verify plan, no host handshakes in it (NO_HOST is for a card on the
# xe driver, required whenever part of the experts is mirrored; an i915 Arc A-series runs without it, docs/INTEL.md),
# and the stager threads (8 here: the VM has 8 vCPUs; the Docker wrapper's 12 was the author's desktop)
declare -A setting=([STRATA_VERIFY_DEVICE_PLAN]=1 [STRATA_VERIFY_NO_HOST]=1 [STRATA_STAGER_THREADS]=8)
for k in "${!setting[@]}"; do
    [ -n "${!k+x}" ] || export "$k=${setting[$k]}"       # the environment's value wins, 0 and "" included
done
for k in $(compgen -e STRATA_); do                        # 0 or empty means "not set": unset, not passed on
    case "${!k}" in 0|"") unset "$k" ;; esac
done
# the engine is the first thing to go when the VM runs out of RAM (the Docker wrapper used --oom-score-adj 1000);
# raising the score never needs a capability, but the write is best-effort all the same
echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true
bin=${STRATA_SYCL_BIN:-}
if [ -z "$bin" ]; then
    for b in build-sycl-aot/strata build-sycl/strata; do
        [ -x "$here/$b" ] && { bin=$b; break; }
    done
fi
case "$bin" in /*) ;; *) bin=$here/$bin ;; esac
if [ ! -x "$bin" ]; then
    echo "strata-native.sh: no engine binary at $bin (build it: docs/INTEL_ARC.md, or set STRATA_SYCL_BIN)" >&2
    exit 1
fi
cd "$here"
exec "$bin" "$@"
