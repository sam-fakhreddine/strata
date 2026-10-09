# B70 runbook: building, running and flipping the SYCL engine on the Arc Pro B70

Fork `sam-fakhreddine/strata`. Base: branch `b70` at `fb58e0d` (upstream `Niko1221/Strata` main, engine 0.1.41). This file is the operating procedure: what to build, how to start the engine on the LLM VM, how to switch between the optimization branches, and what to do when something goes wrong. The companion files are `docs/B70_TUNING.md` (every switch, what it changes, how to measure it) and `docs/B70_WORKPLAN.md` (the change-by-change design behind each branch). The homelab side (leases, Steward, Sentinel, the owner's gates) is in `sam-fakhreddine/homelab`, `docs/research/strata-b70-flash-next-20261009-022842.md`.

Written 2026-10-09 in a container without a GPU. Every build result below is a compile result (device code to SPIR-V), not a run on the card. Nothing in this file has been measured on the B70 yet; the tuning document says what to measure first.

## 1. The branches

All branches start from `b70` (`fb58e0d`). Each optimization is one branch with one commit, opt-in behind an environment switch, the default path byte-identical to `b70`. `b70-all` merges every branch that compiles, so one binary carries every switch.

| Branch | Commit | Switch | Default | What it changes | Output | Compiles (container, no GPU) |
|---|---|---|---|---|---|---|
| `b70-infra` | `424a8bd` | none (files only) | n/a | `sycl/serve/strata-native.sh` launcher, `strata-flash.service.example`, `strata-flash.json.example`, `sycl/tools/strata_timings.py`, INTEL_ARC.md section | none | yes (0 errors) |
| `b70-d1-profiling` | `845d9a6` | CMake `-DSTRATA_SYCL_PROFILING_QUEUES=OFF` (build-time, default ON) | ON = today | Removes `enable_profiling` from every dpct queue; timing printers read 0 and say "(profiling off)" | identical | yes (0 errors, ON); the OFF engine build of `b70-all` also 0 errors |
| `b70-d4-iq2xs-multi` | `08e3eb5` | `STRATA_IQ2XS_MULTI=1`, `STRATA_IQ1M_MULTI=1` | off | Decode-once lane path (`Multi<17>`, `Multi<29>`) for IQ2_XS and IQ1_M expert dots: one weight fetch per 4 routed entries instead of per entry | bitwise | yes (0 errors), `iq_multi_parity`, `native_expert_parity` build |
| `b70-d6-small` | `b71bc92` | `STRATA_INPUT_COPY_MULTI=1`, `STRATA_ARGMAX_MULTI=1`; `STRATA_DECODE_TIMING=1` prints nodes | off | One multi-copy for the window inputs; multi-block argmax; node count and us/node in the decode timing line | bitwise | yes (0 errors), `verify_parity` builds |
| `b70-d7-mirror-counter` | `593f299` | `STRATA_MIRROR_STATS=1` (summary) or `=2` (per layer) | off | Per-request line: routed entries served from VRAM, the mirror, or neither, worst layers | identical (diagnostic) | yes (0 errors), `resident_plan_parity` builds |
| `b70-d8-adapt-guard` | `40b62d8` | `STRATA_ADAPT_WITH_MIRROR=1` keeps the tier | guard on | With a host mirror under the device plan, the adaptive tier (`--adapt-every`, default 4) is turned off with a WARNING, because an expert it evicts from VRAM never reaches the mirror table | identical in practice (the tier never swaps under `NO_HOST=1`) | yes (0 errors) |
| `b70-d9-mirror-cap` | `0c4975a` | `STRATA_FILL_FADVISE=1` | off; the cap change is always on | Mirror cap from the cgroup-aware available-memory figure (log line `mirror cap N MiB from ...`); optional `posix_fadvise(DONTNEED)` after each fill read; O(N) miss list | identical | yes (0 errors) |
| `b70-p1-grouping-bubble` | `7af9ef6` | `STRATA_GROUP_ASYNC=1` (polled) or `=2` (event wait) | off | Prompt path: the shared expert runs while the host sorts the routed ids, instead of a queue drain per MoE layer | identical | yes (0 errors) |
| `b70-p2-dequant-occupancy` | `ce83f4e` | `STRATA_DEQUANT_WG=2`, `4` or `8` | off (1) | Expert dequant for the prompt GEMMs: N superblocks per work-group, same lane mapping | bitwise | yes (0 errors), `iq_multi_parity`, `dequant_bench` build |
| `b70-p6-r-pass-fuse` | `bced943` | `STRATA_GR_FUSE_READ=1` | off | Prompt path: the hyper-connection mix reads the BF16 image the norm already wrote instead of a second FP32 pass over the residual | rounding-level, not bitwise (quality gate) | yes (0 errors), `gr_prompt_read_parity` builds |
| `b70-t2a-prefetch` | `12f21dd` | `STRATA_PF_SLOTS=N` (0 = off), `STRATA_PF_QUEUE=main|side` | off | Decode with a mirror: predicts layer l+1's experts from layer l's input, copies the predicted mirrored ones into a per-layer VRAM slot ring one layer ahead, the plan points at the slot | identical (same bytes) | yes (0 errors), `resident_plan_parity`, `mirror_prefetch_parity` build |
| `b70-all` | `3c76f9d` | all of the above | as above | Integration branch: every branch above that compiles, merged in the order listed | per switch | yes (0 errors) with all seven parity targets; the profiling-OFF engine build also 0 errors |

"Compiles" was checked with an incremental Ninja build of the `strata` target (and the named parity targets) per branch in this container, oneAPI 2026.1.1, `-DSTRATA_SYCL_PARITY=ON`, device code to SPIR-V. The two things a compile cannot check are listed per branch in section 7.

Parity targets per branch, run on the card (`ONEAPI_DEVICE_SELECTOR=level_zero:gpu ctest --test-dir <build> -R <name> --output-on-failure`):

| Branch | Parity target | What it checks |
|---|---|---|
| d4 | `iq_multi_parity` (check 3), `native_expert_parity` | opt-in lane path memcmp-equal to the per-entry path and the old kernels, groups of 0 to 8 entries |
| d6 | `verify_parity` | the window inputs and the argmax |
| d7 | `resident_plan_parity` | the plan kernel with the stats pointer set and unset |
| p2 | `iq_multi_parity` (`check_dequant_wg`), `dequant_bench [type] [experts] [wg]` | dequant output bitwise at wg 1, 2, 4, 8 for every expert format, shapes 67 x 2560 and 1280 x 2560; the bench prints the FNV hashes, which must match `./dequant_bench 21` |
| p6 | `gr_prompt_read_parity` | the fused mix within the BF16 rounding bound of the FP32 pair, T = 1, 7, 64, 333 |
| t2a | `mirror_prefetch_parity`, `resident_plan_parity` (ring case) | every occupied ring slot holds its source blob, no expert in two slots, at most S copies, no re-copy; the plan pointer equals the ring address for an expert in the ring, else the mirror |
| d1, d8, d9, p1, infra | none new; `ctest` whole set once per build | regressions |

## 2. Prerequisites on the VM (arc-llm, VM 800)

Already present (verified 2026-10-09 from the host): oneAPI 2026.1.1 under `/opt/intel/oneapi` (dpcpp, oneMKL, oneDNN), NEO 26.05.37020.3, libze1 1.28.2, libigc2 2.28.4, kernel 7.0.0-34 with the `xe` driver, cmake 4.2.3, ninja, g++, Python 3. The B70 is PCI `8086:e223`, DRM card1 (card0 is the RTX 5060 Ti). Not present: `intel-ocloc` (AOT needs it; an apt install on the GPU VM is an owner gate).

Checks before any build:

```sh
set +u; source /opt/intel/oneapi/setvars.sh; set -u
sycl-ls                                    # must list "Intel(R) Arc(TM) Pro B70 Graphics" under level_zero
icpx --version                             # 2026.1.x
ulimit -l                                  # unlimited for the shell that runs the engine (the mirror is locked memory)
```

Network: the build needs a llama.cpp checkout at `3cf03257f219afbe7334045ff7c6a06ac68c627d` for ggml. Clone it once outside the build tree; never let CMake fetch at configure time (it clones the full history, unpinned by our rules):

```sh
git clone https://github.com/ggml-org/llama.cpp.git ~/src/llama.cpp
git -C ~/src/llama.cpp checkout 3cf03257f219afbe7334045ff7c6a06ac68c627d
```

The container build used exactly this checkout.

## 3. Building

### 3.1 One build tree per binary you want to flip to

The switches are run-time, so one binary of `b70-all` covers every A/B except two:

- `b70-d1-profiling` is a build-time choice (`-DSTRATA_SYCL_PROFILING_QUEUES=OFF`). It needs its own build tree.
- Comparing a branch against pure `b70` (to prove a switch's "off" path is really the old path) needs a `b70` binary.

Recommended layout under `~/src/strata` (one checkout, several build trees; the launcher picks the binary through `STRATA_SYCL_BIN`):

| Build dir | Checkout | Configure extras | Purpose |
|---|---|---|---|
| `build-b70` | `b70` | none | reference binary |
| `build-all` | `b70-all` | none | every run-time switch |
| `build-all-noprof` | `b70-all` | `-DSTRATA_SYCL_PROFILING_QUEUES=OFF` | the D1 A/B |
| `build-all-aot` | `b70-all` | `-DSTRATA_SYCL_AOT=bmg-g31` | production, once `intel-ocloc` is installed |

A build tree is bound to the checkout it was configured from, so switch branches with a second worktree rather than `git checkout` in place:

```sh
cd ~/src/strata
git worktree add ../strata-all b70-all          # one worktree per branch you build
```

### 3.2 JIT build (today, no ocloc)

```sh
cd ~/src/strata-all
set +u; source /opt/intel/oneapi/setvars.sh; set -u
export SYCL_CACHE_PERSISTENT=0
cmake -S sycl -B build-all -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
      -DSTRATA_GGML_DIR=$HOME/src/llama.cpp
cmake --build build-all --target strata -j 8 2>&1 | tee build-all/build.log
grep -c 'error:' build-all/build.log            # must print 0
```

About 123 translation units; the container (4 cores) took roughly 40 minutes for a clean build of the engine alone. The parity tests and benches are extra targets (`cmake --build build-all` with no `--target` builds all of them, about 157 targets). Build at least the parity targets listed in section 1 for the branches you intend to flip.

For the D1 tree add `-DSTRATA_SYCL_PROFILING_QUEUES=OFF` to the configure line and use `build-all-noprof`.

With `-S sycl` the binary is `<build>/strata`. (With the top-level `-DSTRATA_ENABLE_SYCL=ON` it would be `<build>/sycl/strata`, which the launcher does not look for.)

The first start of a JIT binary compiles about 47 s of kernels. `SYCL_CACHE_PERSISTENT` must stay 0: the persistent cache segfaults on Xe2 during the first compile.

### 3.3 AOT build (after the owner's yes to `apt install intel-ocloc`)

```sh
ocloc ids bmg-g31                                # must list the B70's device id
cmake -S sycl -B build-all-aot -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
      -DSTRATA_GGML_DIR=$HOME/src/llama.cpp -DSTRATA_SYCL_AOT=bmg-g31
cmake --build build-all-aot --target strata -j 8 2>&1 | tee build-all-aot/build.log
```

`intel-ocloc` must match the installed IGC (libigc2 2.28.4). Record its version in `infra/arc-llm-vm/VERSIONS.md` (homelab) with the Strata sha and the llama.cpp sha.

### 3.4 Parity tests (card time, only inside a window with the B70 free)

```sh
ONEAPI_DEVICE_SELECTOR=level_zero:gpu SYCL_CACHE_PERSISTENT=0 ctest --test-dir build-all --output-on-failure
```

Expected: the kernel tests pass; `iq_parity` and `ple_parity` report missing fixtures (they need model files). `iq_multi_parity` must print "bitwise" for every format at both shapes (p2's check) and pass check 3 (d4's). Record the pass list in the bench report.

## 4. Model files

Target layout on the VM, all under `/opt/llm`, owned by `ubuntu`:

| Path | Content |
|---|---|
| `/opt/llm/models/strata/IQ2_XS/` | the two GGUF shards, published names unchanged |
| `/opt/llm/strata/packs/iq2_xs/` | the pack (`iq_pack.py` output: `index.txt`, `dense.bin`, `conversions.json`, `native_experts.txt`, `tokenizer/`) |
| `/opt/llm/strata/mtp-bf16/`, `.../rt/` | the MTP draft layer tensors and its runtime pack |
| `/opt/llm/strata/strata-flash.json` | the serve config (section 5) |
| `/opt/llm/strata/expert-profile-live.bin` | the profile written by `--expert-profile-save` |

Download (resumable; about 68 GB for IQ2_XS; the VM needs egress to huggingface.co once). Revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09` of `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` is what `setup.py` pins:

```sh
R=https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ2_XS
cd /opt/llm/models/strata/IQ2_XS
for i in 1 2; do curl -L --fail --retry 5 -C - -o Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-0000$i-of-00002.gguf "$R/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-0000$i-of-00002.gguf"; done
```

Hash check (setup.py does none for this family): read each file's LFS oid from `https://huggingface.co/api/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ2_XS` (field `lfs.oid`), compare with `sha256sum` (the check that matters; the oid is the file's sha256). `data/gguf_fingerprints.json` (`qwen/IQ2_XS`, `bytes` 68,015,068,160) counts tensor bytes only: the two files are about 11 MB larger (68,026,093,024 measured 2026-10-09) because of the GGUF headers and metadata, so do not compare file sizes with it. Write `<file>.sha256` beside each file. Shard 2 is the `per_layer_token_embd` table shared by every quant of this model: hard-link it if another quant is ever downloaded.

Pack and MTP (the engine's own tools, in the repo's `.venv` from `setup.sh`, or any Python 3.10+ with `numpy`):

```sh
export STRATA_GGUF_PY=$HOME/src/llama.cpp/gguf-py
python -c "import sys; sys.path.insert(0,'$STRATA_GGUF_PY'); import gguf; gguf.GGMLQuantizationType.Q2_0"   # gate: must not raise
python tools/iq_pack.py --gguf /opt/llm/models/strata/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf --out /opt/llm/strata/packs/iq2_xs
python tools/mtp_fetch.py fetch  --out /opt/llm/strata/mtp-bf16        # 4.9 GB, sha256 per tensor built in; STRATA_MTP_REVISION unset
python tools/mtp_fetch.py verify --out /opt/llm/strata/mtp-bf16        # exit 0
python tools/mtp_pack.py --src /opt/llm/strata/mtp-bf16 --experts q2_0 --out /opt/llm/strata/mtp-bf16/mtp-q2_0.gguf
python tools/mtp_rt.py --gguf /opt/llm/strata/mtp-bf16/mtp-q2_0.gguf --out /opt/llm/strata/mtp-bf16/rt
```

If the gguf-py gate fails (Q2_0 = type 42 unknown), clone the commit `setup.py` pins for its own `third_party/llama.cpp` (`LLAMA_CPP_COMMIT` in `setup.py`) into a second directory and point `STRATA_GGUF_PY` at it. Not verified: whether the engine needs `data/draft_vocab.bin` copied into `rt/`; read the engine's start log for a draft-vocab line on the first run.

The expert profile is `data/expert-profile.bin` in the checkout (the original model's; the Coder uses `expert-profile-coder.bin`).

## 5. Running

### 5.1 By hand (the reference run, the bench runs)

The command every measurement uses. `STRATA_VERIFY_NO_HOST=1` is required whenever part of the experts is mirrored in RAM (on xe the per-layer host handshake is not visible across the bus and the logits turn NaN a few tokens in).

```sh
set +u; source /opt/intel/oneapi/setvars.sh; set -u
export SYCL_CACHE_PERSISTENT=0 ZES_ENABLE_SYSMAN=1 ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export STRATA_VERIFY_DEVICE_PLAN=1 STRATA_VERIFY_NO_HOST=1 STRATA_STAGER_THREADS=8
M=/opt/llm/models/strata/IQ2_XS
build-all/strata --pack /opt/llm/strata/packs/iq2_xs \
  --native $M/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf \
  --ple-gguf $M/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf \
  --expert-profile data/expert-profile.bin --expert-cache auto --stream-experts \
  --prefill 4096 --spec 4 --spec-min-p 0.5 --mtp /opt/llm/strata/mtp-bf16/rt \
  --max-context 32768 --kv int8 --vram-reserve-mib 1024 --ple-io direct \
  --tokens-file <ids file> --max-new 256 --greedy
```

Read the start lines: `N experts resident` (expect about 18.3k for IQ2_XS), `MiB of VRAM free with everything loaded` (must not say `LOW`), `mirror cap N MiB from ...` (d9), the mirror size and fill time (about 8.4 GiB), `WARN` lines about aliased arena pages (the engine retries by itself), the `PCIe probe` line (about 35 GB/s on the Gen5 x16 link with the probe fix, `fix/sycl-pcie-probe-pinned`, upstream #1746; without it the probe times a pageable copy and reads about 6 to 8 GB/s, which sets `pcie_frac` near 0.2: pass `--pcie-frac 0.55` until the fix is in). Expected decode on IQ2_XS: 55 to 70 tok/s on the 20-token prompt, from the author's Gen5 figures; measured 2026-10-09 on our B70: 60.5 (reference binary) to 64 (AOT `b70-all`). The token files are comma-separated ids; the community prompts `bench/results/2026-10-07-community-arc-b65-v01402/benchy-short.ids` (20 tokens) and `benchy-long.ids` (2,185) are the 20-token and 2,185-token prompts below. Anything under 45 means the mirror is being read more than planned: turn on `STRATA_MIRROR_STATS=1` (d7).

### 5.2 Served (the production shape)

`b70-infra` adds the three files. Install them:

```sh
cp sycl/serve/strata-flash.json.example /opt/llm/strata/strata-flash.json      # edit paths to section 4's layout
sudo cp sycl/serve/strata-flash.service.example /etc/systemd/system/strata-flash.service   # edit /opt/llm/strata/Strata to the checkout path
sudo install -m 600 -o root -g root /dev/stdin /etc/homelab/secrets/strata.env <<<"STRATA_API_KEY=<key>"   # pipe, never paste the key in a terminal that logs
sudo systemctl daemon-reload && sudo systemctl enable --now strata-flash
curl -s 127.0.0.1:8097/health
```

What the pieces do:

- `sycl/serve/strata-native.sh` is the config's `exe`. It sources `setvars.sh` (`INTEL_ONEAPI_ROOT` overrides the location), exports `SYCL_CACHE_PERSISTENT=0`, `ZES_ENABLE_SYSMAN=1`, `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` (a caller's value wins), applies the port's defaults (`STRATA_VERIFY_DEVICE_PLAN=1`, `STRATA_VERIFY_NO_HOST=1`, `STRATA_STAGER_THREADS=8`), unsets every `STRATA_*` whose value is `0` or empty (the engine tests switches by presence), sets `oom_score_adj` 1000, and execs `build-sycl-aot/strata`, else `build-sycl/strata`, else `STRATA_SYCL_BIN`. Set `STRATA_SYCL_BIN` in the unit's `Environment=` to point at `build-all/strata` or whichever tree you are flipping to.
- The config's `"env"` block is where the run-time switches go (section 6). Its `"log"` key is where the engine's stderr goes; without it the server discards it.
- The unit `Conflicts=` every unit that can own the B70 (`qwen38 cyber-llm llama-swap llama-server llama-gemma shieldgemma`), fails closed unless the card's power cap reads 180 W (`ExecStartPre`), sets `LimitMEMLOCK=infinity` and `OOMScoreAdjust=1000`, and reads the API key from `/etc/homelab/secrets/strata.env`. `TimeoutStartSec=300` covers JIT compile plus the mirror fill. Uncomment `Environment=STRATA_MIRROR_MIB=9216` on a shared VM; unset, the engine takes the available memory minus 4 GiB once at start and a later co-tenant gets squeezed.

Operationally, the homelab side still owns the switch: until Steward has an `llm-mode strata` (homelab plan, step 5), start and stop this unit only under a Steward `llm:*` lease, with `qwen38` or `cyber-llm` stopped first and Sentinel's `qwen_heal` held off by the lease. A bare `systemctl start strata-flash` outside a lease will have `Conflicts=` stop the production model and Sentinel will restart it under you.

### 5.3 Timings through the API

```sh
python sycl/tools/strata_timings.py --base http://127.0.0.1:8097 --suite tools/b70-tuning/prompts/suite.json --repeat 3 --out timings.json
```

(the suite file is in the homelab repo, `tools/b70-tuning/prompts/suite.json`; the driver is standard library only and prints each response's `timings.prompt_per_second` and `timings.predicted_per_second` plus a median row.)

## 6. Flipping between optimizations

Every switch is read once at engine start (static initializers), so a flip is a restart, about 45 s (AOT) to 90 s (JIT) plus the mirror fill. Three ways to set one:

1. By hand: `STRATA_DEQUANT_WG=4 build-all/strata ...`.
2. Served: in `strata-flash.json`, `"env": {"STRATA_VERIFY_NO_HOST": "1", "STRATA_VERIFY_DEVICE_PLAN": "1", "STRATA_DEQUANT_WG": "4"}`, then `systemctl restart strata-flash`. A value of `"0"` or `""` unsets the variable (the launcher drops it).
3. Build-time (D1 only): point `STRATA_SYCL_BIN` at `build-all-noprof/strata`.

The flip matrix. "Side" is the path the switch changes; a decode switch is measured on the 20-token and 2,185-token prompts with 256 greedy tokens, a prompt switch on the 2,185-token and 8,000-token prompts.

| To test | Set | Side | Must hold |
|---|---|---|---|
| D1 no profiling queues | binary from `build-all-noprof` | decode + prompt | tokens identical; `STRATA_DECODE_TIMING` reads 0 in that binary, so measure tok/s from the request timings instead |
| D4 IQ2_XS multi | `STRATA_IQ2XS_MULTI=1` | decode | tokens identical (bitwise) |
| D4 IQ1_M multi (Coder) | `STRATA_IQ1M_MULTI=1` | decode | tokens identical |
| D6 input copy | `STRATA_INPUT_COPY_MULTI=1` | decode | tokens identical |
| D6 argmax | `STRATA_ARGMAX_MULTI=1` | decode | tokens identical |
| D7 mirror stats | `STRATA_MIRROR_STATS=1` | diagnostic | no change in tokens; a per-request line in the log; costs a few atomics per window, turn off for the final number |
| D8 adaptive tier back on | `STRATA_ADAPT_WITH_MIRROR=1` | decode | only to reproduce the hazard; expect no change under `NO_HOST=1` |
| D9 page cache | `STRATA_FILL_FADVISE=1` | startup | `free -m` after the fill shows the page cache not grown by the model size; mirror fill time unchanged or better |
| P1 grouping bubble | `STRATA_GROUP_ASYNC=1`, then `=2` | prompt | tokens identical; `=2` is the form that hung the stager under the Level Zero v2 adapter, so run it last and watch for a hang |
| P2 dequant work-group | `STRATA_DEQUANT_WG=2`, `4`, `8` | prompt | tokens identical (bitwise); `dequant_bench 21 0 N` hashes equal |
| P6 fused residual read | `STRATA_GR_FUSE_READ=1` | prompt | same text, near-tie flips allowed; the 10-prompt `mg_norepeat` gate passes |
| T2a mirror prefetch | `STRATA_PF_SLOTS=8`, then `16` | decode (mirrored quants) | tokens identical; D7's `ring hits` over copies is the hit rate; the start log names the ring's VRAM; `STRATA_PF_QUEUE=side` only after `main` is clean |
| Expert cache +2.9 GiB | `--prefill-borrow` (flag, not env) | decode with a mirror | about 1.4 s refill after each prompt; resident count in the start log rises; decode tok/s on the 2,185-token prompt |
| Tier 0, author's switches | `STRATA_SH_STREAM=0`, `STRATA_LFUSE=1`, `--spec 6`, Level Zero knobs | decode | see `docs/B70_TUNING.md` section 3 |

Protocol for one A/B (from `docs/INTEL.md`, the author's method): fresh engine per side, one warm-up request, then interleaved pairs off/on, medians of at least 3 runs per side, `STRATA_DECODE_TIMING=1` (decode) or `STRATA_PREFILL_TIMING=1` (prompt) in the log, and `diff` of the printed token ids between the sides. Record the numbers in a `bench/results/<date>-b70-<branch>/README.md` in the fork, as the community reports do, with the exact command, the card, the driver and oneAPI versions.

Rollback: unset the switch and restart; or point `STRATA_SYCL_BIN` back at `build-b70/strata`. Nothing persists on disk except `--expert-profile-save`'s file, which the default `data/expert-profile.bin` replaces.

## 7. What the container build could not check

Compile checks cannot see run-time behaviour. Per branch, the first thing to confirm on the card:

| Branch | Risk | Check |
|---|---|---|
| d1 | the stage profiler, `--stage-timing` and both timing lines read 0 in the OFF binary by design; a code path that reads an event's profiling info without going through `strata::prof_ns` would throw at run time | run the OFF binary once with `STRATA_DECODE_TIMING=1` and `STRATA_PREFILL_TIMING=1`; expect "(profiling off)" and no exception |
| d1 | `sycl/tools/fixups.py` is not idempotent on a clean tree for two pre-existing entries (`get_int_from_table_16` argument and the `iq3s_grid` cast in `iq_kernels.dp.cpp` get rewritten twice); the committed tree is correct, re-running fixups.py after a re-migration needs this fixed first | `git diff` after a fixups.py run must be empty |
| d4 | the IQ1_M `finish_u` assumes the compiler contracts `delta * sumy` the same way in both paths | `iq_multi_parity` check 3 for type 29; `native_expert_parity` |
| d6 | `copy_from_mapped_multi` ordering against the window's first kernel | `verify_parity`; tokens identical over 256 greedy tokens |
| d7 | the stats pointer is set before the window graphs are captured; a capture before the setter would count nothing | `STRATA_MIRROR_STATS=1` prints non-zero VRAM counts on the first request |
| d8 | the WARNING fires on the standard IQ2_XS config (mirror present, `--adapt-every 4` default); confirm no tok/s change against `STRATA_ADAPT_WITH_MIRROR=1` | one A/B pair |
| d9 | `host_available_memory` on the VM (no cgroup limit) must equal `MemAvailable`; the log line says which source it used | read `mirror cap N MiB from <source>` at start |
| p1 | the polled sequence number must be written by the queue in order; `=2` hung the stager once under Level Zero v2 | the 8,000-token prompt completes; `STRATA_PREFILL_TIMING=1` host-grouping share drops |
| p2 | `reqd_sub_group_size(32)` with 256-lane groups: IGC must not spill; a `has_capability_or_fail` failure prints and falls back to 1 | `dequant_bench 21 0 8` runs and its GB/s is not lower than wg 1 |
| p6 | host `sycl::half(f)` assumed RTE like the device's `convert<half, rte>` in the parity test | `gr_prompt_read_parity` passes; then the quality gate |
| infra | the power-cap sysfs path on the xe driver (the cap may sit on a tile's hwmon) | `systemctl start strata-flash` does not fail in `ExecStartPre`; if it does, confirm the path against `/usr/local/sbin/b70-power-cap.sh` and fix the unit |

## 8. Troubleshooting

| Symptom | Cause | Action |
|---|---|---|
| Logits NaN a few tokens in, different every run | `STRATA_VERIFY_NO_HOST` unset with a mirror on xe | set it (the launcher does); never run a mirrored model without it |
| NaN at the first layer that routes to one expert, reproducible | aliased 2 MiB pages in a big allocation (xe driver) | the engine's `malloc_device_guarded` retries by itself and prints a warning per retry; if it still happens, `STRATA_VERIFY_ALL_SLOTS=1` reads every slot back against the GGUF once; report the driver version |
| Segfault during the first start | `SYCL_CACHE_PERSISTENT=1` | keep it 0 (the launcher does) |
| `MiB of VRAM free with everything loaded: LOW` | too much reserved for the context, prompt buffers or the cache | raise `--vram-reserve-mib`, lower `--prefill` or `--max-context`; never start a run that says LOW (the xe livelock on VRAM over-commit takes the whole VM down) |
| Decode far under 50 tok/s on IQ2_XS | more mirror reads than planned, or a co-tenant on the card | `STRATA_MIRROR_STATS=1`; `systemctl is-active shieldgemma qwen38 cyber-llm llama-server llama-gemma llama-swap` all inactive; the power cap at 180 W |
| `strata generate: WARNING: --adapt-every 4 with --stream-experts and a host mirror ...` | d8's guard | expected on IQ2_XS; nothing to do |
| Engine refuses `STRATA_DEQUANT_WG=3` | only 2, 4, 8 are valid | the engine runs with the default (1) and says so |
| Mirror smaller than expected, `mirror cap N MiB from cgroup` | the unit or a container has a memory limit | raise `MemoryMax` or set `STRATA_MIRROR_MIB` |
| `strata-flash` fails in `ExecStartPre` | the B70's power cap does not read 180 W, or the sysfs path differs | `systemctl status b70-power-cap`; check the hwmon path (section 7, infra) |
| Prompt hangs with `STRATA_GROUP_ASYNC=2` | the event-wait form under the Level Zero v2 adapter | use `=1` or unset |
| Kernel log line matching `CAT error|Fault response|GT[0-9]: reset|Schedule disable failed|wedged|timedout_job|Engine reset|hang` | GPU fault | stop the engine; Sentinel's `qwen_heal` pattern; do not restart until the card is clean (`sudo journalctl -k --since -5min`) |
| Timing lines read 0, "(profiling off)" | the D1 OFF binary | expected; measure through the API's `timings` instead |

## 9. The trial window, in short

The full checklist with the homelab gates is in the homelab research document (step 4) and `docs/B70_WORKPLAN.md`. The engine-side sequence:

1. Steward lease `llm:*`; `qwen38` (or `cyber-llm`) and `shieldgemma` stopped (owner's decision 2026-10-09: kid chat is not live, shieldgemma is paused for the window; later it moves to its own Arc A380 and the B70 stays single-tenant); `llama-server`, `llama-gemma`, `llama-swap` inactive; power cap 180 W; 40+ GB available; kernel-log mark taken.
2. `ctest` (section 3.4).
3. Reference run by hand (section 5.1) on `build-b70/strata`: resident count, VRAM free, PCIe probe, mirror size, decode tok/s on the 20-token and 2,185-token prompts. This is the baseline every flip is judged against.
4. `build-all/strata` with no switches: must match the baseline tokens and tok/s (the default path is byte-identical by construction; this run proves it).
5. The flips, in the order the tuning document gives (section 6 there), interleaved pairs, medians of 3.
6. Served run (section 5.2), the timings suite, the quality eval through the homelab's `evals.py` target.
7. Stop Strata; restart `shieldgemma`, then the recorded mode's unit; verify on the backends' own ports; `maintenance end`.

Abort on: a new matching kernel-log line, VRAM free under 256 MiB, swap in use, the 5060 Ti's Gemma latency moving.
