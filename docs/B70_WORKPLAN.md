# B70 workplan: the SYCL engine on the Arc Pro B70, change by change

Fork `sam-fakhreddine/strata` at `fb58e0d` (upstream `Niko1221/Strata` main, engine 0.1.41). Companion to the two research documents in `sam-fakhreddine/homelab` (`docs/research/strata-b70-flash-next-20261009-022842.md`, `docs/research/strata-b70-engine-plan-20261009-025812.md`). Those say what and why; this file says how, item by item, with the lines to touch, the switch, the gate and the bench. Every line number was read from this checkout on 2026-10-09; re-check after any merge from upstream.

## How to use this file

- One item is one branch and one PR, upstream-shaped: opt-in behind an env switch, the default path byte-identical, measured on the card, the number and the machine in the PR text (`AGENTS.md` upstream). Our fork's branch for the work is `b70`; each item branches from it as `b70-<id>` (a `b70/` prefix is not possible beside the `b70` branch name); `b70-all` merges them.
- Every item ends with the same three gates: (1) the parity tests it names pass (`ONEAPI_DEVICE_SELECTOR=level_zero:gpu ctest --test-dir build-sycl -R <names>`), (2) output identity on the three standard prompts (20-token Fibonacci, the 2,185-token code-and-prose text, the 8,000-token repeat; 256 greedy tokens; `diff` of the token ids with the switch off and on), (3) a bench row (`STRATA_DECODE_TIMING=1` for decode items, `STRATA_PREFILL_TIMING=1` for prompt items, interleaved off/on pairs of whole engine runs, medians of 3 or more).
- Where an item changes the summation order (marked in its "Must not change"), gate (2) relaxes to: same text, a near-tie flip after N tokens allowed, as the author accepted for `STRATA_SELECT_GEMM` and the aligned-load kernels.
- Everything lives under `sycl/`. Mechanical edits to migrated `.dp.cpp` files go into `sycl/tools/fixups.py` as an idempotent entry with a reason, so a re-migration from upstream reproduces them; hand edits go into the migrated copy and are listed in the PR.
- No item touches a machine: the build and the bench run in the trial window under a Steward `llm:*` lease, with the B70 single-tenant (section "Trial window" below).

## Conventions and facts that every item relies on

| Fact | Value | Source |
|---|---|---|
| Card | Arc Pro B70, 32 GB, 608 GB/s, 32 Xe cores, 256 XMX engines, PCI id 8086:e223, PCIe 5.0 x16 root port, xe driver, kernel 7.0.0-34 | homelab doc 1, section 1 |
| Runtime | oneAPI 2026.1.1 (dpcpp, oneMKL 2026.1, oneDNN present), NEO 26.05.37020.3, libze1 1.28.2, libigc2 2.28.4, no ocloc yet | homelab doc 1, section 1 |
| Device selection | `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` (the VM also carries an RTX 5060 Ti as DRM card0) | `sycl/tools/Dockerfile:8` hard-codes `level_zero:0`; the two-card free-memory fix is `a8534781` |
| Sub-groups | 32 lanes forced everywhere (`-fsycl-default-sub-group-size=32`, `sycl/CMakeLists.txt:61`); Xe2 native is 16 | INTEL.md |
| Decode round | about 40 ms for a 6-token window on the Coder; 2,000 to 2,600 graph nodes; 3.3 to 3.6 tokens accepted on code | INTEL.md, homelab doc 2 section 1 |
| Reference speeds, one B70, Gen5 link, engine 0.1.39-sycl | IQ2_XS 70.0 / 77.5 / 78.5 / 71.8 / 65.4 tok/s at 20 / 2,185 / 8,000 / 40,000 / 128,000 tokens; Coder 77.3 / 74.6 / 71.9 / 67.6 / 63.9 / 56.0 (256K) | `docs/INTEL_PERFORMANCE.md` on `maxfridbe/intel-arc-0.1.40` |
| Mirror | one `sycl::malloc_host` allocation (`sycl/src/core/gguf_expert_source.cpp:127`), filled by 8 threads in profile order; cap `STRATA_MIRROR_MIB` else MemAvailable minus 4 GiB read once (`generate.cpp:4705-4712`); experts read in-kernel from host USM, 64-byte lines, no prefetch | homelab doc 2 section 1 |
| Three standard prompts | `sycl/bench/v1` on the author's branch (20, 2,185, 8,000, 40,000, 128,000, 256,000 tokens) | `sycl/benchy.sh` on `maxfridbe/intel-arc-0.1.40` |

## Tier 0: switches that exist today, one bench window, no code

Run each as interleaved pairs of whole engine runs (fresh engine per side, warmed by one request), 256 greedy tokens on the 20-token and 2,185-token prompts, medians of 3; compare `STRATA_DECODE_TIMING=1` fields (verify, commit, draft ms per window) and the token ids.

| Switch | Off (default) | On | What to expect | Where it lives |
|---|---|---|---|---|
| `STRATA_SH_STREAM` | the shared expert forks onto a side queue per layer (3 barrier nodes a layer) | `=0`: one queue, a linear graph, 144 fewer nodes a round | +30% decode on HIP (M); unknown on Xe2; tokens identical | `sycl/src/core/verify.cpp:98-108` |
| `STRATA_LFUSE` | separate gate GEMV, gate+up, sigmoid scale | `=1`: gate row in the router GEMV, gate+up one launch, sigmoid in the combine; "all bitwise" upstream | about 190 fewer nodes a round; never measured on SYCL | `verify.cpp:67-69`, `shared_expert.dp.cpp:14-41` |
| `--spec 6` | windows of 6 (`--spec 4` plus the suffix drafter) | windows of 8 | +6.5% on code, -1.5% on prose (M) | flag |
| borrowing at 32K: `--prefill 4096` without `--no-prefill-borrow` | below 32K of context the prompt owns about 2.9 GiB of buffers (the port borrows by default only above 32K; there is no `--prefill-borrow` flag, `--no-prefill-borrow` is the opt-out, `generate.cpp`) | force borrowing at 32K: the prompt lends about 700 cache slots, refilled after (needs the small code change noted in I3 if the threshold is not a flag) | about +3 tok/s decode for models with a mirror; about 1.4 s refill per prompt from NVMe | flag or a one-line threshold change |
| `STRATA_MIRROR_MIB=<n>` | MemAvailable minus 4 GiB at start | an explicit cap | no squeeze when a co-tenant starts later | `generate.cpp:4705-4712` |
| `STRATA_STAGER_THREADS=8`, `STRATA_IO_THREADS=64` | 2 to 3 stager threads on 8 to 12 vCPUs | explicit | prompt path with streamed experts | `sycl/src/prefill/prefill.cpp:984-992`, `direct_file.cpp:61-67` |
| `--ple-io direct` | default | keep it; `ram` needs 28.8 GB of RAM and buys time-to-first-token, not throughput | | |
| `--expert-profile-save <file>` | off | the resident set follows our traffic | fewer mirror reads after a week | `generate.cpp:3326, 7630, 8186` |
| Level Zero knobs, decode only | adapter defaults | `SYCL_UR_USE_LEVEL_ZERO_V2=0` and `=1`, `UR_L0_USE_IMMEDIATE_COMMANDLISTS=1`, `UR_L0_USE_COPY_ENGINE=0`, `UR_L0_IN_ORDER_BARRIER_BY_SIGNAL=1` | unknown; one run each | runtime |
| `STRATA_VERIFY_ALL_SLOTS=1` | off | reads every cache slot back against the GGUF once | catches the xe aliased-page class of bug; run once per driver or kernel change | `expert_cache.cpp` |
| `STRATA_MMVQ_A2=0` | aligned-load Q6_K kernel on | off | sanity check that the aligned kernels are active (decode should drop by about 10 tok/s) | INTEL.md |

## Trial window checklist (owner at the console; Steward `llm:*` lease; about 2 hours)

Owner's decisions of 2026-10-09: kid chat is not live, so `shieldgemma` is paused for the window; later it moves to its own Arc A380 and the B70 stays single-tenant.

1. `steward llm-mode status` recorded (normal, or cyber with its `until`); `systemctl list-units 'qwen38*' 'cyber-llm*' 'shieldgemma*' 'llama*'` recorded; `evals.py run --target qwen38 --label pre-strata` captured while `qwen38` is still up.
2. Lease: `cd /opt/homelab/apps/steward && node dist/src/main.js maintenance start --minutes 150 --reason "Strata B70 trial" --target 'llm:*'`.
3. On the VM, under the lease: stop the mode owner (`sudo systemctl stop qwen38`, or `cyber-llm`), `sudo systemctl stop shieldgemma`; `systemctl is-active llama-server llama-gemma llama-swap` all inactive; `systemctl is-active b70-power-cap` active and the sysfs cap reads 180 W; `free -m` 40+ GB available; `START=$(date -u +%FT%TZ)`; `sudo journalctl -k --since -15min -o cat | grep -E -c 'CAT error|Fault response|GT[0-9]: reset|Schedule disable failed|wedged|timedout_job|Engine reset|\bhang\b'` is 0.
4. `ctest --test-dir build-sycl` (the parity tests need the card).
5. Run A: the greedy reference runs (section I3 config, `--max-new 256 --greedy`), Coder if downloaded, then IQ2_XS; read `experts resident`, `MiB of VRAM free with everything loaded` (never `LOW`), the `PCIe probe` line (expect well above 26.5 GB/s), the mirror size.
6. Run B: served through the native wrapper (I1), the sweep (20, 2,185, 8,000, 40,000, 128,000 tokens, 3 runs each), the 12-prompt suite through `strata_timings.py` (I4), `evals.py run --target strata --label strata-iq2xs` through the port forward, then the Tier 0 A/Bs.
7. Abort on: a new kernel line since `$START` matching the pattern in step 3; VRAM free under 256 MiB at start; swap in use; fast-llm's latency moving.
8. After, still under the lease: stop Strata; `sudo systemctl start shieldgemma`; start the recorded mode's unit (`sudo systemctl start qwen38 && sudo systemctl start arc-gateway`, or `sudo systemctl start cyber-llm`; the mode file is left alone); verify on the backends' own ports (`127.0.0.1:8090/health` and `/props` alias `qwen38`, or `:8095/health`; `:8094/health`); one real completion through arc-queue; then `maintenance end`. Not `steward llm-mode normal|cyber` here: both refuse under a foreign lease and `cyber` exits "already on".

## Order of work

1. Window 1: build (I2), model (I3), wrapper (I1), checklist above, Tier 0 A/Bs, the baseline profile (`STRATA_DECODE_TIMING=1`, one `unitrace -d` run). Everything after is judged against this profile.
2. Week 1, decode: D1 and D7 first (they change what the profile means), then D4, D2, D3, D5, D6, D8, D9, in the order the profile ranks them.
3. Week 2, prompt: port the author's items (I5), then P2, P3, P5, P1, P4, P6.
4. Then by the numbers: T2a (prefetch) only if IQ3_S is wanted; T2c (INT8 weights) only if prompts dominate; T2b (SIMD16) after D4.
5. Operations integration (I6) before Strata carries any traffic.

## Branch status (2026-10-09, this container, no GPU)

Every item below that has a branch was implemented on 2026-10-09 and compile-checked in a container with oneAPI 2026.1.1 (device code to SPIR-V, `-DSTRATA_SYCL_PARITY=ON`, the pinned llama.cpp `3cf03257` for ggml). "Compiles" means the `strata` target and the named parity targets build with 0 errors from an incremental Ninja build; nothing has run on a card. The flip matrix and the per-branch run-time checks are in `docs/B70_RUNBOOK.md`; the switches and what to measure are in `docs/B70_TUNING.md`.

| Item | Branch | Commit | Switch | Compiles | Parity targets built |
|---|---|---|---|---|---|
| I1, I4(b) | `b70-infra` | `807c503` | none | yes | none |
| D1 | `b70-d1-profiling` | `845d9a6` | CMake `STRATA_SYCL_PROFILING_QUEUES` (default ON) | yes (ON); OFF: see `b70-all` | none |
| D4 | `b70-d4-iq2xs-multi` | `08e3eb5` | `STRATA_IQ2XS_MULTI=1`, `STRATA_IQ1M_MULTI=1` | yes | `iq_multi_parity`, `native_expert_parity` |
| D6 | `b70-d6-small` | `b71bc92` | `STRATA_INPUT_COPY_MULTI=1`, `STRATA_ARGMAX_MULTI=1` | yes | `verify_parity` |
| D7 | `b70-d7-mirror-counter` | `593f299` | `STRATA_MIRROR_STATS=1|2` | yes | `resident_plan_parity` |
| D8 | `b70-d8-adapt-guard` | `40b62d8` | guard on; `STRATA_ADAPT_WITH_MIRROR=1` | yes | none |
| D9 | `b70-d9-mirror-cap` | `0c4975a` | `STRATA_FILL_FADVISE=1`; cap source always on | yes | none |
| P1 | `b70-p1-grouping-bubble` | `7af9ef6` | `STRATA_GROUP_ASYNC=1|2` | yes | none |
| P2 | `b70-p2-dequant-occupancy` | `ce83f4e` | `STRATA_DEQUANT_WG=2|4|8` | yes | `iq_multi_parity`, `dequant_bench` |
| P6 | `b70-p6-r-pass-fuse` | `bced943` | `STRATA_GR_FUSE_READ=1` | yes | `gr_prompt_read_parity` |
| T2a | `b70-t2a-prefetch` | `12f21dd` | `STRATA_PF_SLOTS=N` (0 = off), `STRATA_PF_QUEUE=main` (`side` parsed, falls back to main) | yes | `resident_plan_parity`, `mirror_prefetch_parity` |
| all | `b70-all` | `3c76f9d` | every run-time switch above | yes (ON at `7b613a5` with all six parity targets; OFF engine build at `7b613a5`; the final head with T2a re-verified incrementally) | all of the above |

Not implemented (design only, below): D2 (QFUSE port), D3 (fused router + top-10 + plan), D5 (drafter graphs), C2 (PLE flag wait), P3 (grouped dequant and batched GEMM), P4 (oneDNN FP16 matmul), P5 (chunked DeltaNet), T2b (SIMD16 build), T2c (INT8 experts), I2 and I3 (procedures, now in the runbook), I4(a) and (c), I5 (porting the author's items), I6 (homelab integration). The order to pick them up is in "Order of work" above and `docs/B70_TUNING.md` section 6; the measurements of the first window decide which of D2, D3, D5 is worth its risk.

Merge conflicts resolved in `b70-all`: `include/strata/kernels/iq_kernels.hpp`, `sycl/src/kernels/cuda/iq_kernels.dp.cpp` and `sycl/src/kernels/iq_multi_parity.cpp` between P2 and D4 (both add declarations, definitions and a test loop at the same place; both sides kept). One follow-up commit on `b70-all` guards the profiling define in P6's new parity test the way D1 guards every other translation unit.

# Part A. Decode path


Scope: fork `sam-fakhreddine/strata` at `fb58e0d` (engine 0.1.41), read only. Every path is relative to the repo root; every line number was read from the file on 2026-10-09. Items are the review's D1, D2, D3, D5, D6, the correctness note C2 (the PLE flag wait), and the two Tier 0 A/Bs. D4 (IQ2_XS / IQ1_M decode-once lanes) is specified elsewhere and is not here.

Common ground for every item:

- The port's rules: everything lives under `sycl/`; a migrated `.dp.cpp` is edited through an entry of `sycl/tools/fixups.py` when the edit is mechanical, by hand otherwise (hand edits survive a re-migration through the 3-way merge in INTEL.md "Keeping up with upstream"); upstream files under `src/` and `include/` are never edited. Port-only declarations go in a header under `sycl/include/strata/kernels/` (model: `sycl/include/strata/kernels/resident_plan_mirror.hpp`). Port-only device-code shared between two translation units goes in a `.dp.hpp` under `sycl/src/kernels/cuda/` (model: `sycl/src/kernels/cuda/s26_tsum.dp.hpp`).
- Baseline numbers (INTEL.md:293-297, from unitrace): a decode round is 80 to 85% kernel time and about 5 us of launch gap per node over 2,400 to 2,600 nodes; `sycl/include/strata/sycl_doorbell.hpp:48` counts 2,366 nodes in one window graph. The review's section 1 puts the node gap at 12 to 31% of a 40 ms round and says the per-node cost is the first thing to measure. INTEL.md:458-470 ("Decode round 2") is the method every item below reuses: 19-token prompt, 256 greedy tokens, two runs each, output tokens identical.
- "The three standard prompts" means what INTEL.md uses for output identity (INTEL.md:564-566, :645-646): the Coder on the 19/20-token Fibonacci prompt (`bench/results/2026-10-07-community-arc-b65-v01402/benchy-short.ids`, 20 ids) and on the 2,184-token prompt (`benchy-long.ids`, 2,184 ids), 256 greedy tokens each, plus IQ2_XS on the same two prompts. INTEL.md also used a 40K prompt with borrowing; add it when the item touches the prompt path (none here does).
- The by-hand run every number is measured with (INTEL.md:232-239):

  ```
  STRATA_VERIFY_DEVICE_PLAN=1 STRATA_VERIFY_NO_HOST=1 \
  build-sycl-aot/strata --pack <iq pack> --native <shard1> --ple-gguf <shard2> \
      --expert-profile data/expert-profile-coder.bin --expert-cache auto --stream-experts \
      --prefill auto --spec 4 --spec-min-p 0.5 --max-context 8192 --tokens <ids> --max-new 256 --greedy
  ```

  `STRATA_DECODE_TIMING=1` adds one `strata decode timing:` line per request (`sycl/src/program/generate.cpp:11176-11184`): windows, avg T, tokens/window, ms/window = verify (GPU-reach wait + per-layer host [plan actq jobs CPU] + stage) + commit/emit + draft; the verify / commit / draft split is measured on the host around the three calls at `generate.cpp:11139-11145`. `STRATA_VERIFY_NODES=1` prints the node count of each captured window graph (`sycl/src/core/verify.cpp:1642-1673`; kernels are not named on SYCL, only counted). unitrace: `sycl/tools/Dockerfile.unitrace`, `unitrace -d` around the engine (INTEL.md:268-271). `sycl/rank_kernels.py` is named there but is not in this fork (it is on the author's branch; not verified).
- Interleaved A/B pairs, medians of 5, as INTEL.md:776-778 does. Node counts are compared with `STRATA_VERIFY_NODES=1`. Output identity: diff the printed token ids of the whole 256-token run.
- On the homelab side this is a window with qwen38 stopped and a Steward `llm:*` lease (AGENTS.md); the engineer on the card does not need more than that sentence.

---

### D1. Drop `DPCT_PROFILING_ENABLED` from the engine's translation units; replace `dpct::sync_barrier` with explicit barriers

**Goal and measured motivation.** Every migrated translation unit starts with `#define DPCT_PROFILING_ENABLED` (123 files under `sycl/`, of which 67 are engine TUs under `sycl/src` and the rest parity tests and benches; also the port header `sycl/include/strata/kernels/q8_1_finite.hpp:12`). With it defined, dpct creates every queue with `sycl::property::queue::enable_profiling()` (`sycl/include/dpct/device.hpp:687-689`), the default in-order queue included (`device.hpp:670-673`). On Level Zero a profiling queue needs a timestamp event per command, which is a plausible part of the 5 us per node gap over 2,400 to 2,600 nodes (INTEL.md:297), i.e. of the 12 to 31% of a round the review puts on launch gaps. The size of the effect is unknown until measured: `sycl/probe/launch_cost.cpp` exists to measure exactly this (host submit cost with and without `enable_profiling`, with and without a kept event, and with `ext_oneapi_submit_barrier`) and INTEL.md records no result from it. The review lists this first because it changes what every later profile means.

**Where.**
- `sycl/include/dpct/device.hpp:678-693` (`create_queue_impl`: the `#ifdef DPCT_PROFILING_ENABLED` property), `:616-633` (`create_queue`, `create_in_order_queue`, `create_out_of_order_queue`), `:670-673` (`init_queues`: the default queues), `:974-993` (`dpct::sync_barrier`).
- Engine queue creation, all `dpct::get_current_device().create_queue(true)`: `sycl/src/core/verify.cpp:601` (`copy_`), `:613` (`cs_`), `:621` (`sh_cs_`); `sycl/src/core/mtp.cpp:505, 515, 524` (`cs_`, `side_`); `sycl/src/program/generate.cpp:2868, 3217, 3225, 6882, 6893, 7139, 7644, 11992`; `sycl/src/core/session.cpp:243, 1234`.
- `dpct::sync_barrier` call sites in engine code: `verify.cpp:1222, 1285, 1322, 2424, 3445, 3675`; `mtp.cpp:1018, 1042, 1827, 1836, 1843`; `generate.cpp:1541, 1550, 7844, 7849, 7924, 9834, 10363, 10385, 10503, 10507, 12127`; `session.cpp:444, 448, 452, 456, 521, 545, 627, 652, 950, 1119`; `sycl/src/prefill/prefill.cpp:1976, 2025, 2389, 2411, 2455, 2470, 2573, 2641, 3269, 3454, 3579, 3611, 3643, 3706, 3728, 3742`; `sycl/src/kernels/cuda/native_mmvq.dp.cpp:2884, 2886, 2911, 2913` (a bench inside the kernel TU).
- Already explicit: `verify.cpp:1223, 1286, 1420` and `mtp.cpp:1020, 1064` use `->ext_oneapi_submit_barrier({*ev})` for the cross-queue waits.
- Consumers of profiling timestamps that would break: `generate.cpp:1555-1561` (the PCIe probe's `get_profiling_info<command_start/command_end>`), `session.cpp:467-484, 553-554, 662-693` (the stage profiler behind `--gpu-stages` / `STRATA_VERIFY_PROFILE`, `verify.cpp:566`).
- `sycl/tools/fixups.py` (the place for the mechanical edit; `all_sources()` at `:22-23`, the per-file loop at `:28-60`), `sycl/CMakeLists.txt:62-66` (`add_compile_options`), `sycl/probe/launch_cost.cpp`.

**Current behaviour.** `device.hpp:678-693`:

```cpp
  template <class... Properties>
  sycl::queue *create_queue_impl(bool enable_exception_handler,
                                 Properties... properties) {
    ...
    _queues.push_back(std::make_shared<sycl::queue>(
        _ctx, *this, eh,
        sycl::property_list(
#ifdef DPCT_PROFILING_ENABLED
            sycl::property::queue::enable_profiling(),
#endif
            properties...)));
```

`device.hpp:974-993`: `sync_barrier(event*, queue*)` first does `dpct::get_current_device().queues_wait_and_throw()` when the queue is the default one (a device-wide host wait), then

```cpp
#ifdef DPCT_PROFILING_ENABLED
  *event_ptr = queue->ext_oneapi_submit_barrier();
#else
  *event_ptr = queue->single_task([=]() {});
#endif
```

So removing the define alone turns every `cudaEventRecord` into an empty kernel node, which is worse than today; the two halves of D1 go together. Every engine queue is made by `create_queue(true)` (in-order, `device.hpp:616-627`) and so carries `enable_profiling`.

**Change.**
1. Measure first, on the card: build and run `sycl/probe/launch_cost.cpp` (`icpx -fsycl sycl/probe/launch_cost.cpp -o launch_cost && ./launch_cost 20000`). It prints host submit us/call and until-done us/call for a tiny kernel, a 4096-work-group kernel, a kept event and `submit_barrier + kernel`, with queue profiling off and on. If "on" is not measurably slower than "off" in the "until done" column, stop here and record that; the rest of D1 is then only the `sync_barrier` cleanup.
2. Add a CMake option `STRATA_SYCL_QUEUE_PROFILING` (default OFF) in `sycl/CMakeLists.txt` next to `STRATA_SYCL_SPIN_MAX` (`:33-38`): when ON, `add_compile_definitions(DPCT_PROFILING_ENABLED)` so the old build stays one flag away. Two builds are the A/B (there is no runtime switch for a queue property).
3. New `fixups.py` entry "profiling define off": for every file in `all_sources()` whose name does not end in `_parity.cpp`, `_bench.cpp` or `_test.cpp`, and for `include/strata/kernels/q8_1_finite.hpp`, delete the line `#define DPCT_PROFILING_ENABLED\n`. Tests and benches keep it (they time with events: `sampler_parity.cpp:494-505`, `fused_gr_bench.cpp:174-176`, and they are their own binaries). The `native_mmvq.dp.cpp:2884-2913` bench code lives in an engine TU; it must be guarded (step 5) or moved.
4. New `fixups.py` entry "explicit barriers": in the same engine files, rewrite `dpct::sync_barrier(E, Q)` (two arguments) to `(*(E) = (Q)->ext_oneapi_submit_barrier(), 0)`. The expression form matters because most sites are wrapped in `DPCT_CHECK_ERROR(...)`, which evaluates its argument as an expression and yields 0 on no exception (`verify.cpp:2424, 3445, 3675`, `mtp.cpp:1018-1043, 1827-1843`). Regex: `dpct::sync_barrier\((\w+(?:\[[^\]]*\])?),\s*([\w>()\-\.\[\]]+)\)` with the replacement above; check the result compiles at every site listed under Where (some second arguments are `PV[1][0]->stream()` and `m.copy`, `generate.cpp:10507`, `prefill.cpp:2573`). The one-argument default-queue form (`generate.cpp:1541, 1550`, the PCIe probe) stays as is or becomes `(dpct::get_current_device().queues_wait_and_throw(), *(E) = dpct::get_in_order_queue().ext_oneapi_submit_barrier(), 0)`; it is a startup probe, not on the decode path.
5. Guard the timestamp consumers: in `session.cpp:467-484, 553-554, 662-693` and `generate.cpp:1555-1561`, wrap the `get_profiling_info` reads in `if (q.has_property<sycl::property::queue::enable_profiling>())` and otherwise report "n/a (built without queue profiling)" or fall back to host `steady_clock` differences around a `wait()`. For the stage profiler (`STRATA_VERIFY_PROFILE`) the cleaner route is a dedicated profiling queue created once in `session.cpp` with `sycl::queue(ctx, dev, {in_order(), enable_profiling()})` and used only when the profiler is on; the decode queues stay unprofiled. The `native_mmvq.dp.cpp:2884-2913` bench code gets the same `has_property` guard.
6. Rebuild, run `ctest --test-dir build-sycl-aot` (the parity tests are unchanged binaries), then the engine on the three standard prompts: outputs must be identical to the profiling build.
7. Document in `docs/INTEL.md` under the B70 section: the probe's numbers, the engine's A/B, and that `--gpu-stages` / `STRATA_VERIFY_PROFILE` need `-DSTRATA_SYCL_QUEUE_PROFILING=ON`.

Not verified, check on the card: whether the UR Level Zero adapter actually creates a timestamp event per command on a profiling in-order queue and skips it otherwise (count `zeEventPoolCreate` / `zeEventCreate` and `zeCommandListAppendLaunchKernel` with a non-null signal event with `unitrace -c` on both builds). The two adapters (`SYCL_UR_USE_LEVEL_ZERO_V2=0/1`, review section 2) may differ; run the probe under both.

**Must not change.** Bitwise the same tokens on the three standard prompts: D1 touches no kernel and no arithmetic. The cross-queue dependencies at `verify.cpp:1222-1223, 1284-1286, 1322/1420` and `mtp.cpp:1018-1020, 1042/1064` (record on one queue, barrier on the other) keep the same event. The `sh_fork` / `sh_join` ordering and the doorbell waits are unchanged. Parity tests keep their timing paths.

**Tests.** Existing: every `*_parity` in `sycl/CMakeLists.txt:227-258` (they run on the card and compare against CPU references; unchanged by D1 because they keep the define). The engine-level check is output identity on the three standard prompts plus a run with `STRATA_VERIFY_PROFILE=1 --gpu-stages` on the profiling build to show the guard still prints stages, and on the default build to show it degrades to "n/a" instead of throwing. No new parity test (nothing numeric changes).

**Bench.** (1) `launch_cost 20000`: the four rows, profiling off vs on, "until done" column; this is the per-node cost the review wants measured. (2) Engine A/B of the two builds, same prompts, 256 greedy tokens, 5 interleaved pairs, medians: tok/s on the 20-token and 2,184-token prompts; `STRATA_DECODE_TIMING=1` ms/window and its verify / draft split; `STRATA_VERIFY_NODES=1` node count (should be unchanged: D1 removes no nodes, only their cost). (3) One `unitrace -d` per build over 256 tokens: the gap between consecutive kernels inside a window (the launch gap INTEL.md:297 puts at ~5 us).

**Expected gain, risk, effort.** Unknown until the probe runs; the bound is the whole node gap (5 to 13 ms of a 40 ms round if every node pays a timestamp event; zero if the adapter already elides events on in-order queues). Risk low: a build-level change with the old build one CMake flag away; failure modes are a `sync_barrier` site the regex misses (compile error, not a silent one) and a `get_profiling_info` on an unprofiled queue (a thrown `sycl::exception` at run time; step 5 covers the three consumers). 0.5 person-day plus the measurement window.

**Depends on / conflicts with.** Nothing; do it first because it changes the baseline every other item is measured against (review section 7, window 1 then "D1 and D7 first"). The author's PR #1111 branch keeps the define (it was re-migrated with dpct), so a later 3-way merge of his files re-introduces the line in files he changed; the `fixups.py` entry is idempotent and removes it again after `normalize.sh`.

---

### D2. Port QFUSE: producer-written q8_1 activation images in the hyper-connection read (and finish the GDN output norm half)

**Goal and measured motivation.** Upstream S26 (`STRATA_QFUSE=1`) has the kernel that produces an activation also write its q8_1 image, so the separate `native_quantize_q8_1` / `quantize_q8_1_rows` launch disappears: upstream states "the same bytes" (`src/core/verify.cpp:104-108`). The port already reads the switch (`sycl/src/core/verify.cpp:71-72`) and has the GDN half wired (`verify_kernels.dp.cpp:295-321, 481, 768-772`), but its hyper-connection read never writes the image: `fused_gr_read_multi` returns `false` on both exits (`fused_gr.dp.cpp:1755, 1882`), so the caller quantizes anyway. Nodes saved per window when the HC half lands: two quantize launches per layer per group (the attention input at `verify.cpp:948` for GDN layers and `:1021` for QSA layers, and the MoE input at `verify.cpp:1324-1326`), 96 for 48 layers with one group, 192 with two; plus the 39 GDN output quantizes the existing half already removes. The review sizes it at 100 to 150 nodes and 1 to 1.5 ms per round (about 3%), low risk because upstream is bitwise.

**Where.**
- `sycl/src/core/verify.cpp:71-72` (`g_qfuse`), `:661-676` (`qcnt_` allocation, zeroed once), `:747` (`self_commit` excludes QFUSE), `:905-924` (`gr_read_group` lambda; `:917-921` sets `a.q8_cnt` and `a.q8_mixed` for half 0 into `xq_` and for half 1 into `nat_xq_`), `:926` (`q8_attn`), `:948` and `:1021` (`if (!q8_attn) native_quantize_q8_1(xm, xq_, N, n, cs)` in the GDN and QSA branches), `:984-988` (`gdn_step_norm_multi(..., g_qfuse() ? (void*) xq_ : nullptr)` and `if (!g_qfuse()) native_quantize_q8_1(y_ ...)`), `:973-983` (the `batch_rec_` per-slot recurrences pass no `xq`), `:1209` (`q8_ffn`), `:1324-1326` (`if (!qdedup) if (!q8_ffn) quantize_q8_1_rows(...)`), `:1306` (`qdedup`).
- `sycl/src/kernels/cuda/fused_gr.dp.cpp:1607-1620` (`fused_gr_read_multi` entry and argument check), `:1755` and `:1882` (the two `return false`), `:601-602` (`UPM_COLS = 16`, `UPM_BLOCKS = 160`), `:612-702` (`gr_up_multi_kernel`; the `mixed` store at `:692-697`), `:1864-1875` (its launch on the default sliced path), `:1569` (its launch in the Q8_0 read path), `:1723, 1737` (the v3 `gr_up_v3_kernel` launches), `:1934` (`gr_up_kernel`, single token).
- `sycl/src/kernels/cuda/verify_kernels.dp.cpp:295-321` (`GdnQ81`, `gdn_q8_1_store`), `:330-345` (`gdn_step_norm_multi_kernel<ALL_OUT, Q>`), `:481`, `:768-772` (`if constexpr (Q) gdn_q8_1_store(...)`), `:1824-1830` (launcher takes `xq_out`).
- `sycl/include/strata/kernels/q8_1_finite.hpp` (the port's copy of the #606 helpers `q8_1_finite`, `q8_1_quant`, `q8_1_ds`).
- Upstream to mirror: `src/kernels/cuda/fused_gr.cu:177-212` (`GrQ81`, `gr_q8_tail`), `:373, 1088, 1376, 1602` (its call at the end of each up kernel), `:1660-1671` (the launcher's argument check and the "every token has q8_mixed, token 0 has counters" rule); `src/kernels/cuda/verify_kernels.cu:188-203` (the GDN store with the #606 clamp); `src/core/verify.cpp:1035-1037` (`if (!g_qfuse() || batch_rec_) native_quantize_q8_1(...)`); `include/strata/kernels/fused_gr.hpp:44-47, 58` (`q8_mixed`, `q8_cnt`, the return contract).

**Current behaviour.** `fused_gr.dp.cpp:1876-1882` (end of the default path):

```cpp
        st->parallel_for<dpct_kernel_name<class gr_up_multi_kernel_1a727f>>(
            ... [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gr_up_multi_kernel(m);
            });
    }
    const dpct::err0 e = 0;
    return false;   // SYCL port: the q8_1 images (S26 STRATA_QFUSE) and the Q8_0 reads (S23 STRATA_HC_Q8) are not built; the caller quantizes
```

`verify.cpp:917-921` already prepares the arguments:

```cpp
                if (qcnt_ != nullptr) {   // S26 STRATA_QFUSE: the consumer's q8_1 image written by the read itself
                    a.q8_cnt = qcnt_;
                    if (half == 0) a.q8_mixed = xq_ + (size_t) (t - tb) * (N / 32) * 36;
                    else if (strata::kernels::cpu::expert_layout().native) a.q8_mixed = nat_xq_ + (size_t) t * (N / 32) * 36;
                }
```

Upstream's tail (`fused_gr.cu:186-212`) runs at the end of each up block: fence, one atomic add on the 32-column group's counter, the second of the two 16-column blocks that own the group (`UPM_COLS` 16, so two blocks per q8_1 block) resets the counter and has warp k quantize token k's 32 columns with `native_quantize_q8_1_kernel`'s arithmetic (XOR-tree max and sum, `d = amax / 127`, `roundf(x / d)`, `ds = (d, sum)`), through `q8_1_finite`, `q8_1_quant`, `q8_1_ds`.

The port's GDN store (`verify_kernels.dp.cpp:315-320`) is the pre-#606 form:

```cpp
    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : sycl::round(xi / d);
    xq[idx / 32].qs[idx % 32] = q;
    if (idx % 32 == 0) xq[idx / 32].ds = sycl::half2(d, sum);
```

where upstream `verify_kernels.cu:203` writes `q8_1_ds(d, sum)` ("#606: clamped (the QFUSE path bypassed the finite helper)"). And the port's `verify.cpp:988` lacks upstream's `|| batch_rec_` (upstream `verify.cpp:1035-1037`): with `STRATA_QFUSE=1` and batched slots the per-slot recurrences at `:973-983` write no image and the quantize is skipped, so the out-projection reads a stale `xq_`.

**Change.**
1. Hand edits in `fused_gr.dp.cpp` (not a fixup; this is new device code). Add after `GrMulti` (`:244-251`) the port's `GrQ81 { sycl::half2 ds; int8_t qs[32]; }` and a `gr_q8_tail(const GrMulti& m, int d0, sycl::nd_item<3> item)` device function mirroring `fused_gr.cu:186-212`:
   - `sycl::atomic_fence(memory_order::release, memory_scope::device)` then `item.barrier(local_space)`; thread 0: `sycl::atomic_ref<unsigned, memory_order::acq_rel, memory_scope::device, access::address_space::global_space> c(m.a[0].q8_cnt[d0 / 32]); last = c.fetch_add(1u) == 1u; if (last) c.store(0u);` into a `group_local_memory_for_overwrite<int>` word; barrier; `if (!last) return;` `sycl::atomic_fence(memory_order::acquire, memory_scope::device)`.
   - warp = token (`warp >= m.T` returns), lane = column within the 32-column group `c0 = (d0 / 32) * 32`: read `m.a[warp].mixed[c0 + lane]` through a `sycl::atomic_ref<float, relaxed, device>` load (the other block wrote it; upstream uses a volatile read; on Xe a plain load after the acquire fence should also be fine, but the atomic load is the form the port already trusts for cross-work-group data).
   - amax and sum by XOR shuffles (`dpct::experimental::permute_sub_group_by_xor` as `gdn_q8_1_store` does, `:303-313`), `d = q8_1_finite(amax / 127.0f)`, `q = q8_1_quant(xi, d, amax)`, `ds = q8_1_ds(d, sum)` from `sycl/include/strata/kernels/q8_1_finite.hpp`, with `#pragma clang fp contract(off)` as at `:300` (the GDN store needed it to match the separate quantizer's rounding of `sum`).
   - Call it at the end of `gr_up_multi_kernel` (`:697`, after the `mixed` store loop) guarded by `if (m.a[0].q8_mixed != nullptr)`, and the same in both `gr_up_v3_kernel` variants (`:1723, 1737` launch them) so `STRATA_GR_V3=1` keeps working; `gr_up_kernel` (single token, `:1934`) is reached only through `fused_gr_read`, which upstream also leaves unfused.
2. In `fused_gr_read_multi` (`:1607-1620`): add upstream's checks (`:1660-1671`): the q8 pointers equal across tokens; `bool q8 = a[0].q8_cnt != nullptr; for t: q8 = q8 && a[t].q8_mixed != nullptr; if (!q8) clear every m.a[t].q8_mixed`. Return `q8` at both exits (`:1755` v3, `:1882` default) instead of `false`.
3. Counter ownership: `qcnt_` is `n_embd / 32` unsigned words zeroed once at `verify.cpp:661-676`; every launch leaves them zero (the last block resets). Both halves of a layer use the same counters serially on one queue, and `mtp.cpp` has its own verifier state, so no sharing problem; state this in a comment.
4. Bring the GDN store to upstream's #606 form: in `gdn_q8_1_store` (`verify_kernels.dp.cpp:315-320`) use `q8_1_finite` / `q8_1_quant` / `q8_1_ds` (include `strata/kernels/q8_1_finite.hpp`). Compare the port's `native_quantize_q8_1_kernel` (the separate launch; find it in `sycl/src/kernels/cuda/native_mmvq.dp.cpp` or `quantize_act.dp.cpp` by `grep -n q8_1_finite`) and keep the two bit-identical; `quantize_act_parity` pins the separate one.
5. Fix `verify.cpp:988` to `if (!g_qfuse() || batch_rec_) native_quantize_q8_1(y_ + ...)` as upstream `:1035-1037`. Hand edit (verify.cpp is a hand-ported file, INTEL.md "Copies dpct never produced").
6. Switch: the existing `STRATA_QFUSE=1`, default off (`verify.cpp:72`). No new name. Note the interaction at `:747`: QFUSE disables the one-token self-commit, as upstream.
7. Record in `docs/INTEL.md`: QFUSE is now whole on SYCL, the node count, the A/B.

**Must not change.** Bitwise the same tokens on the three standard prompts with `STRATA_QFUSE=1` and with it unset ("the same bytes" is the upstream contract: the fused tail reproduces `native_quantize_q8_1_kernel`'s operation order). The default path (`STRATA_QFUSE` unset) is untouched. `STRATA_VERIFY_QDEDUP` (`verify.cpp:1306-1326`) keeps its meaning: with QFUSE the FFN image comes from the read and `qdedup`'s quantize is skipped by `!q8_ffn`.

**Tests.** Existing: `gr_multi_parity` (`sycl/src/kernels/gr_multi_parity.cpp`: the multi-token read against the single-token one, memcmp, T = 1..8, with and without inject, in place and out of place; it does not set `q8_mixed` today), `quantize_act_parity` (the quantizer, byte-exact), `gdn_parity`, `gr_parity`, `verify_parity`. New: (a) extend `gr_multi_parity` with a q8 case: set `q8_cnt` (zeroed `n_embd/32` words) and every token's `q8_mixed`; after the read run `native_quantize_q8_1` on the same `mixed` into a second buffer and memcmp the two q8_1 images (36 bytes per 32 columns) for T = 1..8, both halves (`apply` true and false, with and without `w_inject`), and launch three times on the same counters (they must read back as zero after each launch); also a case where one token's `q8_mixed` is null (the read must write none and return false). (b) A GDN case in `gdn_parity` or `verify_parity` comparing `gdn_step_norm_multi` with `xq_out` set against `native_quantize_q8_1` of its `y`, including a row whose 32-block sum exceeds 65504 (the #606 clamp). (c) Engine: `STRATA_VERIFY_DEBUG=1` residual ladder identical with and without QFUSE on the 20-token prompt.

**Bench.** `STRATA_VERIFY_NODES=1` with and without `STRATA_QFUSE=1`: expect 96 (one group) to 192 (two groups) fewer kernel nodes plus 39 from the GDN half. `STRATA_DECODE_TIMING=1` verify ms/window, 5 interleaved pairs on the 20-token and 2,184-token prompts, 256 greedy tokens, medians; `gr_bench` (`sycl/src/kernels/gr_bench.cpp`, prints us per `fused_gr_read_multi` call for 1 to 6 tokens and a checksum) to show the tail's cost on the read itself (expect a few us at most on 160 blocks). unitrace: the `native_quantize_q8_1` / `quantize_q8_1_rows` kernels vanish from the per-kernel table; `gr_up_multi_kernel` grows by less than their sum.

**Expected gain, risk, effort.** 135 to 231 fewer nodes per window: at 5 us a node 0.7 to 1.2 ms, plus the quantize kernels' own device time (small, a few us each): 1 to 1.5 ms of a 40 ms round, about 3% (the review's figure). Risk low: bitwise upstream; the failure modes are the cross-work-group visibility of `mixed` before the tail reads it (caught by the new parity case: a wrong byte in the image) and a counter left non-zero (caught by the repeated-launch case). 2 to 3 person-days including the GDN #606 alignment.

**Depends on / conflicts with.** D1 first (the node cost it saves is what D1 measures). Independent of D3, D5, D6 in code, but D6's down-reduce fold touches the same `fused_gr.dp.cpp` launch sequence (`:1757-1875`): land D2 before D6 or rebase. Upstream PR #1111 (the author's branch) does not have QFUSE on SYCL either (its `fused_gr.dp.cpp` is a re-migration); this item is a candidate upstream PR to `Niko1221/Strata` in its own right.

---

### D3. Router GEMV + top-10 + `resident_plan` in one kernel per routed group

**Goal and measured motivation.** Per layer and group the window's routing is three launches on the compute queue: the BF16 router GEMV (`bf16_gemv_fp32_mmvf_multi`, 512 work-groups), the top-10 (`native_router_top10_multi_ne`, one work-group per token) and the device plan (`resident_plan`, one 128-thread work-group), `verify.cpp:1239-1266`. The last two are single-work-group latency kernels whose only cost is the launch gap and their own serial time. Fusing them into the GEMV's tail removes two nodes per layer per group: 96 (one group) to 192 (two groups) of the 2,400 to 2,600 nodes, 0.5 to 1 ms per round at 5 us a node (the review's estimate). The arithmetic is unchanged, so the result is bitwise by construction; the risk is in the fused kernel's single-work-group tail and in the LFUSE interplay.

**Where.**
- `sycl/src/core/verify.cpp:1227-1248` (the router block: `w_router`, the LFUSE `w_sgi`, the `dec_batch && n > 1 && native_router_enabled() && (NE == 512 || NE == 256) && K == 10` condition at `:1233`, the GEMV at `:1239-1240`, the top-10 at `:1241`), `:1258-1266` (`resident_plan` in the `ar_on()` and `device_plan_` branches), `:760-763` (`lfuse_on`).
- `sycl/src/kernels/cuda/native_bf16.dp.cpp:104-130` (`bf16_f32_mmvf_multi_kernel<BLOCK_SIZE, NT, EXACT_T>`: one work-group per output row, `partials[NT][32]` in local memory, pairs of BF16 per thread), `:521-533` (`mmvf_block_size`: 256 for `n_in` 2560), `:581-640` (`bf16_gemv_fp32_mmvf_multi`: grid `range(1,1,n_out) * range(1,1,N)`, so 512 work-groups of 256 for the router), `:536-560` (`bf16_gemv_fp32_mmvf_multi_aux`, the LFUSE gate row; only with `STRATA_MMVF_ROWS=1`).
- `sycl/src/kernels/cuda/native_router.dp.cpp:60-140` (`route<NE>`: one work-group of (8, 32) per token, only sub-group row 0 active, each lane holds `NE/32` logits, softmax, 10 argmax rounds with lowest-index ties, renormalised weights), `:269-283` (`native_router_top10_multi`: grid `n_tok` work-groups of (8, 32); the comment at `:273-275` says upstream's one-warp-per-token `route_multi` gives the same bits but hung an A750 on first launch), `:302-310` (`_ne`: 256 experts).
- `sycl/src/kernels/cuda/verify_kernels.dp.cpp:2066-2068` (`kResidentPlanMax = 128`, one thread per entry, `kVerifyMaxT * 10 <= 128`), `:2075-2210` (`resident_plan_kernel`: local `s_ids`, `s_excl`, `s_wsum[4]`, `s_bad`; the mirror address `maddr` for a host-mirrored expert; the `plan_err` system-scope store on a missing expert), `:2361-2384` (launcher: `g_mirror_table` slice, one work-group of 128), `sycl/include/strata/kernels/resident_plan_mirror.hpp` (the port's mirror hook).
- `sycl/src/kernels/cuda/verify_kernels.dp.cpp:1355-1398` (`argmax_rows_kernel`: the "last work-group to arrive finishes the job" pattern with an atomic counter and `atomic_fence(acq_rel, device)`, the pattern the fused kernel reuses).
- Tests: `sycl/src/kernels/router_top10_parity.cpp`, `native_multi_parity.cpp` (route_multi vs route token by token), `resident_plan_parity.cpp`, `sycl/CMakeLists.txt:227-258`.

**Current behaviour.** `verify.cpp:1233-1266`:

```cpp
        if (dec_batch && n > 1 && w_router != nullptr && native_router_enabled() && (NE == 512 || NE == 256) && K == 10) {
            try {   // SYCL port: the fused multi-token router for 256 experts as well (it was 512-only)
                if (w_sgi != nullptr)
                    sg_ready = bf16_gemv_fp32_mmvf_multi_aux(mixed_ + tb * N, N, (const uint16_t*) w_router->data,
                                                             logits_ + tb * NE, NE, N, NE, n,
                                                             (const uint16_t*) w_sgi->data, sh_g_ + tb, 1, cs);
                if (!sg_ready)
                bf16_gemv_fp32_mmvf_multi(mixed_ + tb * N, N, (const uint16_t*) w_router->data, logits_ + tb * NE, NE, N,
                                          NE, n, cs);
                native_router_top10_multi_ne(logits_ + tb * NE, ids_ + tb * K, w_ + tb * K, n, (int) NE, cs);
            } ...
        if (ar_on()) {
            resident_plan(ids_ + tb * K, n * (int) K, (int) K, hits_.d_res + l * g.n_expert, (int) g.n_expert,
                          hits_.cache_base, slot_off_d_, (long long) hits_.blob,
                          plan_ + (size_t) grp * (size_t) (plan_i32_ + 16), (long long) max_t_ * K, nullptr, 0, cs, m_plan_err_);
        } else {
            if (device_plan_)   // E-6: every routed expert resident: this group's plan without the host
                resident_plan(..., skip_ + grp, (uint32_t) ((l - lb_) * G + grp + 1), cs);
```

Note that between the top-10 and the plan the non-all-resident path also runs a `doorbell_publish*` kernel (`:1269-1283`) that reads `ids_`; the fused kernel must leave `ids_` and `w_` in memory exactly as today for it.

**Change.**
1. Reject the literal "one work-group per row": the router weight is 512 x 2560 BF16 = 2.6 MB; one Xe core pulls perhaps 30 to 60 GB/s, so a single work-group would take 45 to 90 us per layer against the 10 us of launch gap saved. Keep the GEMV's 512-work-group grid and add a tail.
2. New port-only translation unit `sycl/src/kernels/cuda/route_plan_fused.dp.cpp` and header `sycl/include/strata/kernels/route_plan_fused.hpp` (modelled on `resident_plan_mirror.hpp`), listed in `sycl/CMakeLists.txt` with the other kernel sources. API:

   ```cpp
   /// GEMV (bitwise bf16_gemv_fp32_mmvf_multi) + top-10 (bitwise route<NE>) + resident_plan (bitwise) in one launch.
   /// n_tok 1..8, NE 512 or 256, K 10. skip/ring/plan_err as resident_plan. Returns false when the shapes are not
   /// covered (the caller then takes the three launches).
   bool route_plan_fused(const float* x, int64_t ldx, const uint16_t* w_router, float* logits, int NE, int64_t n_in,
                         int n_tok, int32_t* ids, float* weights, const int32_t* res_layer, int n_expert,
                         const uint8_t* cache_base, const unsigned long long* slot_off, long long blob, int32_t* plan,
                         long long capx, uint32_t* skip, uint32_t ring, uint32_t* plan_err, unsigned* counter, void* stream);
   ```

3. Device code reuse without copies: lift the three device functions into `.dp.hpp` headers included by their current TU and by the new one (the `s26_tsum.dp.hpp` precedent): `bf16_f32_mmvf_multi_kernel` body from `native_bf16.dp.cpp:104-130` into `native_bf16_dev.dp.hpp`; `route<NE>` from `native_router.dp.cpp:60-140` into `native_router_dev.dp.hpp`; `resident_plan_kernel` from `verify_kernels.dp.cpp:2075-2210` into `resident_plan_dev.dp.hpp` (it needs `kResidentPlanMax`, the `mir` table pointer which the launcher resolves at `:2364-2366`, and the `sys_store` from `sycl_doorbell.hpp`). These are hand edits of migrated files; they survive re-migration through the 3-way merge.
4. The fused kernel: grid `range(1,1,NE) * range(1,1,256)` (256 = `mmvf_block_size(2560)`; assert it at run time and return false otherwise). Each work-group computes its expert's logit for the `n_tok` rows exactly as `bf16_f32_mmvf_multi_kernel<256, NT, true>` (same `partials`, same pair order, same reductions) and writes `logits[t * NE + row]`. Then: `atomic_fence(release, device)`, barrier, thread 0 `fetch_add` on `counter` (a device word owned by the verifier, one per group, zeroed at allocation; the last block resets it to 0 as `argmax_rows_kernel` does at `:1496`), the last-arriving block continues after `atomic_fence(acquire, device)`:
   - top-10: warp t (t < n_tok, up to 8 warps in the 256-thread block) runs `route<NE>`'s body on `logits + t * NE` into `ids + t * 10`, `weights + t * 10`. This is upstream's `route_multi` shape (one warp per token). The port avoids `route_multi` because it hung an A750 (`native_router.dp.cpp:273-275`); on the B70 `native_multi_parity` exercises it (not verified that it passes there; run it first). If it hangs on the B70 too, fall back to the sequential form: warp 0 routes the tokens one after another (8 x a few us).
   - barrier, `atomic_fence(acq_rel, device)` so the block's own `ids` writes are visible to all its threads; then threads 0..127 run `resident_plan_kernel`'s body on `ids` (n_entries = n_tok * K <= 80 <= 128) with the group's `skip`, `ring`, `mir`, `plan_err`, writing `plan` exactly as today (`counts`, `start`, `dst`, `tok`, `ptr`, `start2`). The body's barriers become block barriers executed by all 256 threads (threads 128..255 idle but present), so no divergent barrier.
5. Host side, `verify.cpp:1233-1266`: a new `static const bool fused = env "STRATA_ROUTE_FUSED" == "1"` (default off). When `fused && w_sgi == nullptr && (ar_on() || device_plan_)`, call `route_plan_fused(...)` in place of the GEMV + top-10 + `resident_plan` (keeping the `doorbell_publish*` call at `:1269-1283` where it is, after the fused launch, in the non-`ar_on()` branch); when it returns false, take the existing three launches. The LFUSE gate path (`w_sgi != nullptr`, `bf16_gemv_fp32_mmvf_multi_aux`) stays unfused in v1: `STRATA_LFUSE=1` with `STRATA_LFUSE_GATE=1` forces the old path for that layer; say so in the log line once. A v2 can carry the aux row (the `launch_rows` kernels at `native_bf16.dp.cpp:546-560` are a different kernel family; do not mix the two in v1).
6. The `counter` word: one `unsigned` per group (`groups_[T]` <= 2) in the verifier's arena (`Bump::take` at `verify.cpp:537` region), zeroed once with the arena; `qcnt_`'s allocation at `:661-676` is the model.
7. No upstream mirror: upstream `src/core/verify.cpp:1287-1304` has the same three launches; this is a port-first kernel and a candidate upstream PR afterwards (CUDA has the same last-block pattern available).

**Must not change.** Bitwise the same `logits_`, `ids_`, `w_`, plan words and `skip` as the three launches, hence bitwise tokens on the three standard prompts. `resident_plan`'s contract (distinct experts in routing order, entries ascending, the mirror address for a non-resident expert, the empty plan and `plan_err` on a missing one, `skip = 0` vs `ring`) is unchanged; `resident_plan_parity` is the oracle. The `doorbell_publish*` kernels keep reading `ids_` and `w_` from memory after the fused launch.

**Tests.** Existing: `router_top10_parity` (spec reference), `native_multi_parity` (`route_multi` against `route` token by token, n = 1..19, ties, NaN), `resident_plan_parity` (host replay of the plan contract, n = 1..80, with and without `slot_off`, with and without `skip`, the `plan_err` path), `bf16_gemv_parity`, `verify_parity`. New: `route_plan_parity` (add to the 0.1.40 foreach at `sycl/CMakeLists.txt:252-258`): random BF16 router weights (2560 x 512 and 2560 x 256), random `x` rows for n = 1..8, random residency tables with some non-resident ids and a mirror table; run the three launches and the fused kernel on separate buffers and memcmp `logits`, `ids`, `weights`, every plan word the host pool reads (`counts`, `start[0..groups]`, `dst`, `tok`, `ptr`, `start2[0]`), `skip` and `plan_err`; repeat the fused launch three times on the same counter (it must be zero after each). Also run it with `STRATA_SPIN_MAX` untouched (the fused kernel spins on nothing).

**Bench.** `STRATA_VERIFY_NODES=1` with and without `STRATA_ROUTE_FUSED=1`: 96 or 192 fewer kernel nodes. `STRATA_DECODE_TIMING=1` verify ms/window and tok/s, 5 interleaved pairs, both prompts, 256 greedy tokens, medians. unitrace: the fused kernel's device time against the sum of `bf16_f32_mmvf_multi_kernel` + `route_512_multi` + `resident_plan_kernel` per layer (the tail adds the serial top-10 and plan time, which today runs in two kernels anyway; the gain is the gaps). Also run `--spec 6` (windows of 8, two groups) to see the two-group case.

**Expected gain, risk, effort.** 0.5 to 1 ms per round (2 nodes x 48 layers x 1 or 2 groups x ~5 us) less the tail's serial time, which already exists today as two single-work-group kernels: net about 0.4 to 0.9 ms, 1 to 2% of decode. Risk medium: a device-wide "last block" dependency inside one kernel (the pattern exists in `argmax_rows_kernel` and in upstream's `gr_q8_tail`, but a counter left non-zero or a visibility gap gives a wrong plan, which the all-resident graph would run silently: `plan_err` only covers a missing expert); the A750 `route_multi` hang history; LFUSE_GATE excluded in v1. 2 to 3 person-days.

**Depends on / conflicts with.** D1 first. Conflicts in `verify.cpp:1227-1266` with the Tier 0 LFUSE A/B only in the sense that `STRATA_LFUSE_GATE=1` disables the fusion per layer; measure D3 with LFUSE off, then LFUSE on without GATE. Shares the `verify.cpp` router block with the review's D7 (per-layer mirror-hit counters, `verify_kernels.dp.cpp:2099-2104`): if D7 lands first, its counter stores move into the lifted `resident_plan_kernel` body and the fused tail inherits them. No relation to PR #1111.

---

### D5. Drafter: launch all step graphs at once, with a device-side min-p cut, removing the host round trips

**Goal and measured motivation.** With `--spec-min-p 0.5` (the setup default, INTEL.md:778) the drafter runs its chain one graph at a time: the round graph, then for each step j the host waits for draft j's probability to land in mapped memory, compares it with `min_p`, and only then submits step j+1's graph (`mtp.cpp:1690-1715`). That is up to `max_steps - 1` host round trips per decode round (submit, GPU idle while the host reacts, poll), 2 to 3 per round at `--spec 4`, which the review sizes at 0.2 to 0.5 ms a round. The `min_p <= 0` path already submits every graph back to back and polls the outputs as they land (`mtp.cpp:1582-1649`). The host's window rule applies `min_p` to the probabilities again anyway (`generate.cpp:10997`, `:12199`), so the tokens are the same either way; the only cost of submitting everything is the GPU time of the steps past the cut, which a device-side cut can only partly remove. Draft time per window is not in INTEL.md: `STRATA_DECODE_TIMING=1` prints it (`draft %.2f` ms/window).

**Where.**
- `sycl/src/core/mtp.cpp:1533-1735` (`MtpDrafter::draft`), `:1542-1545` (graphs captured lazily: `capture_round`, `capture_step`), `:1582-1649` (the `min_p <= 0.0f` path: round + every step submitted, then `h_out_[j]` polled in order with `prefetch_ple` between, final `cs_->wait()`), `:1650-1730` (the `min_p > 0` path: `wait_step` lambda `:1668-1684`, the loop `:1690-1715` submitting step j only after `pj >= min_p`), `:1732-1735` (`ms_draft`), `:1227-1271` (`capture_round`: ends with `mtp_select` `:1268` and `force_token` `:1269`), `:1280-1303` (`capture_step`), `:1760-1870` (`prepare_chain` / `chain_launch`: the pipelined path that already submits round + steps at once with events `ev_step_` and `ev_chain_`), `:1880-1945` (`chain_outputs_ready`, `chain_poll`), `:431-432` (`h_out_`, `h_prob_` mapped), `:92` (the mapped outputs are polled host memory).
- `sycl/src/kernels/cuda/verify_kernels.dp.cpp:956-984` (`mtp_select_kernel`: copies row `*row_dev` of R to row 0, writes `out_p[j] = probs[row]`, a system-scope fence, `out[j] = tok`), `:1132-1148` (its launcher: one work-group of 256), `:999-1003` (`force_token_kernel`: reads a mapped word `((const volatile int32_t*) force)[j]` inside a captured graph, the pattern for a per-request value read at replay time), `:1150-1163`.
- `sycl/src/program/generate.cpp:11139-11145` (the call `mtp.draft(T, outv.data(), p, a, drafts.data(), err, dprob.data(), (float) req_spec_min_p)` and the draft timer), `:9255, 9281` (`req_spec_min_p` is per request: `spec_min_p` in the request), `:10997` and `:12199` (the window rule: `while (T < S_mtp && dprob[T - 1] >= min_p) ++T`), `:11023` (`p_mtp` from the probabilities).
- Upstream: `src/core/mtp.cpp:1294-1360` has the same two host paths; nothing to mirror.

**Current behaviour.** `mtp.cpp:1686-1715` (the `min_p > 0` loop):

```cpp
        wait_step(0);
        drafts[0] = ((volatile int32_t*) h_out_)[0];
        float pj = ((volatile float*) h_prob_)[0];
        if (probs) probs[0] = pj;
        n = 1;
        for (int j = 1; j < max_steps && pj >= min_p; ++j) {
            if (DPCT_CHECK_ERROR((cs_)->ext_oneapi_graph(
                    *(cp ? step_exec_c_[j] : step_exec_[j]))) != 0) {
                ...
            }
            (void)DPCT_CHECK_ERROR(((cs_)->ext_oneapi_empty()));
            prefetch_ple(drafts[j - 1]);
            wait_step(j);
            drafts[j] = ((volatile int32_t*) h_out_)[j];
            pj = ((volatile float*) h_prob_)[j];
```

and `mtp_select_kernel`'s tail (`verify_kernels.dp.cpp:971-983`):

```cpp
    if (item_ct1.get_group(2) == 0 && item_ct1.get_local_id(2) == 0) {
        const int32_t tok = ids[row];
        *tok_dst = tok;
        if (probs != nullptr && out_p != nullptr) ((volatile float*) out_p)[j] = probs[row];
        sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
        if (out != nullptr) ((volatile int32_t*) out)[j] = tok;
    }
```

The `min_p <= 0` path (`:1582-1649`) is the shape wanted: all graphs submitted at once, outputs polled as they land, PLE rows prefetched per draft.

**Change.**
1. Switch `STRATA_SPEC_DRAFT_ALL=1` (default off). When set and `min_p > 0`, `draft()` takes the all-at-once shape: submit the round graph and every step graph `1..max_steps-1` (as `:1582-1612`), then poll `h_out_[j]` in order (as `:1619-1630`), computing `pj` from `h_prob_[j]` and stopping the *host's* use of outputs at the first `pj < min_p` (n = j + 1) exactly as the per-step loop does, so `drafts[j >= n] = 0`, `probs[j >= n] = 0` and the caller's window rule sees the same values as today. Final `cs_->wait()` as today (`:1716`).
2. Device-side cut (what makes the later steps cheap, within what a captured graph allows): add to `MtpDrafter` a mapped word pair `h_minp_` / `m_minp_` (allocated with the others at `:426-432` through the same `mapped(...)` helper, so it comes from `host_malloc_polled` and is readable from a kernel) and a device word `cut_` (arena). Extend `mtp_select` with `const float* minp_mapped, int32_t* cut` (new port-only overload declared in a header under `sycl/include/strata/kernels/`, the upstream declaration in `include/strata/kernels/verify_kernels.hpp` untouched). In the kernel, thread 0 of block 0: `if (j == 0) *cut = 0;` then `const float mp = minp_mapped ? ((const volatile float*) minp_mapped)[0] : 0.0f; const float p = probs[row]; const bool stop = (*cut != 0) || (p < mp);` and if `stop`: `*cut = 1; out_p[j] = 0.0f; fence; out[j] = -2;` else as today. The host writes `*h_minp_ = min_p` before each round (as it writes `h_row_`), so the graphs stay reusable across requests with different `spec_min_p`. The host's poll loop treats `-2` as "cut here" (n = j, no further reads; still `cs_->wait()` at the end).
3. What the cut does and does not save: the step graphs after the cut still run their forward (a captured graph cannot skip nodes; the SYCL graph extension vendored here has no device-side conditional node, not verified against the installed 2026.1.1 headers, check `sycl/ext/oneapi/experimental/graph/*.hpp` for anything conditional or dynamic before deciding). So the cut saves only the `R` copy and the output writes in `mtp_select` of later steps, and above all it keeps the host logic simple and identical to today's. The real trade is: host round trips removed (0.2 to 0.5 ms per round, E) against the GPU time of the steps past the cut (a draft-layer forward each; its size is what `draft %.2f` in `STRATA_DECODE_TIMING` will show before and after). On code (85% acceptance, INTEL.md:420) the cut rarely binds (min-p 0.3 to 0.7 was within noise, INTEL.md:294); on prose it binds more often. Measure both.
4. Optional v2 to recover the waste, only if the measurement says the extra steps cost more than the round trips saved: capture one graph per `(T, n_steps)` pair (round + n_steps steps, `n_steps = 1..max_steps-1`) and pick the graph by a cheap host-side predictor (e.g. last round's `n`), falling back to the per-step loop when the predictor is wrong. Not in v1.
5. `prefetch_ple` keeps its order (once per draft as it lands, `:1623-1629`). `ms_draft` accounting unchanged. The coupled sampler (`cp`, `:1541`) uses the same two paths and is covered by the same switch.
6. `prepare_chain` / `chain_launch` (`:1760-1870`) already submit everything at once for `--pipeline-windows`; reuse their event handling only as a reference, do not route the serial loop through them (they need `set_force_capture`).

**Must not change.** Bitwise the same tokens: each step's inputs are the previous step's `R` row and token exactly as before, the host applies the same `min_p` rule to the same probabilities, and the window rule at `generate.cpp:10997` sees the same `dprob` values (zeros after the cut as today). `--spec-min-p 0` behaviour is untouched (it already is the all-at-once path). `n_drafts` and `probs` outputs to the caller keep their meaning.

**Tests.** Existing: none of the parity tests covers `mtp.cpp` (the drafter is checked through the engine). `verify_parity` could host a `mtp_select` case: new test comparing the new overload against the old kernel for `minp_mapped = nullptr` (identical outputs) and for a `min_p` above the probability (out = -2, prob 0, cut = 1, and the next call with j > 0 also writes -2). Engine: the three standard prompts with `STRATA_SPEC_DRAFT_ALL=1` and without, identical tokens; a prose prompt (the 200-token story of INTEL.md:761) where the cut binds; `--spec-min-p 0` unchanged.

**Bench.** `STRATA_DECODE_TIMING=1`: the `draft %.2f` ms/window field and total ms/window, with and without `STRATA_SPEC_DRAFT_ALL=1`, 5 interleaved pairs, 256 greedy tokens, on the 20-token and 2,184-token code prompts and on the prose prompt; tok/s medians. Log the accepted drafts per round (the engine prints acceptance) to confirm the window rule is unchanged. unitrace: the draft-layer kernels' device time per round before and after (the waste is visible as extra draft forwards).

**Expected gain, risk, effort.** 0.2 to 0.5 ms per round on code (0.5 to 1% of decode), possibly negative on prose if the extra steps cost more than the round trips (then keep the switch off for prose or do v2). Risk medium: the host logic has two shapes already and a third predicate (`-2`) joins them; a wrong `n` would change the window (caught by output identity); the mapped `min_p` word is read inside a replayed graph, which the port says works for `force_token` on the B70 but visibility of host stores during a kernel "stays unreliable on this platform" (INTEL.md:256): the read happens in a kernel launched after the host wrote it, which is the safe case, but verify with `STRATA_VERIFY_DEBUG` that the cut fires where expected. 1 to 2 person-days.

**Depends on / conflicts with.** D1 first (its gaps are part of what the round trips cost). Independent of D2, D3, D6. The review's `--spec 6` Tier 0 A/B changes `max_steps` and so the number of round trips: run D5's bench at both `--spec 4` and `--spec 6`. No upstream mirror; a candidate upstream PR afterwards (CUDA has the same two paths).

---

### D6. Fold the HC down-reduce into the up kernel's launch; the three input copies as one `copy_from_mapped_multi`; multi-block argmax on SYCL

**Goal and measured motivation.** Three small node savings the review groups at about 100 nodes and 0.3 to 0.5 ms a round. (a) The default sliced hyper-connection read is three kernels per call, norm, sliced down, reduce, then up (`fused_gr.dp.cpp:1757-1875`), two calls per layer plus the head: the reduce is a 16 to 21 work-group kernel that only sums 80 slice partials in a fixed order; removing it saves one node per read, about 97 per window. (b) The window's inputs are three `copy_i32_from_mapped` launches of a few hundred words (`verify.cpp:772-774`); `copy_from_mapped_multi` (`elementwise.dp.cpp:812-838`) already does up to 8 copies in one launch and the port's `mtp.cpp:1251-1265` already uses it: two nodes per window. (c) The head's greedy pick is `sample_tokens`' one-block-per-row kernel (`sampler.dp.cpp:1489-1525`, 1,024 threads over the vocabulary) because `argmax_rows_wanted()` tests for CUDA major 8 (`verify_kernels.dp.cpp:1576-1591`); the multi-block `argmax_rows` kernel exists in the port and `verify_parity` pins it bitwise against `sample_tokens` (`verify_parity.cpp:60-63`). One block scanning about 1 MB per row (n_vocab floats) runs tens of us; `argmax_rows` spreads it over up to 61 blocks per row. It runs once per verify window (`verify.cpp:1549-1557`) and once per draft step (`mtp.cpp:1136-1141`).

**Where.**
- (a) `sycl/src/kernels/cuda/fused_gr.dp.cpp:525-531` (the sliced read's rationale and `GRS_COLS = 128, GRS_SLICES = 80, GRS_ROWS = 324`), `:532-570` (`gr_down_sliced_kernel`: work-group `sl` owns columns `[128 sl, +128)`, writes `m.part2[(sl * kFusedGrMaxT + k) * GRS_ROWS + r]`), `:571-581` (`gr_down_reduce_kernel`), `:582-585` (`gr_down_sliced()`, default on, `STRATA_GR_DOWN_SLICED=0` the direct kernel), `:586-596` (`slice_partials`: the per-queue `part2` buffer, 80 x 8 x 324 floats), `:1793-1807` (the sliced launch and, at `:1805-1807`, the reduce launch `nred = (n_tok * GRS_ROWS + 127) / 128` work-groups of 128), `:1864-1875` (the up launch), `:612-702` (`gr_up_multi_kernel`; its prologue at `:624-635` reads `m.part` for all `T * LR` rows: `x += m.part[(sp * kFusedGrMaxT + k) * LR + r]` over `m.nsplit` splits, then silu, into local `lo[k][r]` and `m.a[k].lo[r]`), `:244-251` (`GrMulti`: `part`, `nsplit`, `part2` are the port's fields), `sycl/src/kernels/gr_bench.cpp` (us per call and a checksum).
- (b) `sycl/src/core/verify.cpp:771-774`; `sycl/src/kernels/cuda/elementwise.dp.cpp:783-838` (`MappedCopies`, `copy_from_mapped_multi_kernel`: grid `(n copies, bx <= 32)` x 256, float4 path when 16-byte aligned and `words % 4 == 0`, else volatile int32), `:773-780` (`copy_i32_from_mapped_kernel`, one work-group of 128), `include/strata/kernels/elementwise.hpp:129-134` (`MappedCopy`, upstream header); the port's own use `sycl/src/core/mtp.cpp:1251-1265`; upstream `src/core/verify.cpp:756` still has the three separate copies (no upstream mirror for this spot).
- (c) `sycl/src/kernels/cuda/verify_kernels.dp.cpp:1261` (`kArgMaxBlocks = 128, kArgThreads = 256`), `:1355-1398` (`ArgRow`, `argmax_rows_kernel`), `:1501-1525` (`argmax_rows`: `nb = min(128, max(1, (n + 4095) / 4096))` blocks per row), `:1564-1574` (`multi_block_head_ops`: `STRATA_MULTI_BLOCK_ARGMAX`, default on), `:1576-1591` (`argmax_rows_wanted`: `major == 8 && multi_block_head_ops()`), `sycl/include/dpct/device.hpp:39-65` (`detail::get_version` parses `info::device::version`), `sycl/src/core/verify.cpp:537` (`arg_scratch_`), `:1549-1557` (the head's pick), `sycl/src/core/mtp.cpp:1136-1144` (the draft head's pick and `row_top_prob_split`, which already takes `multi_block_head_ops()`), `sycl/src/kernels/cuda/sampler.dp.cpp:1489-1525` (the greedy one-block kernel), `sycl/src/kernels/verify_parity.cpp:60-63, 144` (`test_argmax`).

**Current behaviour.** (a) `fused_gr.dp.cpp:571-581`:

```cpp
__dpct_inline__ void gr_down_reduce_kernel(GrMulti m) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = (int) item.get_global_id(2);
    if (i >= m.T * GRS_ROWS) return;
    const int k = i / GRS_ROWS, r = i - k * GRS_ROWS;
    if (r >= LR && m.a[0].w_inject == nullptr) return;
    float x = 0.0f;
    for (int sl = 0; sl < GRS_SLICES; ++sl) x += m.part2[((size_t) sl * kFusedGrMaxT + k) * GRS_ROWS + r];
    if (r < LR) m.part[(size_t) k * LR + r] = x;   // split 0
    else m.a[k].inject_out[r - LR] = x;
}
```

(b) `verify.cpp:772-774`:

```cpp
    copy_i32_from_mapped(tok_, m_tok_, T, cs);
    copy_i32_from_mapped(step_, m_step_, (int64_t) T * kStepCount, cs);
    copy_i32_from_mapped(pos_, m_pos_, (int64_t) MT * (NH + NKV + IQ), cs);
```

(c) `verify_kernels.dp.cpp:1576-1591`:

```cpp
bool argmax_rows_wanted() try {
    ...
    static const bool want = [] {
        int dev = 0, major = 0;
        if (DPCT_CHECK_ERROR(dev = dpct::get_current_device_id()) != 0 ||
            DPCT_CHECK_ERROR(major = dpct::get_device(dev).get_major_version()) != 0)
            return false;
        return major == 8;
    }();
    return want && multi_block_head_ops();
```

`get_major_version` parses the device's `info::device::version` string (`device.hpp:39-65`); on Level Zero that is not a CUDA compute capability, so `want` is false on the B70 (not verified: print it once) and the window takes `sample_tokens` with `greedy = true`.

**Change.**
(a) The fold, behind `STRATA_GR_DOWN_FOLD=1` (default off):
1. Do not fold the reduce into `gr_up_multi_kernel` as literally stated: each of the 160 up blocks needs every `lo[k][0..320)` for every token, so each would have to re-sum 80 slices x 324 rows x T itself, 160 x 26k x T loads of `part2` (about 100 MB of L2 traffic per call at T = 6 against the 6.5 MB of `w_down` the read exists to stream). Reordering the sums into the up dot (`sum_j w_up[r][j] * sum_sl part2[sl][j]`) is not bitwise.
2. Fold it instead into the tail of `gr_down_sliced_kernel`: add `unsigned* part2_cnt` to `GrMulti` (allocated beside `part2` in `slice_partials`, `:586-596`, zeroed once); at the end of the kernel, `atomic_fence(release, device)`, barrier, thread 0 `fetch_add` on the counter, the last of the 80 slice blocks (`== GRS_SLICES - 1`) resets it, `atomic_fence(acquire, device)`, then its 256 threads run the reduce body for `i in [tid, T * GRS_ROWS) step 256` with the same `sl = 0..79` sequential sum, writing `m.part` and `inject_out` exactly as `gr_down_reduce_kernel`. Remove the reduce launch at `:1805-1807` when the switch is on. The pattern is `argmax_rows_kernel`'s (`:1388-1398`).
3. Expected cost of the tail: the last block alone reads 80 x 324 x T floats (830 KB at T = 8) with a 10 KB stride between consecutive `sl` for one `(k, r)`: 200k scattered 4-byte loads from one Xe core, likely 10 to 30 us (E), against the roughly 5 us launch gap plus the reduce kernel's own few us it removes. The fold may well lose; `gr_bench` decides and the switch stays off if it does. A variant worth one try inside the same switch: write `part2` transposed as `[k][r][sl]` in the sliced kernel (a different store pattern, same values) so the tail's loads are 320-byte contiguous runs per `(k, r)`; the reduce order `sl = 0..79` is unchanged so the sums are bitwise.
(b) Hand edit `verify.cpp:772-774` into

```cpp
    const MappedCopy in[3] = {{tok_, m_tok_, T}, {step_, m_step_, (int64_t) T * kStepCount},
                              {pos_, m_pos_, (int64_t) MT * (NH + NKV + IQ)}};
    copy_from_mapped_multi(in, 3, cs);
```

(the `MappedCopy` struct and the function are in the upstream header `include/strata/kernels/elementwise.hpp:129-134`, already visible). The layer-split copies at `:784-786` and the PLE copy at `:836` stay separate: the PLE copy must follow the flag wait, and the hand-in copies are a different stage. Switch: `STRATA_COPY_MULTI=0` restores the three launches (default on: the kernel is the port's own and already in use in `mtp.cpp`, and the copies are words, bitwise by nature). Two nodes per window.
(c) In `argmax_rows_wanted()` (`verify_kernels.dp.cpp:1576-1591`), hand edit (or a `fixups.py` entry on the `return major == 8;` line): on SYCL return `multi_block_head_ops() && env "STRATA_ARGMAX_ROWS" == "1"` (default off, opt-in). This turns on `argmax_rows` for the verify head (`verify.cpp:1549-1550`) and the draft head (`mtp.cpp:1136`) at once; `row_top_prob_split` at `mtp.cpp:1143` already runs multi-block. Scratch is already allocated (`verify.cpp:537`, `mtp.cpp:463`).

**Must not change.** (a) Bitwise: the fold keeps the reduce's `sl = 0..79` order and the sliced kernel's dots; the direct kernel (`STRATA_GR_DOWN_SLICED=0`) and the un-folded sliced path stay reachable. INTEL.md:472-477 records that the sliced kernel's output is "the original one" against the direct kernel's near-tie flip; the fold must not move that. (b) Bitwise trivially (word copies). (c) `argmax_rows` is pinned bitwise against `sample_tokens`' greedy pick, ties to the lowest index, by `verify_parity` (`:60-63`); the sampled path (`greedy = false`) is untouched. Overall: bitwise the same tokens on the three standard prompts for all three parts.

**Tests.** Existing: `gr_multi_parity` (multi-token read vs single token, memcmp: it runs the default sliced path and so covers the fold when the switch is on; run it with `STRATA_GR_DOWN_FOLD=1` as a second ctest entry), `gr_parity`, `verify_parity` (`test_argmax`), `elementwise_parity`. New: in `gr_multi_parity`, run the read twice with the fold on the same `part2_cnt` (counter back to zero); a `copy_from_mapped_multi` case in `verify_parity` or `elementwise_parity` with the three real sizes (T, T x kStepCount, MT x (NH+NKV+IQ)) including an unaligned src (the volatile int32 path) memcmp against three `copy_i32_from_mapped`. For (c) `verify_parity` already covers it; add the engine-level identity run with `STRATA_ARGMAX_ROWS=1`.

**Bench.** (a) `gr_bench` for 1 to 6 tokens, fold off vs on (us per call and the checksum, which must not change); then the engine A/B (`STRATA_VERIFY_NODES=1`: 97 fewer nodes per window; tok/s and verify ms/window, 5 interleaved pairs, medians). (b) node count minus 2; tok/s within noise. (c) unitrace device time of `sampler_greedy_kernel` vs `argmax_rows_kernel` per window and per draft step; `STRATA_DECODE_TIMING=1` verify and draft ms/window; tok/s A/B.

**Expected gain, risk, effort.** (a) up to 97 nodes, 0.3 to 0.5 ms if the tail is cheap, zero or negative if it is not (the measurement decides; low risk because it is behind a switch and `gr_bench` catches a loss before the engine run). (b) 2 nodes, about 10 us, no risk. (c) one-block scan of about 1 MB per row (n_vocab x 4 bytes, several tens of us) replaced by about 61 blocks: roughly 20 to 30 us per pick, times 1 verify + 1 round + up to 3 steps per round, 50 to 150 us per round (E). Low risk; the failure mode in (a) is a visibility gap on `part2` (a wrong sum; `gr_multi_parity` catches it) and in (c) nothing new (an existing kernel with its parity test). 2 to 3 person-days in all, most of it (a).

**Depends on / conflicts with.** D1 first. (a) edits the same launch sequence in `fused_gr.dp.cpp` as D2 (`:1757-1875`): land D2 first, then (a) on top. (c) also changes `mtp.cpp:1136`'s pick, so run D5's bench after (c) is in. No upstream mirror for (b); upstream has (c) gated on CUDA major 8 (`src/kernels/cuda/verify_kernels.cu:835`) and (a) not at all (the sliced kernel is the port's). PR #1111 is a re-migration of the same `fused_gr.dp.cpp`; a 3-way merge of that file after (a) will conflict in the sliced-path launch block and must be resolved by hand.

---

### C2. The PLE flag wait at layer 1 is bounded by 20,000 reads; a slow PLE gather lets the window continue on stale rows

**Goal and measured motivation.** On the all-resident / device-plan path the host launches the window graph (`verify.cpp:1912`) and then gathers the window's PLE rows and raises the flag (`:1944-1953`), overlapping the NVMe reads with layer 0. The GPU's layer-1 wait (`:835`) is a bounded spin: 20,000 reads on an xe card (`sycl_doorbell.hpp:52-77`), at about 2.4 us per uncached read (`sycl_doorbell.hpp:20-26`) roughly 48 ms (E). INTEL.md:484-490 documents a 226 ms PLE read stall once per process ("Two-speed runs"), and INTEL.md:707-717 documents what a timed-out spin does on the A750: the GPU goes on with the inputs missing and the answer is garbage or token 0. When the stall lands on a decode window on the B70, `wait_flag_ge_kernel` runs out, `copy_from_mapped(ple_, m_ple_)` copies whatever `h_ple_` holds (the previous window's rows), and the window completes silently with the wrong PLE input; nothing on the host checks. The owner's instruction: make it fail loudly or wait correctly.

**Where.**
- `sycl/src/core/verify.cpp:831-837` (`if (l == 1 && ple_on) { if (grp == 0) { if (ar_on()) wait_flag_ge(m_flag_, 1, cs); copy_from_mapped(ple_, m_ple_, (int64_t) T * N, cs); }`), `:1912` (the window launch), `:1938-1956` (`no_host`, `steps`, `test_stall`, the `ar_on()` branch: `gather_batch` then `*flag = 1`), `:2055` (`cs_->wait()` for the window), `:2138-2142` (the only post-window check: `h_plan_err_`), `:1813, 2826` (`*h_flag_ = 0` per window), `:474, 477` (`h_flag_`/`m_flag_` and `h_plan_err_`/`m_plan_err_` mapped through `mapped(...)`), `:247-253` (`STRATA_TEST_VERIFY_STALL`, the #267 test hook: `g_test_stall`), `:1943` (`test_stall`), `:155` (`strata_ring_timeout`), `:255` (`release_gpu_waits`), `:3467-3480` (the pipelined `service()` path does the same gather and `*h_flag_ = 1`).
- `sycl/src/kernels/cuda/verify_kernels.dp.cpp:2047-2056` (`wait_flag_ge_kernel`), `:2443-2457` (launcher: `strata::spin_max(*q_of(stream))`), `include/strata/kernels/verify_kernels.hpp:43` (upstream declaration, not to be edited).
- `sycl/include/strata/sycl_doorbell.hpp:20-40` (`sys_load` uncached read, `sys_store`), `:42-77` (the bound: `STRATA_SPIN_MAX`, `STRATA_SYCL_SPIN_MAX`, `intel_gpu_driver() == "xe" ? 20000u : 2000000u`), `sycl/CMakeLists.txt:33-38`.
- `sycl/tools/fixups.py:80-105, 132-154` (the doorbell and `spin_bound_at_run_time` entries that shape these kernels).
- INTEL.md:484-490 (the stall), :707-717 (the A750 failure that motivated the per-device bound), :689 (`--ple-io ram` is the B70 default; the owner's plan in the review is `direct`).

**Current behaviour.** `verify_kernels.dp.cpp:2047-2056`:

```cpp
__dpct_inline__ void wait_flag_ge_kernel(const volatile uint32_t *flag,
                                         uint32_t value, uint32_t spin_max) {
    for (uint32_t spin = 0; spin < spin_max && strata::sys_load(flag) < value; ++spin) strata_spin_pause();
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
}
```

No output on the bound. `verify.cpp:1944-1953`, after the launch at `:1912`:

```cpp
    if (ar_on() && !test_stall) {
        if (do_ple) {
            const Clock::time_point tp = Clock::now();
            if (!ss.ple.table->gather_batch(ple_rows, (size_t) T, h_ple_, err)) return false;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            _mm_sfence();
            *flag = 1;
            ms_host += ms_since(tp);
        } else {
            *flag = 1;
        }
    } else
```

The test hook `STRATA_TEST_VERIFY_STALL=<window>` (`:247-253, 1943`) skips the flag for one window: today the engine does not notice.

**Change.** Two parts: always fail loudly, and optionally wait correctly.
1. Fail loudly (default on; cost is one store on the failure path only). New port-only launcher `wait_flag_ge_status(const uint32_t* flag, uint32_t value, uint32_t* status, void* stream)` in `verify_kernels.dp.cpp` with a header `sycl/include/strata/kernels/wait_status.hpp`; its kernel is `wait_flag_ge_kernel` plus `if (spin == spin_max && strata::sys_load(flag) < value) strata::sys_store(status, 1u);` where `status` is a mapped word (allocate `h_wait_err_` / `m_wait_err_` with `mapped(64, ...)` at `verify.cpp:477`, zero it per window beside `h_plan_err_` at `:1899`). Use it at `verify.cpp:835`. After the window's `cs_->wait()` (`:2055`), beside the `h_plan_err_` check at `:2138-2142`: `if (*(volatile uint32_t*) h_wait_err_ != 0) { err = "verify: the PLE rows were not ready when layer 1 needed them: the bounded wait (" + spin_max + " reads) ran out; the window is discarded. Set STRATA_PLE_WAIT_HOST=1 to gather the rows before the launch, or raise STRATA_SPIN_MAX"; return false; }`. Also in the pipelined path (`:3467-3480` / `pl_finish`), the same check wherever that path reads `h_plan_err_`. The request then fails with a message instead of continuing on stale rows; with the host-side fix below it should never fire.
2. Wait correctly, `STRATA_PLE_WAIT_HOST=1` (default off in v1; propose on by default for `--ple-io ram` after measuring): in `run()` move the `gather_batch` + fence out of the post-launch block (`:1946-1950`) to before the launch at `:1912` when the switch is on, and set `*flag = 1` before the launch as well (the GPU's wait then sees it on its first read). The overlap lost is the gather time: with `--ple-io ram` a memcpy of T x 12 rows (tens of us); with `--ple-io direct` T x 12 random 4 KB reads at queue depth 64 (`STRATA_IO_THREADS`, INTEL.md:290), about 0.1 to 0.5 ms (E) that today overlaps layer 0 (about 0.5 ms). Measure; if `direct` loses measurably, keep the overlap and rely on part 1 plus part 3.
3. A longer bound for this one wait, `STRATA_PLE_SPIN_MAX=<reads>` (default: the device bound): the wait at `:835` is the only device spin on the B70 path that waits for real host work (the device-plan windows "only spin in a failure", `sycl_doorbell.hpp:52-54`), so it may take a larger bound than the 20,000 chosen for failure spins; `wait_flag_ge_status` takes the bound as an argument (the launcher already computes `spin_max` per call at `:2444`). Risk: INTEL.md:254-259 and `sycl_doorbell.hpp:42-50` say an orphaned long spin on xe wedges the GT; a 226 ms spin that does end is not an orphan, but the xe job timeout for a kernel that is still running is not verified here: check `/sys/class/drm/card*/engine/*/preempt_timeout_ms` and `job_timeout_ms` on the card before raising it past about half of those.
4. Test hook: `STRATA_TEST_VERIFY_STALL=<window>` already skips the flag for one window on this path (`:1943-1944`). With part 1 the engine must now fail that window with the new message (today it continues); add this run to the acceptance list.
5. The GDN/QSA doorbell waits on the non-all-resident path (`:1376-1403`) have the same bounded shape but the host loop there detects a never-rung layer (`:1956-2010`, "never rang", timeouts); they are out of scope here. `mtp.cpp`'s drafter polls on the host side (`:1620-1630`), not the device; no change.

**Must not change.** Bitwise the same tokens when the flag arrives in time (the kernel's wait loop is unchanged; one extra store on the timeout path). With `STRATA_PLE_WAIT_HOST=1` the PLE rows are the same bytes, gathered earlier; output identical by construction. `STRATA_VERIFY_NO_HOST=1` semantics, the `h_plan_err_` check and the `#267` hook keep working. No change to the per-device bound for the other spins.

**Tests.** Existing: none for `wait_flag_ge` (the parity tests do not cover the doorbell kernels). New: in `verify_parity`, launch `wait_flag_ge_status` on a flag that is never raised with a small bound (say 1,000) and assert `status == 1` and that the kernel returns; then with the flag raised before launch and assert `status == 0`; both on a mapped word from `host_malloc_polled`. Engine: (1) the three standard prompts unchanged with and without `STRATA_PLE_WAIT_HOST=1`; (2) `STRATA_TEST_VERIFY_STALL=3` fails window 3 with the new message instead of continuing; (3) `STRATA_PLE_TRACE=1` runs until a slow run (INTEL.md:485: identical runs come out at two speeds) shows the 226 ms gather landing in a decode window, and the run either waits (part 2) or fails loudly (part 1), never silently continues.

**Bench.** This is correctness, so the bench only shows the cost: `STRATA_DECODE_TIMING=1` "GPU-reach wait" and "per-layer host" fields and ms/window with `STRATA_PLE_WAIT_HOST=1` against off, under `--ple-io ram` and under `--ple-io direct`, 5 interleaved pairs on the 2,184-token prompt (more PLE rows in flight), tok/s medians. Record the stall frequency with `STRATA_PLE_TRACE=1` across 10 starts.

**Expected gain, risk, effort.** No speed gain; removes a silent wrong-output mode that today costs one wrong window whenever a once-per-process 226 ms stall lands in decode. Risk low: part 1 is one store and one host check; part 2 moves 10 lines; part 3 is the only risky knob and stays at the device bound unless the sysfs timeouts allow more. 0.5 to 1 person-day.

**Depends on / conflicts with.** Independent of D1 to D6; do it in week 1 with D1 (review section 7) because it changes nothing in the profile and protects every later A/B from a stale-row window being counted as a valid run. The review's D8 (the adaptive tier's mirror table) is the other correctness note and is separate. PR #1111's re-migrated `verify.cpp` has the same bounded wait (`fixups.py:80-105` produce it); the status word is a port-only addition that merges cleanly as a hand edit.

---

### Tier 0 A/B: `STRATA_SH_STREAM=0` and `STRATA_LFUSE=1`

Both switches exist, are read once per process, and were never A/B'd on Xe2. Both are "tokens identical by design", so a token diff is part of each run's pass criterion.

**What they do.**
- `STRATA_SH_STREAM` (`sycl/src/core/verify.cpp:100-117`): `=1` (the default off HIP) forks the shared expert onto the second in-order queue `sh_cs_` per layer and joins it back: `dpct::sync_barrier(ev_fork_, cs); sh_cs_->ext_oneapi_submit_barrier({*ev_fork_})` at `:1222-1223` (or late, after the doorbell publish, `:1285-1286`), `dpct::sync_barrier(ev_join_, sh_cs_)` at `:1322`, `cs->ext_oneapi_submit_barrier({*ev_join_})` at `:1415-1421`: three barrier nodes per layer, 144 per 48-layer window, and a graph that is no longer linear. `=0` runs the shared expert on `cs` with no barriers. The comment records +30% on HIP with the fork off (62.9 vs 43.4 t/s on a gfx1201) and that the profiler never showed it because `prof_on_` turns the fork off (`:1216`). The drafter's own side queue is a different switch (`STRATA_MTP_SHARED_BRANCH`, `mtp.cpp:1006`) and is not part of this A/B.
- `STRATA_LFUSE=1` (`verify.cpp:67-69, 760-763`): the shared expert's gate row rides in the router GEMV (`bf16_gemv_fp32_mmvf_multi_aux`, `:1235-1237`), gate + up share one launch (`shared_expert_multi` flag 2 at `:1313`), and the sigmoid + row scale move into the combine (`native_moe_combine_multi_hits_gated`, `:1422-1425`); "all bitwise" upstream. Conditions (`:760-763`): `ar_on() && native_moe_combine_enabled() && dec_batch && n > 1 && n <= 8 && shared_expert_native_bf16_enabled()`. `ar_on()` is `all_resident_ && !ar_off_` (`verify.hpp:338`), and `all_resident_` is false when any routed expert is outside VRAM (`verify.cpp:640-651`), so LFUSE is inert for IQ2_XS with a host mirror and live for the Coder (all experts in VRAM). The gate row's aux path additionally needs `STRATA_MMVF_ROWS=1` (`native_bf16.dp.cpp:539-543`); without it `sg_ready` stays false and only the pair fusion and the combine change (`STRATA_LFUSE_GATE`, `STRATA_LFUSE_PAIR` sub-switches at `:68-69`).

**How to run.** One binary (the AOT build), the by-hand command at the top of this file, `--max-new 256 --greedy`, on `benchy-short.ids` (20 tokens) and `benchy-long.ids` (2,184 tokens), the Coder first (both switches live), then IQ2_XS (SH_STREAM only; LFUSE is expected inert there and the run confirms it with an unchanged node count). Each configuration is one environment: A = defaults (`STRATA_SH_STREAM` unset, `STRATA_LFUSE` unset), B1 = `STRATA_SH_STREAM=0`, B2 = `STRATA_LFUSE=1`, B3 = `STRATA_LFUSE=1 STRATA_MMVF_ROWS=1`, B4 = `STRATA_SH_STREAM=0 STRATA_LFUSE=1`. Interleave A B A B A B A B A B (5 pairs per B) so clocks and the once-per-process PLE stall (INTEL.md:484-490) average out; every run also sets `STRATA_DECODE_TIMING=1` and the first run of each configuration `STRATA_VERIFY_NODES=1`. Through `sycl/serve/strata-sycl.sh` every `STRATA_*` variable of the environment is passed into the container (`strata-sycl.sh:25-29`), so the same A/B can be repeated in serve mode later.

**What to compare.**
1. Output identity: the 256 token ids of every B run against the A run on the same prompt, byte for byte. Any difference fails the switch (both are "bitwise by design"; a difference is a bug to file, not a trade).
2. Node count (`STRATA_VERIFY_NODES=1`, the `strata verify: the %d-token window graph has %zu nodes:` line per captured window size): expect B1 about 144 fewer than A on the 6-token window (3 barrier nodes x 48 layers); B2/B3 about 190 fewer per the review (the separate gate GEMV, the second of gate/up, and the sigmoid kernels); B4 the sum.
3. `strata decode timing:` ms/window and its `verify` and `draft` fields, medians over the 5 runs per configuration; tok/s for the 256-token decode, medians; the difference A vs B in tok/s with the run-to-run spread (INTEL.md:778 puts noise at about 1 tok/s).
4. One `unitrace -d` run of A and of the best B over the same 256 tokens: the per-kernel device time table (the shared-expert kernels should appear in the same amounts; what changes is the gaps and the join barriers) and the total device time per window.
5. For LFUSE on IQ2_XS: the node count must equal A's (inert); if it does not, `ar_on()` was true for that run and the result belongs with the Coder's.

**Decision.** A switch that is identical in output and faster by more than the spread in both prompts becomes the default in `sycl/serve/strata-sycl.sh`'s `setting` map (`:25`) and in `setup_intel.py`, with the measurement in `docs/INTEL.md`. A switch that is slower is recorded with its number so nobody tries it again (as INTEL.md does for `STRATA_GDN_KEYHEAD`). No code changes; half a day with the card.

# Part B. Expert kernels and the mirror


Scope: fork `sam-fakhreddine/strata` at `fb58e0d` (engine 0.1.41), files under `sycl/` only. Line numbers below were read from the files on 2026-10-09; every one is to a function, struct or kernel named next to it. Sources of numbers: `docs/INTEL.md` (INTEL) and the review `strata-b70-engine-plan-20261009-025812.md` (REVIEW). "(E)" marks an estimate, "not verified" marks something that must be checked on the card.

Shared facts used by several items:

- The default decode expert path on the B70 is the port's lane kernels: `native_expert_grouped` (`sycl/src/kernels/cuda/iq_kernels.dp.cpp:5012`) switches on `L.gu_type` through `STRATA_GU_FMTS` (`:771-774`, the list is `16 17 18 21 22 23 29 42 12 13 6 2 3 8`) into `launch_gu<T>` (`:4172-4187`), which takes `launch_gu_lanes` (`:4121-4128`) unless `STRATA_EXPERT_SPLIT=1`; `launch_gu_port<TG, LN>` (`:4107-4120`) launches `native_gu_kernel<TG, LN>` (`:1812-1853`) with `[[sycl::reqd_sub_group_size(32)]]`, 256 threads, `LN = kExpertLanes = 8` lanes a row (`:795-799`, `STRATA_GU_LANES` / `STRATA_DOWN_LANES` at run time, `:4101-4106`). The down projection is the same shape: `launch_down_port` / `native_down_kernel<TD, LN>` (`:4129-4152`, `:1983-2024`).
- Inside `native_gu_kernel`, `row_entries<TG, LN>` (`:1784-1810`) takes a group's entries four at a time through `row_dot_multi<TY, LANES, E>` (`:1753-1781`) when `Multi<TY>::has`, else one at a time through `row_dot_lanes<TY, LANES>` (`:808-816`), which calls `Fmt<TY>::dot` per entry. `Multi<TY>` exists for 18 (IQ3_XXS), 22 (IQ2_S), 21 (IQ3_S), 20 (IQ4_NL) and 42 (Q2_0) (`:1587-1751`); the default `Multi` has `has = false` (`:1585`). IQ2_XS (17) and IQ1_M (29) therefore decode the weight once per entry.
- Parity: `iq_multi_parity` (`sycl/src/kernels/iq_multi_parity.cpp`, built at `sycl/CMakeLists.txt:238-241`, `--bench` prints us per call) compares the grouped kernel with `iq_set_old_kernels(true)` against `false` by `memcmp` (`:383-386`, `:418-424`) for gate/up types `{16, 17, 18, 21, 22, 23, 29, 42}` (`:416`) and checks IQ2_XS against a CPU decode (`:180-190`). `native_grouped_parity` (CMake `:247-250`) compares the grouped launches against the v1 launches bitwise. `native_expert_parity <shard1.gguf> [layer]` (CMake `:292-298`, needs `STRATA_NATIVE_EXPERTS`) checks the GPU kernel against ggml's float reference on real rows and, with `NATIVE_BENCH=1`, times one layer's worth of hits (G = 10 experts, E = 4 entries each, 20 iterations; `sycl/src/kernels/native_expert_parity.cpp:249-275`). `sycl/src/kernels/native_expert_bench.cpp` exists but is not in `sycl/CMakeLists.txt` and still contains `cudaMalloc` (`:90-91`): it is not the bench; where the task says "native_expert_bench", use `native_expert_parity ... NATIVE_BENCH=1` and `iq_multi_parity --bench`.
- The mirror: `GgufExpertSource::mirror` (`sycl/src/core/gguf_expert_source.cpp:115-167`) reads every expert without a VRAM slot into one `sycl::malloc_host` block; `generate.cpp` builds the device table `mirror_table_d` once at start (`sycl/src/program/generate.cpp:4725-4732`) and hands it to the plan kernel through `resident_plan_set_mirror` (`generate.cpp:6850`; `verify_kernels.dp.cpp:2353-2360`). `resident_plan_kernel` (`sycl/src/kernels/cuda/verify_kernels.dp.cpp:2073-2221`) points a group at the mirror address when the expert has no slot (`:2099-2104`, `:2186-2187`); the expert kernels then read the blob from host USM over PCIe, 4 to 8 bytes a lane.
- `STRATA_VERIFY_NO_HOST=1` is required whenever experts are outside VRAM (INTEL 748-752; `sycl/src/core/verify.cpp:1936-1941`): the host never publishes a per-layer plan, so a group the device cannot plan has no fallback.
- Output identity: the three standard greedy runs (Coder 19-token prompt, Coder 2,184-token prompt, IQ2_XS; 256 tokens, INTEL 564) must produce the same tokens before and after each item unless the item says it changes rounding (none below does).

---

### D4. `Multi<17>` (IQ2_XS) and `Multi<29>` (IQ1_M) decode-once lane paths, plus aligned 4-byte loads in the IQ2_XS single-entry dot

**Goal and measured motivation.** The expert dot kernels are ALU bound in the i-quant grid decode (77% XVE active, INTEL 295-298), about 4 to 5 ms of a 40 ms round (REVIEW section 1). For IQ3_XXS, IQ3_S, IQ2_S, IQ4_NL and Q2_0 the port decodes each weight word once per four entries (`Multi<TY>`); IQ2_XS and IQ1_M, the gate/up formats of the two models that matter here (Flash-Next IQ2_XS, Coder IQ1_M), still redo the grid lookup, sign unpack and SWAR compare/subtract once per entry (3.3 to 3.6 entries per routed expert per window on code). unitrace per-call figures for the neighbouring formats are IQ2_S 52.5 us, IQ3_S 60.8 us, IQ3_XXS 60.3 us (INTEL 383-386, comment at `iq_kernels.dp.cpp:382-384`); REVIEW D4 expects IQ2_XS gate/up from about 60 to about 40 us a call and 1 to 2 ms a round. For the roughly 130 mirrored experts a round on IQ2_XS (184 MB, REVIEW section 1) the decode-once path also reads each PCIe-resident weight word once per four entries instead of once per entry. The second part: `vec_dot_iq2_xs_q8_1` loads its codes with `get_int_b2` (two 16-bit loads, `iq_kernels.dp.cpp:47-52`); `block_iq2_xs` is 74 bytes (`third_party/ggml/ggml-common.h:388-392`: half d, 32 uint16 qs, 8 uint8 scales), so the `qs` run of every odd block starts 2 mod 4 and the B70 splits the load (the same finding as Q6_K, INTEL 446-452). The other formats already use `load8_a2` / `load4_a2` (`:367-380`) in their `prep`.

**Where.**
- `sycl/src/kernels/cuda/iq_kernels.dp.cpp:172-209` `vec_dot_iq2_xs_q8_1` (codes at `:179-180`, scales `:181-182`, loop `:184-202`, integer scale `:203`, float `:204-207`).
- `sycl/src/kernels/cuda/iq_kernels.dp.cpp:313-347` `vec_dot_iq1_m_q8_1`.
- `sycl/src/kernels/cuda/iq_kernels.dp.cpp:367-380` `load8_a2`, `load4_a2`; `:385-390` the `STRATA_GRID_SLM` and `STRATA_IQ4NL_FAST` compile-time defaults.
- `sycl/src/kernels/cuda/iq_kernels.dp.cpp:735-736` `Fmt<17>`, `:747-748` `Fmt<29>` (`step = 1` for IQ1_M, `2` for IQ2_XS).
- `sycl/src/kernels/cuda/iq_kernels.dp.cpp:910-954` `SplitLs2` and `Split<17>` (the per-column decode-once split, the model for the arithmetic), `:1076-1123` `Split<29>`.
- `sycl/src/kernels/cuda/iq_kernels.dp.cpp:1583-1751` `MultiW`, `Multi<>` default and the five specialisations (`Multi<21>` at `:1662-1697`).
- `sycl/src/kernels/cuda/iq_kernels.dp.cpp:1753-1781` `row_dot_multi<TY, LANES, E>`; `:1784-1810` `row_entries`; `:1812-1853` `native_gu_kernel` (`rib`/`sub` at `:1840-1841`); `:4107-4128` `launch_gu_port` (`reqd_sub_group_size(32)` at `:4117`) / `launch_gu_lanes`; `:4129-4152` `launch_down_port` (`:4141`) / `launch_down_lanes`; `:4172-4187` `launch_gu`.
- `sycl/src/kernels/cuda/iq_kernels.dp.cpp:2811-2816` `env_on`, `g_old_kernels`; `:4225-4226` `iq_set_old_kernels`.
- `sycl/src/kernels/iq_multi_parity.cpp:292, 316, 412, 416` (the type lists the test and its `--bench` iterate; 17 and 29 are already in them), `:383-386` (`run(old)` toggles `iq_set_old_kernels`).
- `include/strata/kernels/iq_kernels.hpp:62-77` (upstream header: `native_expert_grouped`, `native_grouped_set_v1`, `iq_set_old_kernels`; not editable, see step 5).

**Current behaviour.** `row_entries` falls to the per-entry loop for 17 and 29:

```cpp
    int e = e0;
    if constexpr (Multi<TY>::has) {
        for (; e + 4 <= e1; e += 4) { ... row_dot_multi<TY, LANES, 4>(wr, xs, nb, sub, o, grid); ... }
        ...
    }
    for (; e < e1; ++e) {
        const float v = row_dot_lanes<TY, LANES>(wr, x + (size_t) ent_idx[e] * x_stride, nb, sub);
        if (sub == 0) dst[(size_t) e * dst_stride] = v;
    }
```
(`iq_kernels.dp.cpp:1789-1809`). The generic multi loop decodes once and applies per entry:

```cpp
        MultiW m;
        M::prep(row, kbx, iqs, m, grid);
        for (int e = 0; e < E; ++e) {
            int u[8]; float ds;
            M::acts(xs[e] + kbx * (F::qk / 32), iqs, u, ds);
            int s0 = 0, s1 = 0;
            for (int j = 0; j < M::NW / 2; ++j) s0 = ggml_cuda_dp4a(m.w[j], u[j], s0);
            for (int j = M::NW / 2; j < M::NW; ++j) s1 = ggml_cuda_dp4a(m.w[j], u[j], s1);
            s[e] += M::finish(s0, s1, m, ds);
        }
```
(`:1763-1777`). `vec_dot_iq2_xs_q8_1` keeps two half sums over l0 < 4 and l0 >= 4 and scales them as `(sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4`, then `d = half2float(bq2->d) * bq8_1[iqs / 2].ds[0]; return d * sumi;` (`:203-207`), the same expression `Multi<22>::finish` already produces for IQ2_S (`:1655-1658`). `vec_dot_iq1_m_q8_1` differs: beside the two integer sums it keeps two float sums `sumf[l0 / 4] += delta * sumy` where `sumy` is the byte sum of the activation words and `delta` depends only on the weight's `qh` bit (`:322-335`), then `return d * ((sumi[0] + sumf[0]) * sc0 + (sumi[1] + sumf[1]) * sc1)` (`:343-346`); the generic `finish(s0, s1, m, ds)` cannot express it.

**Change.**
1. `Multi<17>` (new, after `Multi<22>` at `:1660`): `has = true`, `NW = 8`. `prep` = the body of `Split<17>::load` (`:926-953`) written into `MultiW`: `m.w[l0 + 0] = swar_sub4(grid_pos.x() ^ signs0, signs0)` and `m.w[l0 + 1]` likewise for l0 = 0, 2, 4, 6, with the codes extracted by shifts (never through a `uint16_t*` to an `int2`, INTEL 597-601); `m.a = scales[iqs / 2] & 0x0F`, `m.b = scales[iqs / 2] >> 4`, `m.d = half2float(bq2->d)`. Loads under `#if STRATA_IQ4NL_FAST`: `const sycl::int2 qq = load8_a2((const uint8_t*) bq2->qs + 4 * iqs); q2_lo = qq.x(); q2_hi = qq.y();`, else the two `get_int_b2` as today (the same pattern as `Multi<18>::prep`, `:1591-1597`). `acts` = `Multi<22>::acts` (`:1652-1656`). `finish` = `Multi<22>::finish` (`:1657-1659`) verbatim: `return m.d * ds * (float) ((s0 * m.a + s1 * m.b + (s0 + s1) / 2) / 4);`. The j order of the generic loop (0..3 into s0, 4..7 into s1) matches `sumi0` / `sumi1` of the single dot (l0 = 0 and 2 feed `grid_l, grid_h` = j 0..3), so the integer sums are the same values in the same order.
2. `Multi<29>` (new): `has = true`, `NW = 8`. Extend `MultiW` with `float delta[4]` (`:1583`; the other formats leave it dead, the optimiser removes unused fields; confirm with the T2b spill dump). `prep` = `Split<29>::load` (`:1079-1105`): `m.w[l0] = (grid >> 0) & 0x0F0F0F0F`, `m.w[l0 + 1] = (grid >> 4) & 0x0F0F0F0F`, `m.delta[l0 / 2] = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f * IQ1M_DELTA / 0x08)`, `m.a = sc0`, `m.b = sc1`, `m.d = half2float(scale.f16)`. `acts`: `u[j] = get_int_b4(y[iqs].qs, j)`, `ds = y[iqs].ds[0]` (index `iqs`, not `iqs / 2`: `Fmt<29>::step = 1`, as `Split<29>::apply` reads `bq8_1[iqs]`, `:1111-1112`, `:1120`). Change the generic `finish` signature to `finish(int s0, int s1, const MultiW& m, float ds, const int* u)` in all six specialisations and at the call site `:1776` (the existing five ignore `u`). `Multi<29>::finish`: for k = 0..3 `int sumy = 0; sumy = dp4a(u[2k], 0x01010101, sumy); sumy = dp4a(u[2k + 1], 0x01010101, sumy);` then `float sumf0 = 0.0f; sumf0 += m.delta[0] * sumy0; sumf0 += m.delta[1] * sumy1;` and `sumf1` from k = 2, 3 (keep the `0.0f +=` form and the order: `-fp-model=precise` must not see a different expression); `return m.d * ds * ((s0 + sumf0) * m.a + (s1 + sumf1) * m.b);`. This is `vec_dot_iq1_m_q8_1:343-346` with `d = m.d * ds` formed first, as there.
3. Opt-in switch, old path reachable. Add `template<int TY> inline constexpr bool kMultiOptIn = (TY == 17 || TY == 29);` next to `kSplit` (`:866`). Give `native_gu_kernel` a third template parameter `bool MULTI_NEW` passed into `row_entries<TG, LN, MULTI_NEW>`, whose first branch becomes `if constexpr (Multi<TY>::has && (!kMultiOptIn<TY> || MULTI_NEW))`. `launch_gu_port<TG, LN>` instantiates both and picks at run time: `static const bool on = env_on("STRATA_IQ_MULTI_XS1M"); const bool use = on && !g_old_kernels;` (name: `STRATA_IQ_MULTI_XS1M`, default off; `iq_set_old_kernels(true)` forces the per-entry path so the parity test compares something real). Two more kernel instances per format in the AOT image; the `dpct_kernel_name<class native_gu_port, ...>` needs the extra scalar. The down kernels are untouched (IQ4_NL / Q2_0 down already have `Multi`).
4. Aligned loads in the single-entry dot: in `vec_dot_iq2_xs_q8_1` (`:179-180`) replace the two `get_int_b2` with `load8_a2((const uint8_t*) bq2->qs + 4 * iqs)` under `#if STRATA_IQ4NL_FAST` (`-DSTRATA_IQ4NL_FAST=0` keeps ggml's loads, as for IQ4_NL, `:389`). Same ints out, so the dot is bitwise unchanged; `Split<17>::load` (`:929-930`) gets the same two lines so the per-column mmvq path matches.
5. Test hook: the port cannot edit `include/strata/kernels/iq_kernels.hpp`. Declare `void iq_set_multi_optin(bool on);` in a port-only header (new `sycl/include/strata/kernels/iq_kernels_sycl.hpp`, same pattern as `sycl/include/strata/kernels/resident_plan_mirror.hpp`) and define it next to `iq_set_old_kernels` (`:4225`), so `iq_multi_parity` can turn the new path on without the environment.
6. No `fixups.py` entry: these are hand edits to the port's own `// SYCL port: multi-entry dots` block (`:1579-1582`), which dpct never produced; a re-migration (INTEL "Keeping up with upstream") merges them with `git merge-file`. Upstream `src/kernels/cuda/iq_kernels.cu` has no `Multi<>` (checked: no match), so there is no upstream pattern to mirror; the model is the port's `Multi<21>` (`:1662-1697`).

**Must not change.** The outputs of the three standard runs; `iq_multi_parity` must report "bitwise equal to the old kernel" for 17 and 29 with the new path on, and `native_expert_parity` must stay within its tolerance on IQ1_M rows (rel 1.1e-2 against ggml's float reference, INTEL 265-266). The integer scale expressions and the `d * sumi` float order stay exactly those of the single dots. The group stride (#363), the `n_groups` early return (`:1821-1822`) and the `STRATA_GRID_SLM` branch are untouched. Default behaviour is unchanged until `STRATA_IQ_MULTI_XS1M=1`.

**Tests.**
- Existing: `ctest --test-dir build-sycl -R 'iq_multi_parity|native_grouped_parity|iq_parity'`; `iq_multi_parity` with the hook on (step 5) is the bitwise gate for 17 and 29 (its grouped test, `:316, 416`, random gu types, duplicated entries, `cap_groups > groups`).
- Add to the SYCL copy of `iq_multi_parity.cpp`: run the grouped check twice, `iq_set_multi_optin(false)` and `(true)`, and `memcmp` the two outputs; assert zero differences. Also an entries-per-group sweep of 1, 2, 3, 4, 5, 7 so the `e + 4`, `e + 2` and tail branches of `row_entries` all run for 17 and 29.
- `native_expert_parity <coder shard1> 5 20 40` (IQ1_M) with and without `STRATA_IQ_MULTI_XS1M=1`: same pass/fail lines and identical `got_g` checksum (add a printed hash of `got_g` if there is none).
- Engine: the three standard runs with `STRATA_IQ_MULTI_XS1M=1`, tokens `diff`ed against the baseline.

**Bench.** `iq_multi_parity --bench` (us per call for types 17 and 29 at 1 to 8 columns and the grouped shapes, `:289-330`), `NATIVE_BENCH=1 native_expert_parity <shard> <layer>` on an IQ1_M layer and an IQ2_XS layer, each with `STRATA_IQ_MULTI_XS1M` unset and `=1`, warm clocks first (INTEL 270-271). `unitrace -d` plus `sycl/rank_kernels.py` on 256 decoded tokens: device time of `native_gu_port<17, 8, *>` per call and per round. Engine: tok/s on the 19/20-token and 2,184-token prompts, 256 greedy tokens, 4 interleaved pairs, medians, IQ2_XS and Coder; plus the forced-miss run `--expert-cache 8000` on IQ2_XS (mirrored experts read once per four entries).

**Expected gain, risk, effort.** IQ2_XS gate/up about 60 to about 40 us a call (E, REVIEW), 1 to 2 ms a round (3 to 5% of decode on IQ2_XS); on the Coder, IQ1_M gate/up a similar fraction of its 4 to 5 ms of expert-dot time (E). Risk low: the arithmetic is copied from the single dots and the two parity tests are bitwise; the failure mode is a non-bitwise `finish` (caught by `iq_multi_parity`) or register pressure from `MultiW::delta` slowing the other formats (caught by the bench; then give `Multi<29>` its own `W` type instead). 1 to 2 person-days.

**Depends on / conflicts with.** None to start. Feeds T2b (the same kernels under SIMD16) and the ESIMD dot (REVIEW section 4, "only after D4"). No conflict with maxfridbe PR #1111 or the maxious pre-unpack (those touch the dense Q6_K/Q5_K path and the prompt path, not the expert lane kernels). A re-migration after an upstream `iq_kernels.cu` change merges these hand edits by `git merge-file`.

---

### D7. Per-layer mirror-hit counter and group-size histogram in `resident_plan_kernel`, printed per request

**Goal and measured motivation.** Every mirror decision (how big the mirror must be, whether prefetch pays, what `--expert-cache` to choose) needs the number of experts a layer-window reads over PCIe and how many entries share an expert. The host pool's counters (`pcie_experts`, `cache_hits`, `multi_misses`) are incremented by the host planner (`sycl/src/core/expert_source.cpp:2655-2668`, `++d.pcie_experts` at `:2668`) and the device-plan path skips the host planner entirely (`verify.cpp:1375-1381`, `:1398-1400` "no CPU share when the device planned the group"), so with `STRATA_VERIFY_DEVICE_PLAN=1` and `STRATA_VERIFY_NO_HOST=1` the `STRATA_DECODE_TIMING` line prints `VRAM hits 0.00, PCIe 0.00` per layer-window (the fields at `generate.cpp:11173-11185`). The only mirror number measured so far is the forced-miss run (`--expert-cache 8000`, 1,879 experts / 3.6 GiB out: decode 40.9 vs 43.3 tok/s, INTEL 433-437) and the start-up count (18,329 in VRAM / 6,247 mirrored for IQ2_XS, INTEL 499-500). REVIEW D7: "the number every mirror decision needs", 0.5 day.

**Where.**
- `sycl/src/kernels/cuda/verify_kernels.dp.cpp:2062-2066` `kResidentPlanMax = 128` and its static asserts; `:2073-2221` `resident_plan_kernel` (per-entry residency and `maddr` at `:2097-2107`; the `s_bad` branch at `:2110-2125`; `first_j`, `rank_in_group`, `count_same` at `:2128-2140`; `is_first` at `:2141`; `ptr[grp_idx]` at `:2186-2187`; `counts[]` at `:2201-2208`).
- `sycl/src/kernels/cuda/verify_kernels.dp.cpp:2353-2360` `g_mirror_res`, `g_mirror_table`, `resident_plan_set_mirror`; `:2361-2384` the `resident_plan` launcher (the layer's mirror slice from `res_layer - g_mirror_res` at `:2364-2366`).
- `sycl/include/strata/kernels/resident_plan_mirror.hpp:1-10` (port-only header; the new setter goes here).
- `sycl/src/core/verify.cpp:1259-1267` the two `resident_plan` calls inside the captured window (ar_on and device_plan_); `sycl/include/strata/core/verify.hpp:50-57` `VerifyHits`.
- `sycl/src/program/generate.cpp:6845-6855` where `resident_plan_set_mirror` is called; `:10285-10294` `dec_timing` and `DecSnap`; `:11173-11185` the per-request `strata decode timing:` line; `:12418-12421` the `pcie experts` print of the `generate` command.
- `sycl/src/kernels/resident_plan_parity.cpp` (CMake `:252-258`): calls at `:137` and `:202` pass no `mir` (0 occurrences of `mir` in the file).

**Current behaviour.** The kernel decides residency per entry and marks groups:

```cpp
    unsigned long long maddr = 0;   // SYCL port: a host-mirrored expert (not in VRAM): its pinned mirror, read over PCIe
    if (tid < n) {
        eid = ids[tid];
        s_ids[tid] = eid;
        slot = (eid >= 0 && eid < n_expert) ? res[eid] : -1;
        if (slot < 0 && eid >= 0 && eid < n_expert && mir != nullptr) maddr = mir[eid];
        if (slot < 0 && maddr == 0)
            dpct::atomic_fetch_or<sycl::access::address_space::generic_space>(&s_bad, 1);
    }
    ...
    const bool is_first = (tid < n && first_j == tid && (slot >= 0 || maddr != 0));
    const int my_cnt = is_first ? count_same : 0;
```
(`:2099-2107`, `:2141-2142`). Nothing is counted; `counts[2] = 0` (`:2207`, the host plan's PCIe group count, always zero on the device). The launcher knows the layer only implicitly: `mir = g_mirror_table + (res_layer - g_mirror_res)` (`:2365-2366`).

**Change.**
1. Device buffer: `uint32_t* d_plan_stats` of `n_layers * kPlanStatWords` (`kPlanStatWords = 16`: `[0]` groups, `[1]` mirror groups (is_first with `slot < 0`), `[2]` mirror entries (`tid < n && slot < 0`), `[3]` entries, `[4 + min(count_same, 8) - 1]` histogram of group sizes 1..8 (from `is_first` threads), `[12]` plans with `s_bad`), allocated in `generate.cpp` next to the `resident_plan_set_mirror` call (`:6850`) before any graph capture (the pointer is baked into the captured graphs), zeroed with a waited `memset`.
2. Setter in `resident_plan_mirror.hpp`: `void resident_plan_set_stats(uint32_t* d_stats, int n_expert);` storing `g_plan_stats`, `g_plan_n_expert`. In the launcher (`:2361-2384`) compute `layer = (res_layer - g_mirror_res) / g_plan_n_expert` when `g_mirror_res != nullptr` (the same pointer arithmetic the mirror slice uses) and pass `stats = g_plan_stats ? g_plan_stats + layer * kPlanStatWords : nullptr` as a new last kernel argument. Note the layer is derivable only when `hits_.d_res` is set; `generate.cpp:5649-5667` always sets `thits.d_res` on the graph-hits path, which is the only path that plans on the device.
3. Kernel: after `is_first` (`:2141`), when `stats != nullptr`: `if (is_first) { atomic_add(stats[0], 1); if (slot < 0) atomic_add(stats[1], 1); atomic_add(stats[3 + min(count_same, 8)], 1); } if (tid < n) { atomic_add(stats[3], 1); if (slot < 0) atomic_add(stats[2], 1); }`, and in the `s_bad` branch (`:2110-2111`) `if (tid == 0) atomic_add(stats[12], 1)`. Use `sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>` (device memory, not host-mapped: nothing in INTEL's "host-mapped flags" warning applies). At most 80 atomics on 13 words per call, 96 calls a round: not measurable against a 2.5 us node (E), but keep it switchable: the kernel takes `stats == nullptr` as off, and `generate.cpp` passes the buffer only when `STRATA_PLAN_STATS=1` (default off) or `STRATA_DECODE_TIMING` is set. Both the ar_on call (`verify.cpp:1259`) and the device-plan call (`:1264`) get it through the launcher; `resident_plan_par_kernel` (`:2223`) is defined but not launched by `resident_plan` (checked `:2372-2381`): leave it.
4. Print: at the per-request timing site (`generate.cpp:11173`), when the buffer exists: `dpct::get_in_order_queue().memcpy(host, d_plan_stats, bytes).wait()` (the queue is idle between requests), then one line per request `strata plan stats: windows W; per layer-window groups G, mirror groups M (x%), mirror entries ME, group sizes 1:a 2:b 3:c 4:d 5+:e; bad plans B` and, under `STRATA_PLAN_STATS=2`, one line per layer with its mirror share (so a layer whose experts are mostly mirrored shows up). Then `memset` the buffer to zero (waited) so the next request starts clean. Also add `mirror groups / layer-window` to the `DecSnap` fields for the existing line.
5. The `serve` path and the `generate` command share the site above; the `pcie experts` print at `:12418-12421` stays as the host-pool number (it remains correct whenever the host plans).

**Must not change.** The plan words (`counts`, `start`, `dst`, `tok`, `ptr`, `start2`) and their layout; `resident_plan_parity` must still pass by `memcmp` with `stats == nullptr` and with a buffer. Output identity: counting does not touch any value the expert kernels read. With `STRATA_PLAN_STATS` unset the kernel argument is `nullptr` and the graph is the same as today apart from one pointer argument.

**Tests.**
- Existing: `resident_plan_parity` (`:137`, `:202`): still passes.
- Add to `resident_plan_parity.cpp`: a mode that passes a `mir` table (half the experts non-resident but mirrored) and a stats buffer; assert after each call `stats[0] == distinct experts`, `stats[1] == distinct experts with res < 0`, `stats[2] == entries with res < 0`, `stats[3] == n`, and that the histogram sums to `stats[0]` with `sum(size_i * count_i) == n`. This also closes the gap that the mirror branch (`:2104`) has no test today.
- Engine: `STRATA_PLAN_STATS=1` on the IQ2_XS standard run: the per-request line's mirror share should match the start-up ratio (6,247 of 24,576 experts mirrored, weighted by routing; expect well under 25% because the profile keeps the hot experts resident) and `bad plans` must be 0; on the Coder (all resident) `mirror groups` must be 0.

**Bench.** None needed beyond showing no regression: 256 greedy tokens on the 19-token prompt, `STRATA_PLAN_STATS=1` vs unset, 4 interleaved pairs, medians (expect equal within noise). The item's output is the forced-miss run `--expert-cache 8000` on IQ2_XS with the stats on: the per-layer mirror share is what T2a is sized from.

**Expected gain, risk, effort.** No speed gain; it produces the numbers for D8, D9 and T2a. Risk none (off by default; atomics on device memory). 0.5 person-day.

**Depends on / conflicts with.** Nothing. T2a needs it first. D3 (router + top-10 + plan in one kernel) would move this code into the fused kernel; keep the counting in a small inline function so it moves with it.

---

### D8. Correctness: the adaptive tier against the start-written mirror table

**Goal and measured motivation.** With `--stream-experts` and experts outside VRAM, the adaptive tier is on by default (`adapt_every = 4`, `adapt_swaps = 96`, `generate.cpp:613, 640`; it runs whenever `drive.d.usage` is non-empty, which `:7623-7626` sets when not every expert is resident). Each round it evicts up to 96 experts and marks them `kNotResident` in `host_res`, then uploads the residency table (`res_upload`). The device mirror table was written once at start from the experts that were missing then (`generate.cpp:4725-4732`); an evicted expert has no entry, so the next window that routes it gets `slot < 0 && maddr == 0` in `resident_plan_kernel` (`verify_kernels.dp.cpp:2104-2107`): `s_bad`, the plan is not written and `*skip = 0` (`:2110-2111`). Under `STRATA_VERIFY_NO_HOST=1` (required for any run with misses, INTEL 748-752) no host plan ever comes: `wait_flag_ge_or_kernel` spins `spin_max` reads (20,000 on xe, `sycl/include/strata/sycl_doorbell.hpp:60-77`) and falls through (`verify_kernels.dp.cpp:2310-2321`), the group runs on whatever `pl` holds, and the output is silently wrong, or, in the all-resident graph (`skip == nullptr`), `plan_err` is set and the plan emptied (`:2112-2125`). REVIEW section 1 flagged this; whether a swap has actually happened in the recorded runs is not verified (the recorded IQ2_XS and IQ3_S runs were greedy test runs of 64 to 256 tokens: 16 to 64 adaptive rounds at `adapt_every 4`, so swaps were possible whenever an expert was routed twice, the `u[e] >= 2.0f` candidate gate at `:7719` (blocking tier), `:7946` (`a_choose`) and `:12063` (the `generate` command).

**Where.**
- `sycl/src/program/generate.cpp:4685-4744` the mirror build (`miss` list `:4695-4703`, cap `:4704-4712`, `mirror()` call `:4715`, device table `:4725-4732`, the `REFUSED ... neither in VRAM nor mirrored` check `:4735-4742`).
- `sycl/src/program/generate.cpp:7623-7626` `all_experts_resident` and `drive.d.usage`; `:7629-7634` `--expert-profile-save` needs the tier.
- `sycl/src/program/generate.cpp:7745-7830` the blocking `adapt()` body in serve (`host_res[out] = kNotResident` at `:7761`, `:7785`, `:7819`; `srcp->prefetch(s.layer, s.out)` at `:7786`, `:7820`; `pending.emplace_back` `:7787`, `:7821`); `:11087-11088` its trigger (`(rounds + 1) % o.adapt_every == 0`, no `ajob`).
- `sycl/src/program/generate.cpp:7886` `struct ASwap`, `:7903-7910` the asynchronous tier (`--adapt-async 1`, only with the resident RAM mode, so not under `--stream-experts`), `:8107-8172` `adapt_tick` with `AState::CopyBack` at `:8125-8148` (`host_res[out] = kNotResident` `:8139`, `res_upload()` `:8142`).
- `sycl/src/program/generate.cpp:12100-12121` the `generate` command's own tier (`:12116`).
- `sycl/src/core/gguf_expert_source.cpp:76-113` `GgufExpertSource::blob` (returns the mirror pointer for a mirrored expert, `:78-81`, else a ring read from the SSD); `:169-182` `pinned`, `device_alias`; `sycl/include/strata/core/gguf_expert_source.hpp:72-75` `mirror_`, `mirror_off_`, `layer_first_`.
- `sycl/include/strata/core/expert_source.hpp:152, 155` `prefetch` / `release` are base no-ops (GgufExpertSource does not override them).
- `sycl/src/kernels/cuda/verify_kernels.dp.cpp:2104-2125` the `s_bad` path; `:2310-2321` `wait_flag_ge_or_kernel`; `sycl/src/core/verify.cpp:1375-1381` the skip-aware wait and copy; `:1936-1941` `no_host`.

**Current behaviour.**

```cpp
                case AState::CopyBack: {   // step 2: the evicted experts leave the GPU's plan
                    ...
                        host_res[out] = strata::core::kNotResident;   // the CPU computes it from the next window on
                        srcp->prefetch(w.layer, w.out);
                        aswaps[k++] = w;
                    }
                    aswaps.resize(k);
                    res_upload();
```
(`generate.cpp:8125-8142`; the blocking tier at `:7819-7821` does the same). The comment "the CPU computes it from the next window on" is the CUDA contract; on this card the CPU does not compute anything under `STRATA_VERIFY_NO_HOST=1`, and the device plan's only fallback for a non-resident expert is the mirror table, which never learns about `out`:

```cpp
                std::vector<unsigned long long> tab((size_t) (g.n_layers * g.n_expert), 0ull);
                for (int64_t l = 0; l < g.n_layers; ++l)
                    for (int64_t e = 0; e < g.n_expert; ++e)
                        if (gguf_src.pinned(l, e))
                            tab[(size_t) (l * g.n_expert + e)] = (unsigned long long) gguf_src.device_alias(l, e);
                mirror_table_d = sycl::malloc_device<unsigned long long>(tab.size(), dpct::get_in_order_queue());
                dpct::get_in_order_queue().memcpy(mirror_table_d, tab.data(), tab.size() * sizeof(unsigned long long)).wait();
```
(`:4726-4732`). The swapped-in expert `in` is fine: `blob(l, in)` returns its mirror pointer (`gguf_expert_source.cpp:78-81`), the copy lands in the slot, `host_res[in] = slot` and the plan takes `res[eid]` before the mirror.

**Change.** Two steps; ship step 1 now, step 2 behind a switch.
1. Refuse (default). In `generate.cpp` right after `drive.d.usage.assign(...)` (`:7625-7626`) and at the same place in the `generate` command's path (`:12100` region, find the matching `usage.assign`): if `mirror_table_d != nullptr` (experts are mirrored) and the tier is on, print `strata: the adaptive tier is off with --stream-experts and a host mirror (an evicted expert would have no mirror entry; STRATA_ADAPT_MIRROR=1 enables the mirror refresh)` and clear `drive.d.usage` (`o.adapt_every = 0` equivalently), unless `STRATA_ADAPT_MIRROR=1`. Side effect to state in the message: `--expert-profile-save` needs the tier (`:7629-7634`), so it is off too; the Tier 0 flag "`--expert-profile-save` from day one" (REVIEW section 2) depends on step 2.
2. Refresh (opt-in, `STRATA_ADAPT_MIRROR=1`). `in` and `out` are in the same layer, so their blobs have the same size (`lay.bytes[l]`), and under `STRATA_VERIFY_NO_HOST` every non-resident expert is mirrored (`:4735-4742` refuses otherwise), so `in` always has a mirror offset. The swap becomes an exchange inside the mirror:
   a. `GgufExpertSource::mirror_exchange(int64_t layer, int64_t in, int64_t out, uint8_t* slot_bytes_src, sycl::queue& q)`: D2H `q.memcpy(mirror_ + mirror_off_[in], xcache.device_slot(slot), bytes).wait()` (the slot still holds `out` at CopyBack time, before `a_copy_in` overwrites it; the waited copy is what makes this safe), then `mirror_off_[out] = mirror_off_[in]; mirror_off_[in] = -1;` (`layer_first_` can keep its value: `device_alias(layer, 0)` is used only as "does this layer have mirrored experts", `gguf_expert_source.hpp:51-53`). Also keep `where_` untouched (the ring is for SSD reads).
   b. Device table: two 8-byte `memcpy`s on the in-order queue, waited: `mirror_table_d[l * n_expert + out] = device_alias(l, out)`, `mirror_table_d[l * n_expert + in] = 0`.
   c. Order inside `CopyBack` (`:8130-8142`) and in the blocking tier (`:7819-7821`, `:12116`): (i) D2H copy of `out`'s slot into `in`'s mirror bytes, waited; (ii) the two table writes, waited; (iii) `host_res[out] = kNotResident`; (iv) `res_upload()`. Only then may `a_copy_in` / `adapt_copy_h2d` overwrite the slot with `in` (the asynchronous tier already posts `a_copy_in` after `res_upload`, `:8143`; the blocking tier issues its H2D copies in the same loop as the eviction (`:7773-7822`), so there the D2H and table writes must be hoisted into a first pass over `swaps` before any `adapt_copy_h2d`).
   d. Hazard to state in code: `in`'s mirror bytes are overwritten while a window in flight may still be reading `in` from the mirror. The blocking tier runs beside `ver.commit` and the draft (`:11084-11088`), after the window's verify has completed (`cs_->wait()` in the window), and the next window starts after `adapt_thr.join()`; so between the end of verify and the next window no kernel reads experts. For `--pipeline-windows` (windows in flight, `adapt_fence` at `:7757-7764`) the fence must also cover the D2H copy; simplest: refuse `STRATA_ADAPT_MIRROR=1` with `--pipeline-windows > 0` and with `--adapt-async 1` (that one is already off under `--stream-experts`: `src.complement_ready()` is false, `:7906`).
   e. `srcp->prefetch(w.layer, w.out)` and `srcp->release(w.layer, w.in)` stay (no-ops on GgufExpertSource).
3. `stage_exchange` / `commit_exchanges` (`:8133`, `:7691`) are the resident-RAM source's and are skipped when `w.exchange == false` (`:7967`: `src.has_resident(...)` is false without the resident RAM copy); no change.

**Must not change.** The three standard runs with the tier refused (step 1) are identical to today's with `--adapt-every 0`. With step 2 on, every window's device plan must find each routed expert either in `res` or in the mirror table (`stats[12]` of D7, `bad plans`, stays 0). The `REFUSED ... neither in VRAM nor mirrored` start-up check (`:4735-4742`) is unchanged. No change to `resident_plan_kernel`.

**Tests.**
- Existing: none covers this. `resident_plan_parity`'s mirror mode from D7 covers the kernel side.
- Engine check (the real test): IQ2_XS, `--expert-cache 8000` (forces about 1,879 experts out so swaps are frequent), `--adapt-every 1 --adapt-swaps 16`, 256 greedy tokens on the 2,184-token prompt, `STRATA_PLAN_STATS=1`: today expect `bad plans > 0` and tokens that differ from the `--adapt-every 0` run (or a `never rang` / timeout); with step 1, the tier prints its refusal and tokens equal the `--adapt-every 0` run; with step 2 (`STRATA_ADAPT_MIRROR=1`), `bad plans == 0`, `experts swapped into the VRAM tier` non-zero in the report (`:12397`), tokens identical to `--adapt-every 0` (routing does not depend on where an expert lives).
- Unit test for `mirror_exchange` in a new `sycl/src/core/gguf_mirror_test.cpp` (plain host code, no model: a fake 2-layer layout is not available without a pack, so mark it as needing the IQ2_XS shard; otherwise test only the offset bookkeeping with a stubbed read).

**Bench.** Not a speed item. Record on the forced-miss run: tok/s with the tier refused vs `STRATA_ADAPT_MIRROR=1` with `--adapt-every 4` (the default), 256 greedy tokens, 2 interleaved pairs; the mirror refresh adds one 1.4 MB D2H copy per swap, 96 swaps per round at most, between windows.

**Expected gain, risk, effort.** Correctness. Step 1: risk none, 0.25 day. Step 2: medium risk (ordering of the D2H copy against the slot overwrite and against windows in flight; a mistake shows as a wrong token or a `bad plans` count), 1 day including the engine check. The forward value of step 2 is that the resident set follows the traffic (`--expert-profile-save`, REVIEW Tier 0).

**Depends on / conflicts with.** D7 (the `bad plans` counter is the test oracle). T2a's slot ring must also never be chosen for an expert the tier just swapped in (it checks `res` first, so it is safe). No upstream overlap: upstream's tier assumes a CPU fallback that this platform does not have.

---

### D9. Mirror cap from the cgroup-aware `host_available_memory`; `posix_fadvise(DONTNEED)` after each fill batch; the O(N squared) miss-list build

**Goal and measured motivation.** The mirror's default cap is `MemAvailable - 4 GiB` read from `/proc/meminfo` once at start (`generate.cpp:4704-4712`): inside a VM with a 48 GB (later 64 GB, owner-gated) budget shared with fast-llm's co-tenant, a cgroup memory limit is invisible to it, and `MemAvailable` counts reclaimable page cache, which the fill itself inflates. The port already has `strata::core::detail::host_available_memory` (`sycl/src/core/expert_source.cpp:241-...`, declared `sycl/include/strata/core/expert_source.hpp:82-84`, `HostMemory{available, cgroup_limit, commit}` at `:72-76`) that walks the cgroup v2 `memory.max` ancestors and v1 `memory.limit_in_bytes`; it is used for the arena, not the mirror. The fill (`generate.cpp:4529-4575`) and the mirror (`gguf_expert_source.cpp:140-150`) read 24.6 GiB plus 8.4 GiB of GGUF through the page cache (IQ2_XS, INTEL 499-500, load 41 s of which the mirror is about 9 s, INTEL 609-611); on a 48 GB VM that evicts the co-tenant's pages. The miss-list build (`:4699-4703`) does a linear `std::find` over `miss` for every (layer, expert) pair not in the profile: for IQ3_S (24,576 pairs against about 13,000 misses) that is on the order of 3e8 pair comparisons, REVIEW D9 puts it at 0.5 to 1 s per start (E; not timed on the card).

**Where.**
- `sycl/src/program/generate.cpp:4695-4703` the `miss` vector build; `:4704-4712` the `MemAvailable` read and `cap`; `:1112` a second `MemAvailable` read (`m.avail_mib`, the `rss_probe` report, informational only).
- `sycl/src/core/gguf_expert_source.cpp:115-167` `GgufExpertSource::mirror` (`malloc_host` `:132`, reader threads `:140-150`, `read_into` `:147`); `:184-203` `read_into` (three `pread`s per expert, `fds_` and `lay.gguf_off` give the file and offset).
- `sycl/src/program/generate.cpp:4529-4575` the pipelined fill (`kBatch = 64` at `:4544`, two pinned halves `:4547`, reader threads `:4553-4568`, copies `:4570-4574`).
- `sycl/src/core/expert_source.cpp:2097-2098` an existing `posix_fadvise(fd_, off, bytes, POSIX_FADV_DONTNEED)` in `FileExpertSource` (the pattern to copy); upstream `src/core/expert_source.cpp:572` (`POSIX_FADV_RANDOM`) and `:2359` (`DONTNEED` whole file).
- `sycl/src/core/expert_source.cpp:241-300` `host_available_memory`.

**Current behaviour.**

```cpp
        for (int64_t l = 0; l < mirror_end; ++l)                     // pairs the profile does not list at all
            for (int64_t e = 0; e < g.n_expert; ++e)
                if (xcache.slot_of(l, e) == strata::core::kNotResident &&
                    std::find(miss.begin(), miss.end(), std::pair<int64_t, int64_t>{l, e}) == miss.end())
                    miss.push_back({l, e});
        uint64_t avail = 0;
        if (FILE* f = std::fopen("/proc/meminfo", "r")) {
            char key[64]; unsigned long long kb = 0;
            while (std::fscanf(f, "%63s %llu kB", key, &kb) == 2)
                if (std::strcmp(key, "MemAvailable:") == 0) { avail = kb << 10; break; }
            std::fclose(f);
        }
        const char* mv = std::getenv("STRATA_MIRROR_MIB");
        const uint64_t cap = mv ? (uint64_t) std::atoll(mv) << 20 : (avail > (4ull << 30) ? avail - (4ull << 30) : 0);
```
(`generate.cpp:4699-4712`). The mirror's readers (`gguf_expert_source.cpp:143-150`) and the fill's readers (`generate.cpp:4560-4566`) call `read_into`, which `pread`s each slice and never advises the kernel.

**Change.**
1. Miss list: before the first loop, `std::vector<uint8_t> listed((size_t) (g.n_layers * g.n_expert), 0);` mark `listed[l * n_expert + e] = 1` as each profile pair is pushed; the second loop tests `!listed[...]` instead of `std::find`. Same order of `miss` (profile order first, then layer-major), so the same experts are mirrored under a cap and the same `tab`.
2. Cap: replace `:4704-4710` with `strata::core::detail::HostMemory hm; uint64_t avail = 0; if (strata::core::detail::host_available_memory(hm)) avail = std::min(hm.available, hm.cgroup_limit);` (fall back to the `/proc/meminfo` read when it returns false, which it does when the cgroup tree is unreadable). Keep `STRATA_MIRROR_MIB` as the override and the 4 GiB headroom; add `STRATA_MIRROR_HEADROOM_MIB` (default 4096) so the co-tenant's share can be set without computing the cap by hand. Print the chosen cap and its source (`cgroup limit`, `MemAvailable`, `STRATA_MIRROR_MIB`) in the existing mirror log line (`:4721-4724`). Do the same at `:1112` only if the `rss_probe` output is used for decisions (it is informational: leave it).
3. `posix_fadvise(DONTNEED)`: add `bool drop_cache` to `GgufExpertSource` (set from `STRATA_FILL_DROP_CACHE=1`, default off, see risk) and in `read_into` (`:192-201`) after each slice's `pread` loop: `(void) posix_fadvise(fd, (off_t) src, (off_t) per[r], POSIX_FADV_DONTNEED);` (the FileExpertSource form at `sycl/src/core/expert_source.cpp:2097`). Both the mirror threads (`:147`) and the fill threads (`generate.cpp:4565`) go through `read_into`, so one change covers "after each fill batch" at a finer grain; if the per-slice calls measure as too many syscalls (about 3 per expert, 75,000 per start), batch them per fill batch in `generate.cpp:4569` by advising the byte range of each of the batch's three tensors (`lay.gguf_off[3 * l + r] + per[r] * e` for the batch's min and max `e`, the fills are sorted by layer then expert, `:4539-4541`). Note `fds_` are opened `O_RDONLY | O_CLOEXEC` (`:68`); `O_DIRECT` is not an option because blob offsets are not sector aligned (expert sizes like 1.44 MB are 256-byte rounded at most, `:123`).
4. Also apply the fadvise to the SSD ring path `blob()` (`:101-110`)? No: those reads are rare after start and the cache helps a repeat.

**Must not change.** Which experts are mirrored and the order of `miss` (so `tab` and the resident placement are the same; outputs identical). `STRATA_MIRROR_MIB` semantics. The `REFUSED` check (`:4735-4742`). The fill's placement (`xcache.admit` order, `:4536-4541`).

**Tests.**
- Existing: none for this code (it is start-up host code). `STRATA_VERIFY_ALL_SLOTS=1` (`generate.cpp:4586-4627`) after the fill still reports 0 differing slots with `STRATA_FILL_DROP_CACHE=1` (the advice must not alter what was read).
- Add a host unit test `sycl/src/core/mirror_misslist_test.cpp` (plain C++, no device): the old `std::find` build and the new marked build over a synthetic profile and residency give identical vectors; register with `add_test` under `STRATA_SYCL_PARITY`.
- `host_available_memory` already has upstream's parameterised form (`meminfo`, `self_cgroup`, `cgroup_root` for tests, `expert_source.hpp:78-84`); a test that writes a fake `memory.max` below `MemAvailable` and asserts the mirror cap follows it belongs in the same file.
- Engine: start IQ2_XS with and without the change; the log line `N of M experts missing from VRAM mirrored` must show the same N and GiB.

**Bench.** Start time: the engine's launch to first token (INTEL 609-611: IQ2_XS 41 s), and the `mirroring ... %.1f s` line (`:4721-4724`), 3 starts each, cold page cache (`echo 3 > /proc/sys/vm/drop_caches` needs root on the VM, otherwise warm) with and without `STRATA_FILL_DROP_CACHE=1`; `free -g` / the co-tenant's RSS before and after a start as the memory metric. For the miss list, time the block `:4695-4703` with `Clock::now()` in a debug print (REVIEW: 0.5 to 1 s expected for IQ3_S; IQ2_XS has 6,247 misses so about a quarter of that).

**Expected gain, risk, effort.** 0.5 to 1 s per start (E) on the miss list; a mirror cap that respects a cgroup limit (no OOM-kill of the engine or the co-tenant); the fadvise keeps 30 to 40 GB of GGUF out of the page cache so fast-llm's pages stay. Risk low; the trade-off to decide with the owner: `DONTNEED` makes every restart a cold read (the fill measured 18.8 s cold vs faster warm, INTEL 609-611), which is why it is opt-in here. 1.5 person-days.

**Depends on / conflicts with.** None. Tier 0's `STRATA_MIRROR_MIB=<explicit>` becomes unnecessary once the cgroup cap works. If `STRATA_LEND_MIRROR` (the author's branch, REVIEW section 2) is ported, its 1.9 to 2.3 GiB must come out of the same cap.

---

### T2a. Tier 2: prefetch of predicted mirror misses into a per-layer slot ring, driven by lookahead routing

**Goal and measured motivation.** A mirrored expert is read in-kernel from host USM, 4 to 8 bytes a lane in 64-byte lines (`verify_kernels.dp.cpp:2099-2104` picks the address, `iq_kernels.dp.cpp` lane dots read it). On Gen5 x8 IQ2_XS already decodes at the all-resident Coder's speed (REVIEW section 0, author's table), so the lever is IQ3_S (about 13,000 of 24,576 experts mirrored at 32 GB; 30 to 41 tok/s on prose against a 78 ceiling, INTEL 731-760 and REVIEW section 4) and any later model that does not fit. The forced-miss run on the Gen3 x8 card put 1,879 misses (3.6 GiB) at about 6% of decode (40.9 vs 43.3, INTEL 435-437). The prefetch replaces the per-lane PCIe reads of a predicted miss with one 1.4 to 2 MB streaming copy issued one layer ahead, and keeps the in-kernel read as the fallback for mispredictions. REVIEW: 6 to 10 days, high risk; upstream's Foresight (`STRATA_FS_SLOTS`) measured neutral on CUDA where it only displaced CPU compute; here there is no CPU compute to displace.

**Where.**
- `sycl/src/core/verify.cpp:1240-1267` the route step inside the captured window (`moe_route(wt, g, l, K, mb, mixed_ + t * N, cs, err, nullptr)` per token at `:1252-1256`; `resident_plan` calls `:1259-1267`); `:1270-1283` the doorbell publish that follows; `:1340-1400` `post()` (plan layout `:1343-1352`, `grouped()` launch `:1358-1366`, the pcie_mode 2 staging `:1387-1391` with `fetch_blobs` and `rebase_ptrs`).
- `sycl/src/core/layer.cpp:390-392` `moe_route(const WeightTable&, const ModelGeometry&, int64_t layer, int64_t k, const MoEBuffers&, const float* x, void* stream, std::string&, const Doorbell*)` (the router of `layer` on `x`: calling it with `layer + 1` and layer `l`'s input is the prediction).
- `sycl/src/kernels/cuda/verify_kernels.dp.cpp:847-860` `fetch_blobs_kernel` (grid-stride `uint4` copy from an array of source addresses), `:862-869` `rebase_ptrs_kernel`, `:1007-1047` their launchers (`48 * 8` work-groups of 256; `blob_bytes % 16 == 0` required); `:2073-2221` `resident_plan_kernel` (`maddr` at `:2099-2104`, the group pointer at `:2186-2187`).
- `sycl/include/strata/core/verify.hpp:446-447` `kStagingBlobs = 16`, `kPcieGroupRows = 4`; `:254` `set_pcie_mode`.
- `sycl/src/core/verify.cpp:621-626` the shared-expert side queue and its events (`sh_cs_`, `ev_fork_`, `ev_join_`); `:1212-1223` the fork inside the window (`sh_fork`, `sh_fork_late`, `sync_barrier(ev_fork_, cs)` then `sh_cs_->ext_oneapi_submit_barrier`), `:1284-1287` the late fork: the pattern for a second in-order queue inside the graph.
- `sycl/include/strata/core/expert_source.hpp:197-211` upstream's host `RouterLookahead` (`submit(layer, const float* x, n_tok, host_res)` takes host floats; it predicts layer + 1 from layer l's MoE input on a CPU thread); `sycl/src/program/generate.cpp:4954-4973` wires it only for the arena source (`srcp == &src && src.warms()`), so it is inactive under `--stream-experts`; `include/strata/core/foresight_swap.hpp:1-20` (upstream, not in the port: `grep -c ForesightSwap sycl/src/program/generate.cpp` is 0) states "Not with STRATA_VERIFY_DEVICE_PLAN".
- `sycl/src/program/generate.cpp:4725-4732` the mirror table; `:6850` `resident_plan_set_mirror`.
- `include/strata/kernels/verify_kernels.hpp:21` `kVerifyMaxT = 8`; `verify.cpp:1934` `G` (1 or 2 groups per layer).

**Current behaviour.** The plan kernel points a non-resident expert straight at its pinned mirror address and the expert kernels read it over PCIe:

```cpp
        ptr[grp_idx] = slot >= 0 ? (unsigned long long) (cache_base + (slot_off ? (size_t) slot_off[slot] : (size_t) slot * (size_t) blob))
                                 : maddr;
```
(`verify_kernels.dp.cpp:2186-2187`). A staging copy kernel exists for the host-planned PCIe share and is unused under the device plan:

```cpp
            if (sink_.pcie_mode == 2) {                            // stage it with a copy kernel, then point at staging
                const int64_t per = G == 2 ? kStagingBlobs / 2 : kStagingBlobs;
                uint8_t* stage = staging_ + (size_t) (grp * per) * lay.max_blob;
                fetch_blobs(p_ptr2, p_counts + 2, stage, (int64_t) lay.blob_bytes(l), (int) per, cs);
                rebase_ptrs((unsigned long long*) p_ptr2, p_counts + 2, stage, (int64_t) lay.blob_bytes(l), cs);
            }
```
(`verify.cpp:1387-1391`). Nothing predicts the next layer on the device; the host lookahead needs host rows that `STRATA_VERIFY_NO_HOST` never publishes.

**Change.** Everything runs inside the captured graph (no host in the loop, no graph update, no host-to-device visibility during the window: the three constraints INTEL records). Design, in enough detail to start:
1. Memory. A ring of `S` slots per layer (`STRATA_PF_SLOTS`, default 8; `S * n_layers * max_blob`: 8 x 48 x 1.44 MB = 553 MB for IQ2_XS, 8 x 48 x 1.97 MB = 756 MB for IQ3_S), allocated with `strata::malloc_device_guarded` (INTEL 740-744: every allocation of 32 MiB or more goes through it) before capture, outside the expert cache. Per layer a device table `ring_eid[S]` (int32, the expert each slot holds, -1 empty) and `ring_base` (the slot addresses, fixed). Scratch for the prediction: `pred_logits_` (`max_t_ * NE` floats), `pred_ids_` (`max_t_ * K` int32), `pred_w_` (unused weights).
2. Predict. In the route step of layer `l` (after `:1256`, before `resident_plan`), for `l + 1 < le_`: call `moe_route(wt, g, l + 1, K, pred_mb, mixed_ + t * N, pf_cs, err, nullptr)` per token of the group with `pred_mb` pointing at the scratch (the same per-token loop as `:1252-1256`; D3's fused router, if done first, gives a one-call form). This runs layer `l + 1`'s `ffn_gate_inp` on layer `l`'s MoE input, which is what upstream's `RouterLookahead` does (`expert_source.hpp:195-196`: "a prediction only warms pages: it never changes which experts are computed or how"). Prediction quality on this model is not verified; D7's stats plus a counter of ring hits (below) measure it.
3. Prefetch kernel `mirror_prefetch_kernel(pred_ids, n * K, res_{l+1}, mir_{l+1}, ring_eid_{l+1}, ring_base_{l+1}, S, blob_bytes / 16, cursor_{l+1})`, one work-group of 128 threads then a grid-stride copy phase, or two kernels: (a) select: dedupe `pred_ids` as `resident_plan_kernel` does (`first_j`, `:2128-2140`), keep entries with `res[eid] < 0 && mir[eid] != 0` that are not already in `ring_eid`, take at most `S` of them in routing order, assign slots round-robin from a per-layer `cursor` word, write `ring_eid[slot] = eid` for the new ones and `-1` for slots being refilled; (b) copy: `fetch_blobs_kernel`'s loop (`:847-860`) over the selected `(mir[eid], ring_base[slot])` pairs, `uint4` loads from host USM, 48 x 8 x 256 threads. The copy reads host memory with 16-byte coalesced loads instead of the expert kernels' 4 to 8 byte lane loads; whether that reaches the link's rate from a kernel is not verified (measure `fetch_blobs` alone first: time it on 16 blobs with unitrace; REVIEW's 26.5 GB/s is the DMA rate on Gen5 x8).
4. Plan. In `resident_plan_kernel` (`:2099-2104`) add `ring_eid` / `ring_base` arguments (per layer slice, passed by the launcher like `mir`): `if (slot < 0 && maddr != 0) { for (int s2 = 0; s2 < S; ++s2) if (ring_eid[s2] == eid) { maddr = ring_base[s2]; ++hit; } }` (S <= 16 compares per thread). The group pointer then lands in VRAM for a hit and in the mirror for a miss: the expert kernels are unchanged. Count hits into D7's stats (`[13]` ring hits, `[14]` ring copies).
5. Ordering and reuse. Both kernels of layer `l` must complete before layer `l + 1`'s plan runs, and the copy into a slot must not overlap a window still reading that slot. Simplest correct form: predict and copy on the main queue `cs` right after layer `l`'s routing, before its expert kernels (in-order queue: everything is serialised, the copy's time is on the critical path, about 130 x 1.44 MB / 26.5 GB/s = 7 ms a round at the DMA rate, spread over 48 layers). Better: on a side in-order queue like the shared expert's (`sh_cs_`, fork `:1212-1223` or `:1284-1287`, join before layer `l + 1`'s plan), which hides the copy behind layer `l`'s expert kernels at the cost of 3 barrier nodes a layer (144 a round, the same price `STRATA_SH_STREAM` pays, which Tier 0 is to A/B). Slot reuse within a window is safe because a layer's slots are read only by that layer's expert kernels, which precede the next window's prefetch of the same layer on the same queue; with `G == 2` (split window) both groups of layer `l` share the layer's ring, so group 1's prefetch must not evict what group 0's plan just took: run the prefetch once per layer (group 0 only) with both groups' predicted ids, or `S` per group.
6. Switch: `STRATA_PF_SLOTS=<n>` (default 0 = off, nothing allocated, kernels not captured). Also `STRATA_PF_QUEUE=side|main`. Report: per request `ring hits / mirror groups` (D7).
7. Where the code goes: the two kernels and their launchers in `sycl/src/kernels/cuda/verify_kernels.dp.cpp` next to `fetch_blobs` (`:1007`), declarations in `sycl/include/strata/kernels/resident_plan_mirror.hpp`; the capture-time calls in `sycl/src/core/verify.cpp`'s route step; allocation and the table pointers in `sycl/src/program/generate.cpp` next to `mirror_table_d` (`:4725-4732`) and `resident_plan_set_mirror` (`:6850`). Upstream pattern: `foresight_swap.hpp`'s slot and stamp discipline (`Slot::last_ref`, `depth`) is the host-side version of step 5; the port does not take that code.

Open questions (flag before starting): (a) prediction accuracy of `router(l+1)` on `x(l)` for this model (measure with D7: ring hits over predicted copies; if under 50% the bytes wasted on mispredictions compete with the real reads on the same link); (b) copy rate of a kernel reading host USM (step 3) versus a copy-engine DMA (not usable here: a DMA node's addresses are fixed at capture); (c) the side-queue fork's node cost (Tier 0's `STRATA_SH_STREAM` A/B answers it); (d) IQ3_S needs the owner-gated 64 GB VM to exist first (REVIEW companion); on IQ2_XS at Gen5 x16 the expected gain is small and may not clear the noise; (e) with D8 step 2 the tier changes `res` between windows: the ring check runs after `res`, so a swapped-in expert is taken from its slot, correct; (f) `blob_bytes % 16 == 0` (`:1009`) holds for the 256-rounded mirror offsets (`gguf_expert_source.cpp:123`) but the GGUF blob sizes themselves must be checked per model.

**Must not change.** The expert kernels' inputs are the same bytes whether read from a ring slot or the mirror: output identity holds by construction, and `STRATA_PF_SLOTS=0` leaves the graph as today. `resident_plan_parity` with `ring_eid == nullptr` unchanged. No new public exposure, no change to `--expert-cache` sizing (the ring is extra VRAM: say so in the start-up log and subtract it from the `MiB of VRAM free` line, `--vram-reserve-mib` must cover it or the cache auto-sizing must know about it, `generate.cpp:4188-4260` region, not read in detail here).

**Tests.**
- Extend `resident_plan_parity` (D7's mirror mode) with a ring table: for an expert both mirrored and in the ring the plan pointer must equal the ring address; otherwise the mirror address; `memcmp` of the plan otherwise unchanged.
- New `mirror_prefetch_parity` (synthetic, no model): random `pred_ids` with duplicates, a residency table, a mirror table over host USM blobs of random bytes, S slots; after select + copy, every `ring_eid[s] >= 0` slot's bytes equal the source blob; no expert appears in two slots; at most S copies; a second call with overlapping predictions does not re-copy an expert already in the ring.
- Engine: the three standard runs with `STRATA_PF_SLOTS=8`, tokens identical; `STRATA_VERIFY_ALL_SLOTS=1` once (the guarded allocation of the ring).

**Bench.** Forced-miss run `--expert-cache 8000` on IQ2_XS (so the Coder-class link numbers have misses to hide), 256 greedy tokens on both standard prompts, `STRATA_PF_SLOTS=0` vs `8` vs `16`, 4 interleaved pairs, medians; D7's `ring hits` per layer-window; unitrace device time of `mirror_prefetch`, `native_gu_port`, `native_down_port` per round; the graph's node count (REVIEW section 1: 2,000 to 2,600 a window) with the side queue on and off. Then IQ3_S when the VM has the RAM.

**Expected gain, risk, effort.** IQ2_XS on Gen5 x16: small, possibly nothing measurable (REVIEW section 8); IQ3_S: the lever between 30 to 41 and the 78 ceiling, size unknown until the prediction hit rate and the kernel copy rate are measured (E). Risk high: graph ordering between two queues, slot reuse, VRAM taken from the cache, wasted link bandwidth on mispredictions; the failure mode of an ordering bug is a wrong token (caught by output identity) or a GT reset from a spin (none added here: no flags are waited on). 6 to 10 person-days (REVIEW), of which 1 day is the measurement gate (steps 2 to 4 with the copy on the main queue) before the side queue.

**Depends on / conflicts with.** D7 (counters) and D8 (the tier must not invalidate ring contents; with step 1 refused there is no interaction). D3 would fold the prediction into the fused router kernel. Conflicts with `STRATA_SH_STREAM=0` only in that both change the queue topology (measure together). Upstream `STRATA_FS_SLOTS` is a different mechanism (host dispatch, copy stream, not with the device plan); if upstream later adds a device-plan prefetch, align the switch names.

---

### T2b. SIMD16 build of the E=4 multi dots and the IQ1_M path after an IGC spill check

**Goal and measured motivation.** The engine is built with `-fsycl-default-sub-group-size=32` (`sycl/CMakeLists.txt:60-61`) and every expert launch asks for `[[sycl::reqd_sub_group_size(32)]]` (`iq_kernels.dp.cpp:4117, 4141`). dpct flagged the multi kernels for register pressure (DPCT1110 at `iq_kernels.dp.cpp:1248` `row_dot_40_sub8`, `:1335` `row_dot_20_sub16`, `:1393` `row_dot_80_sub16`, `:1863` `native_gu_multi_kernel`, `:2030` `native_down_multi_kernel`); the E=4 lane dot keeps `MultiW` (11 dwords, 15 after D4), `u[8]`, `s[4]`, four 64-bit activation pointers and loop state live, about 40 to 50 dwords a lane (E). On Xe2 a thread has 128 GRF of 64 bytes in the default mode (not verified for bmg-g31; the dump header prints the GRF count), which is 64 dwords a lane at SIMD32 and 128 at SIMD16: a spill at SIMD32 is plausible and would make the ALU-bound dots (77% XVE active, INTEL 295-298) slower than they need to be. The earlier SIMD16 experiment covered only the dense mmvq kernels (`STRATA_MMVQ_SG`, INTEL 441-445: up to 1.45x on IQ4_XS alone, no engine-level win); the expert lane kernels were never built at 16. REVIEW: 0 to 10% of expert-dot time, low risk to measure, 2.5 days.

**Where.**
- `sycl/CMakeLists.txt:39-65` the SYCL flags (`-fsycl-targets=spir64_gen` AOT, `-Xsycl-target-backend` for ocloc at `:43-54`, `-fsycl-default-sub-group-size=32` at `:61`, `-fsycl-device-code-split=per_kernel` at `:64-65`).
- `sycl/src/kernels/cuda/iq_kernels.dp.cpp:795-806` `kExpertLanes`, `lanes_sum<LANES>` (uses `dpct::experimental::permute_sub_group_by_xor(0xffffffffu, sg, v, o)`); `:808-816` `row_dot_lanes`; `:1753-1781` `row_dot_multi<TY, LANES, E>`; `:1812-1853` `native_gu_kernel`; `:1983-2024` `native_down_kernel`; `:4107-4152` `launch_gu_port`, `launch_gu_lanes`, `launch_down_port`, `launch_down_lanes`.
- `sycl/src/kernels/cuda/iq_kernels.dp.cpp:313-347` `vec_dot_iq1_m_q8_1` (the IQ1_M per-entry path; after D4 also `Multi<29>`).
- `sycl/include/dpct/util.hpp:671-681` `permute_sub_group_by_xor` (`logical_sub_group_size = 32`: `start_index = id / 32 * 32`, `target = (id % 32) ^ mask`; inside a 16-lane sub-group `id < 16`, so masks below 16 stay in range).
- `sycl/src/kernels/cuda/native_mmvq.dp.cpp:154-159` `sg_sum<SG>` (plain `sycl::permute_group_by_xor`), `:164-170` `mmvq_sg()` / `mmvq_sg16()` (the existing `STRATA_MMVQ_SG` switch); `:2031-2046` the two `reqd_sub_group_size(16)` launches of `native_mmvq_multi_kernel<..., 16>` (the pattern to copy).
- `sycl/src/kernels/mmvq_sg_bench.cpp:1-3` (`STRATA_MMVQ_SG=16` and `=32`, us per call and a checksum).
- `sycl/tools/build.sh` (not read here; the flag plumbing for a dump-enabled build goes there).

**Current behaviour.**

```cpp
template <int TG, int LN>
void launch_gu_port(unsigned groups, dpct::queue_ptr s, ...) {
    constexpr int ROWS = 256 / LN;
    const unsigned gx = (unsigned) ((2 * L.n_ff + ROWS - 1) / ROWS);
    auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};
    s->parallel_for<dpct_kernel_name<class native_gu_port, dpct_kernel_scalar<TG>, dpct_kernel_scalar<LN>>>(
        sycl::nd_range<3>(sycl::range<3>(1, groups, (size_t) gx * 256), sycl::range<3>(1, 1, 256)), exp_props,
        [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
            native_gu_kernel<TG, LN>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
        });
}
```
(`iq_kernels.dp.cpp:4107-4120`). Inside, a row is `LN = 8` consecutive threads (`rib = local_id / LN`, `sub = local_id % LN`, `:1840-1841`) and the reduction is an xor tree within those 8 lanes:

```cpp
template <int LANES>
__dpct_inline__ float lanes_sum(float v) {
#pragma unroll
    for (int o = LANES / 2; o > 0; o >>= 1)
        v += dpct::experimental::permute_sub_group_by_xor(0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), v, o);
    return v;
}
```
(`:800-806`). Nothing in the lane kernels depends on the sub-group being 32 wide as long as a row's `LN` lanes lie inside one sub-group (true for `LN <= 16` at SIMD16; `LN = 32` needs SIMD32). The 32-lane paths (`row_dot_multi<TY, NC, ...>` at `:1216-1241` with `k += 32` and `warp_sum`, `row_dot_80_sub16`, the `STRATA_EXPERT_SPLIT` kernels) are not candidates.

**Change.**
1. Spill report first (half a day). Two routes, use both: (a) IGC shader dumps: run the AOT build step (or, for a JIT build, the engine itself) with `IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=<dir>`; the per-kernel `*.asm` files carry a header with the kernel name, the GRF count, `//.spill size` and `//.private memory size` (the vISA finalizer's header); for an AOT build the env must reach `ocloc` during the CMake link step (`CMakeLists.txt:43-54`), so wrap that step or build once in JIT mode (`-fsycl-targets=spir64`) for the dump. Kernel names are `dpct_kernel_name<class native_gu_port, dpct_kernel_scalar<TG>, dpct_kernel_scalar<LN>>` (`:4115`) and `native_down_port` (`:4137`), one module each thanks to `per_kernel` splitting (`:64-65`). (b) `zeKernelGetProperties`: `ze_kernel_properties_t::spillMemSize` and `privateMemSize` per kernel; a small probe in `sycl/probe/` that loads the engine's kernel bundle and reads them through `sycl::get_native<sycl::backend::ext_oneapi_level_zero>(kernel)`; whether the fields are populated for AOT `spir64_gen` binaries on this driver is not verified. Record, for `native_gu_port<17|29|18|21|22, 8>` and `native_down_port<20|42, 8>`: GRF count, spill bytes, private bytes. If spills are zero at SIMD32, stop after step 2's one measurement (the item then answers "no").
2. SIMD16 variant: add `int SG` to `launch_gu_port<TG, LN, SG>` / `launch_down_port<TD, LN, SG>` with `[[sycl::reqd_sub_group_size(SG)]]`, selected by `STRATA_EXPERT_SG=16|32` (default 32; mirrors `mmvq_sg()` at `native_mmvq.dp.cpp:164-167`), instantiated for `SG = 16` only when `LN <= 16`. Replace `lanes_sum`'s dpct masked permute with `sycl::permute_group_by_xor(sg, v, o)` (the form `sg_sum` uses, `native_mmvq.dp.cpp:154-159`): same tree, no 32-wide mask assumption. The kernel bodies are unchanged; the work-group stays 256 threads (16 sub-groups of 16 instead of 8 of 32). Also try the IGC large-GRF mode as the other arm: the Level Zero build option `-ze-opt-large-register-file` through `-Xsycl-target-backend=spir64_gen "-options -ze-opt-large-register-file"` (whether bmg-g31 honours it is not verified; the dump header shows the GRF count either way).
3. IQ1_M path: nothing separate to write; `vec_dot_iq1_m_q8_1` and (after D4) `Multi<29>` run inside the same lane kernels and inherit the switch. Record its spill line separately because its `sumf[2]` floats and `delta` add live state.
4. Per-format choice if SIMD16 wins on some formats and loses on others (the dense kernels' pattern, INTEL 442-444): `STRATA_EXPERT_SG` takes a list (`16:17,29` = SIMD16 for types 17 and 29 only), decided in `launch_gu_lanes` from `TG`.
5. Where: all in `iq_kernels.dp.cpp` (launchers and `lanes_sum`); the probe under `sycl/probe/`; the dump recipe in `docs/INTEL.md` under profiling (INTEL 268-271) once it works. No `fixups.py` entry (the launchers are the port's own code, `:4098-4152`).

**Must not change.** Bitwise output: each lane's k sequence (`k = sub; k += LANES`) and the 8-lane xor tree are the same at SIMD16 and SIMD32, so `iq_multi_parity` and `native_grouped_parity` must pass at both settings and the three standard runs must be identical. Default stays SIMD32 until the engine-level A/B beats it beyond run-to-run noise (the dense experiment's standard, INTEL 444-445). `STRATA_GU_LANES=32` with `STRATA_EXPERT_SG=16` must be refused (a 32-lane row cannot sit in a 16-lane sub-group).

**Tests.**
- Existing: `iq_multi_parity`, `native_grouped_parity`, `native_expert_parity` run twice, `STRATA_EXPERT_SG=16` and unset, outputs `memcmp` equal (add the double run to the SYCL copy of `iq_multi_parity.cpp`, or run `ctest` twice with the env).
- New: a check in `launch_gu_lanes` that `SG >= LN` (else fall back to 32 with a one-line note), exercised by `STRATA_GU_LANES=32 STRATA_EXPERT_SG=16`.
- Engine: three standard runs, tokens identical.

**Bench.** `NATIVE_BENCH=1 native_expert_parity <shard> <layer>` and `iq_multi_parity --bench` at `STRATA_EXPERT_SG=16` vs 32 (warm clocks); unitrace device time per call of `native_gu_port<TG, 8>` and `native_down_port<TD, 8>` for TG in {17, 29, 21, 18, 22}; the spill table from step 1 beside each. Engine: tok/s on the 19/20-token and 2,184-token prompts, 256 greedy tokens, 4 interleaved pairs, medians, Coder (IQ1_M) and IQ2_XS; also the forced-miss run, because fewer live registers change how many PCIe loads are in flight per Xe core.

**Expected gain, risk, effort.** 0 to 10% of expert-dot time (REVIEW), 0 to 0.5 ms a round; or a clean "no spills, no change" answer. Risk low (a launch attribute and a shuffle spelling; the failure mode is a slower kernel, caught by the bench, or a wrong `permute` width, caught bitwise). 1.5 to 2.5 person-days including the dump recipe.

**Depends on / conflicts with.** Do after D4 so the measured kernels are the final ones (D4 adds `delta[4]` to `MultiW`; the spill check must see it). Independent of the mirror items. Shares the `lanes_sum` change with any later ESIMD dot (REVIEW section 4). No upstream overlap (upstream has no sub-group size notion; `STRATA_MMVQ_SG` is the port's precedent).

# Part C. Prompt path


Scope: items P1 to P6 of `docs/research/strata-b70-engine-plan-20261009-025812.md` section 3 and the Tier 2 row "INT8
expert weights ... oneDNN" (T2c here). Read-only review of `/home/user/strata` at `fb58e0d` (engine 0.1.41); every line
number below was read in that checkout on 2026-10-09. Measurements come from `docs/INTEL.md` (this tree) and from the
author's `docs/INTEL_PERFORMANCE.md` and `sycl/TODO.md` on `maxfridbe/intel-arc-0.1.40`, cited as such. Nothing was
built or run; there is no card here. Where a fact needs the card it says "not verified" and what to check.

Conventions used in every section:

- "The three standard prompts" are the ones INTEL.md's merge notes compare output on: the 19-token Fibonacci prompt,
  the 2,184-token prompt and the 40K prompt (same text repeated), Coder IQ1_M, 256 greedy tokens, run as "How to run it
  by hand" in INTEL.md (`STRATA_VERIFY_DEVICE_PLAN=1 STRATA_VERIFY_NO_HOST=1 build-sycl-aot/strata ... --greedy`), plus
  the IQ2_XS 19-token and 2,184-token runs when a change touches an expert format the Coder does not use.
- "Bitwise" means the same output tokens and, where a parity test exists, the same bits in the tensors it compares.
  "Rounding order changes" means what the author accepted for `STRATA_SELECT_GEMM` (TODO.md 3b): the same text, a
  near-tie may flip after some tokens, equally coherent; the gate is then an eyeball of both outputs plus `llm-evals`.
- Every item ships behind an env switch, opt-in unless stated, the old path kept and the default output byte-identical
  (upstream `AGENTS.md`: one change per PR, opt-in, default path byte-identical, numbers with what they were measured on).
- The prompt phase timer is `STRATA_PREFILL_TIMING=1` (`sycl/src/prefill/prefill.cpp:1939-2005`, phases at 1943-1949:
  `embed+steps, hc read, gdn, qsa proj, qsa indexer, qsa select, qsa attn, router+shared, host grouping, gather, wait
  copy, dequant, gemm gate/up, gemm down, combine, ple, kv stage, gdn conv+gates, gdn recurrence, gdn out proj`). It
  costs 8 to 10% itself (INTEL.md "XMX v2 in the full matrix") and its event fold is what made "host grouping" look
  like 6.5% (author's INTEL_PERFORMANCE.md: at 40K the fold took 1,412 of the phase's 1,527 ms). So: phase shares with
  the timer on, tok/s and time to first token with it off, never mixed.
- Per-kernel device time: `unitrace -d` around the engine in the `strata-sycl-dev:unitrace` image plus
  `sycl/rank_kernels.py` (INTEL.md "Profiling"). Warm the clocks first.
- The standard bench is the author's `sycl/benchy.sh` (v1 prompts at 20, 2,185, 8,000, 40,000, 128,000 and 256,000
  tokens, 256 greedy tokens each, cold page cache; `sycl/bench/v1`, output `sycl/benchy-results/v1-<date>/matrix.md`).
  It is not in this tree yet (the review's Tier 0 port list). Until it is: the by-hand greedy runs above at 2,185,
  8,000 and 40,000 tokens, A and B interleaved, 3 to 5 pairs, medians, output tokens diffed.
- Build and tests: `bash sycl/tools/build.sh strata` in the dev image (`AOT=bmg-g31`), then `ctest --test-dir
  build-sycl-aot` (the parity tests registered in `sycl/CMakeLists.txt:224-262`). `-fsycl-default-sub-group-size=32`
  and `-fp-model=precise` are load-bearing (`sycl/CMakeLists.txt:55-66`); new kernels inherit them.
- The MoE geometry everywhere below: `N = 2560` (n_embd), `n_ff = 640`, gate+up interleaved as 1280 rows of 2560,
  down 2560 rows of 640, `K = 10` routed experts of 512 per token (`kernels.dp.cpp:23`, `prefill.cpp:3949-3959`).
  A dequantized expert is 1280 x 2560 + 2560 x 640 FP16 = 6.55 MB + 3.28 MB = 9.83 MB.

---

### P1. Hide the grouping bubble: wait on the ids copy, not on the queue, with the shared expert in flight

**Goal and measured motivation.** Per MoE layer per chunk the host groups the routed (token, k) pairs by expert. It
copies the ids down and then drains the whole compute queue with `m.cs->wait()`, so the GPU idles from the end of the
shared-expert projections until the host has counted, offset, filled and copied back (three small loops over T x K
ints plus two uploads). The author timed the host side at 47 ms of loops and 24 ms of uploads over a 40K prompt's 528
groupings (INTEL_PERFORMANCE.md, 2026-10-04; a later run: 14 ms and 5 ms), and the "6.5 to 9.5%" the phase timer shows
for "host grouping" is mostly the timer's own event fold. What is left to win is the GPU-idle gap of one wake-up plus
~0.1 to 0.3 ms of host work per layer, 48 layers x 10 chunks at 40K: 0 to 4% of prompt time (the review, E). INTEL.md's
GPU timeline at 80K put "the per-layer grouping sync" at 5.2 s of 101 s (5%), measured with the timer on. This item is
an A/B, not a given.

**Where.**
- `sycl/src/prefill/prefill.cpp:3168-3185`: the MoE head, `route(...)` then the shared expert (`native_proj` x3,
  `swiglu_pair`, `m.gemm.bf16` for the scalar gate).
- `sycl/src/prefill/prefill.cpp:3279-3303`: `pt.mark(kPfHostGroup, cs)`, the ids copy (`copy_i32` into the
  host-mapped `m.grp_dev` at 3287, or `m.cs->memcpy` into `m.ids_host` at 3295-3296), `m.cs->wait()` at 3301,
  `pt.fold()` at 3303.
- `sycl/src/prefill/prefill.cpp:3304-3318`: the peer (`pe.on`) wait, which also needs the full drain today.
- `sycl/src/prefill/prefill.cpp:3319-3363`: the count, offset, slot and src loops (host, reads `ids_h`, writes
  `slot_h`, `src_h`).
- `sycl/src/prefill/prefill.cpp:3379-3403`: the two uploads back (`copy_i32` from host-mapped memory, or `memcpy`).
- `sycl/src/prefill/prefill.cpp:996-1023`: the host-mapped grouping buffer `m.grp_host` / `m.grp_dev`
  (`sycl::malloc_host`, `STRATA_GROUP_COPY=1` forces the pageable copies instead).
- `sycl/src/prefill/prefill.cpp:1939-2005`: `PfTimer::mark` / `fold` (the fold reads profiling info of every mark
  recorded so far, so every mark must have completed when it runs).
- `sycl/src/prefill/kernels.dp.cpp:2708-2740` `route`; `2862-2880` `copy_i32` (a kernel, not a memcpy).
- `sycl/include/strata/sycl_queue.hpp:12-13` `q_of`: `m.cs` is the dpct in-order queue.
- Upstream for the mirror: `src/prefill/prefill.cpp:3013-3018` (`cudaStreamSynchronize(m.cs)` at the same place).

**Current behaviour.** The shared expert is enqueued first, then the ids copy, then the host waits for everything:

```cpp
// prefill.cpp:3177-3182
route(m.logits, m.ids, m.w, T, m.g->n_expert, m.cs);
// the shared expert and its scalar gate
if (!native_proj(m.gemm, wsg, m.mixed_h, m.sgate, T, v.name("ffn_gate_shexp.weight"), err)) return false;
if (!native_proj(m.gemm, wsu, m.mixed_h, m.sup, T, v.name("ffn_up_shexp.weight"), err)) return false;
swiglu_pair(m.sgate, m.sup, m.sh_h, T, m.cs);
if (!native_proj(m.gemm, wsd, m.sh_h, m.shared, T, v.name("ffn_down_shexp.weight"), err)) return false;
```
```cpp
// prefill.cpp:3283-3303 (comments elided)
const bool grp_mapped = m.grp_host != nullptr;
int32_t* ids_h = grp_mapped ? m.grp_host : m.ids_host.data();
...
if (grp_mapped) copy_i32(m.grp_dev, m.ids, T * K, m.cs);
else m.cs->memcpy(m.ids_host.data(), m.ids, (size_t)T * K * 4);
core::progress_at("reading the prompt (batched): waiting for the GPU (attention, router) at layer", l, p0);
m.cs->wait();
core::progress_at("reading the prompt (batched): layer", l, p0);
pt.fold();
```

Because `m.cs` is in-order, waiting on the ids copy's event is the same as `m.cs->wait()` for everything enqueued
before the copy; the copy is already the last thing in the queue, so swapping `wait()` for the event alone gains
nothing. The gain comes from moving the shared expert after the copy so it runs while the host groups. The comment at
3281-3282 matters: the full drain is also what orders this layer's host writes to `grp_host` (slot/src) after the
previous layer's `copy_i32` kernels that read it; an event wait on the ids copy keeps that guarantee (in-order queue:
the copy completes after every earlier command).

**Change.**
1. Reorder the MoE head so the ids copy is enqueued right after `route`, before the shared expert. Keep the event:
   `sycl::event ids_ev = grp_mapped ? copy_i32_ev(m.grp_dev, m.ids, T * K, m.cs) : m.cs->memcpy(...)`. `copy_i32`
   returns void today (`kernels.dp.cpp:2862`); add `sycl::event copy_i32_ev(...)` beside it (same kernel, returns the
   `parallel_for`'s event) and declare it in a port-only header (`sycl/include/strata/prefill/kernels_sycl.hpp`, new;
   upstream's `include/strata/prefill/kernels.hpp` is not edited). The `route` kernel reads `m.logits` written by the
   router GEMM at 3176 and nothing the shared expert touches, so the move is legal.
2. Enqueue the shared expert (the four lines at 3179-3182 and the two `gemm.bf16` scalar-gate lines at 3184-3185)
   after the copy. Its inputs (`m.mixed_h`, `m.mixed_bf`) and outputs (`m.sgate`, `m.sup`, `m.sh_h`, `m.shared`,
   `m.sg`) are not read or written by the host grouping.
3. Replace `m.cs->wait()` at 3301 with `ids_ev.wait()` when the new switch is on and `!pe.on`. With `pe.on` (a peer
   device; a stub on SYCL, `peer_experts.cpp:19-23` per the review) keep the full wait.
4. The timer: `pt.fold()` must only fold marks whose events have completed. Add `PfTimer::fold_until(size_t n)` that
   folds marks `[0, n)` and keeps `[n, used)` for the next fold, and record `n = pt.used` right after the ids copy's
   mark (a new `pt.mark(kPfHostGroup, cs)` sits before the copy as today; the shared expert's own mark `kPfRouter`
   then follows the copy, so its time is charged to "router+shared" as before). With `STRATA_PREFILL_SYNC=1` the
   marks already wait on the queue; nothing changes there.
5. The host loops (3319-3363) read `ids_h` only after `ids_ev.wait()`; the uploads at 3379-3403 go onto `m.cs` after
   the shared expert and before the gather (`gather_rows16` at 3438), so every consumer stays ordered on the in-order
   queue. No other change to the loops.
6. Switch: `STRATA_PF_GROUP_ASYNC=1` (default off). Off: the exact order of today, byte for byte. `STRATA_GROUP_COPY=1`
   (pageable copies) keeps working under both.
7. Mirrors nothing upstream: upstream drains the stream at the same place (`src/prefill/prefill.cpp:3017`) and its
   GPU-side grouping (`fused::group`, 3228) exists only on the fused int8 path the SYCL build does not compile.

**Must not change.** Output bitwise on the three standard prompts: the change is a pure reordering of independent
launches on one in-order queue. No new VRAM. The `grp_host` host-mapped buffer is reused per layer; the ordering
argument above (in-order queue, event completes after every earlier command) is what keeps that safe, so a reviewer
should check it before merge. Nothing may touch `m.ids`, `m.w`, `m.slot_dev`, `m.src_dev` between the copy and the
wait.

**Tests.** No parity test covers the prompt MoE loop. Add to the output-identity run: the three standard prompts with
the switch on and off, tokens diffed (`cmp` of the `T` lines). A synthetic check worth 20 lines: with
`STRATA_PREFILL_SYNC=1` the marks print in order; confirm the "router+shared" mark now lands after "host grouping".
`STRATA_VERIFY_DEBUG=1` (the per-layer residual ladder, INTEL.md) must print identical R hashes with the switch on.

**Bench.** Timer off: prompt tok/s at 2,185, 8,000 and 40,000 tokens, interleaved pairs, medians of 3 to 5; time to
first token at 40K. Timer on, once: "host grouping" and "router+shared" phase ms before and after (expect the host
grouping phase to shrink by roughly the shared expert's device time per layer, not to zero: the fold still charges
the gap to the last mark before it). `unitrace -d`: the gap between the last `route_kernel` and the first dequant
kernel per layer is the number this item moves.

**Expected gain, risk, effort.** 0 to 4% of prompt time (the author's host timers say the host work itself is under
1% without the profiler; the GPU-idle gap it hides is one launch latency plus the loops). Low risk: the one failure mode
is a reorder bug that reads stale `grp_host`, which the residual ladder catches on the first prompt. 0.5 day.

**Depends on / conflicts with.** Independent of P2 to P6. P3 changes the grouping output (groups of experts) but
not this ordering; do P1 first so P3's A/B is measured against the hidden bubble. No overlap with the author's branch
(`STRATA_SELECT_GEMM`, `STRATA_ATTN_PERCELL` touch QSA only).

---

### P2. Dequant kernel occupancy: 128 to 256 lane work-groups over 4 to 8 superblocks

**Goal and measured motivation.** The expert dequant is the largest single phase of a short prompt and the second of
a long one: 31.8% at 2,184 tokens, 22.4% at 8,000, 19.4% at 40K (author's INTEL_PERFORMANCE.md, 2026-10-04); 20% of a
4K IQ3_S prompt (INTEL.md, 2026-10-07). It writes 9.83 MB of FP16 per expert in 42 us for gate/up (INTEL.md "XMX":
twice the 20 us GEMM that reads it), about 160 GB/s against a card that streams 600 (`sycl/probe/bw.cpp`). The kernel
runs one 32-lane work-group per 256-value superblock: 12,800 work-groups for gate/up and 6,400 for down per expert,
each a single sub-group with 16 bytes of store per lane. The review's expectation: 160 to 300 GB/s, up to 8 to 10% of
prompt time; the earlier vector-store change in the same kernel took dequant per expert from 0.085 to 0.030 ms and the
2,184-token prompt from 496 to 575 tok/s (INTEL.md "Speed work, 2026-09-30").

**Where.**
- `sycl/src/kernels/cuda/iq_kernels.dp.cpp:4390-4418` `iq_dequant_gu_f16`: the launch,
  `nd_range(range(1, 2, n_ff * per_row) * range(1, 1, 32), range(1, 1, 32))`.
- `sycl/src/kernels/cuda/iq_kernels.dp.cpp:4292-4317` `iq_dequant_f16` (the down matrix, "flat"): the launch,
  `nd_range(range(1, 1, n / 256) * range(1, 1, 32), range(1, 1, 32))`.
- `sycl/src/kernels/cuda/iq_kernels.dp.cpp:2755-2774` `dequant_flat_kernel` and `dequant_gu_kernel`: one superblock
  per group, lane = `get_local_id(2)`.
- `sycl/src/kernels/cuda/iq_kernels.dp.cpp:2723-2753` `dq_dispatch` and the per-type `dq_*` (IQ2_XS 2403, IQ3_S 2450,
  IQ1_M 2468, IQ4_NL 2487): every one takes `tid` in 0..31 and writes 8 values as one 16-byte run (`store_run`).
- `sycl/src/prefill/prefill.cpp:3964-3973`: the caller (`compute` lambda), `iq_dequant_gu_f16` into `m.dq_gu[q]`,
  `iq_dequant_f16` into `m.dq_d[q]`; `blob_dequant_f16` at 3972 is the Q2_0 pack's path (`kernels.dp.cpp:2763`), out
  of scope.
- `include/strata/kernels/iq_kernels.hpp:29, 37`: the declarations (unchanged).
- `sycl/src/kernels/dequant_bench.cpp` (header lines 1-3): times both entry points on random blocks and prints a hash
  of the FP16 output, "so two builds can be compared bit for bit". Built at `sycl/CMakeLists.txt:280-281`.

**Current behaviour.**

```cpp
// iq_kernels.dp.cpp:2764-2774
__dpct_inline__ void dequant_gu_kernel(int ty, const void *__restrict__ gate, const void *__restrict__ up,
                                       int64_t per_row, sycl::half *__restrict__ y) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i = item_ct1.get_group(2);
    const int parity = item_ct1.get_group(1);
    const int64_t r = i / per_row, c = i % per_row;
    dq_dispatch<sycl::half>(ty, parity ? up : gate, i, y + ((2 * r + parity) * per_row + c) * QK_K,
                            item_ct1.get_local_id(2));
}
```
```cpp
// iq_kernels.dp.cpp:4405-4410
cgh.parallel_for<dpct_kernel_name<class dequant_gu_kernel_2397ca>>(
    sycl::nd_range<3>(sycl::range(1, 2, (unsigned)(n_ff * per_row)) * sycl::range(1, 1, 32),
                      sycl::range(1, 1, 32)),
```

Each work-group is one sub-group; Xe2's thread dispatcher issues one work-group per slot, so a 32-lane group leaves
the core's other 7 XVE threads to other groups and the dispatch rate, not the memory system, bounds the kernel. (The
per-lane work is 8 values: a few table reads, one 16-byte store.) Not verified on the card: the dispatcher's limit is
the hypothesis the bench tests.

**Change.**
1. Add `constexpr int kDqSb = 4;` (superblocks per work-group) variants of both kernels, selected at run time:
   `dequant_flat_kernel_w<SB>` and `dequant_gu_kernel_w<SB>` with `SB` in {1, 4, 8}. Body: `const int lane =
   get_local_id(2) & 31; const int sb = get_local_id(2) >> 5; const int64_t i = get_group(2) * SB + sb; if (i >= n_sb)
   return; dq_dispatch<sycl::half>(ty, ..., i, y + <same offset formula>, lane);`. The `(parity, r, c)` arithmetic
   and the destination offset are unchanged; only the mapping of (group, local id) to superblock changes.
2. Launch: `nd_range(range(1, 2, ceil(n_sb / SB)) * range(1, 1, 32 * SB), range(1, 1, 32 * SB))` for gate/up (n_sb =
   n_ff * per_row = 6,400), `nd_range(range(1, 1, ceil(n_sb / SB)) * range(1, 1, 32 * SB), ...)` for down (n_sb =
   n / 256 = 6,400). Keep `use_root_sync` and the `fp16` aspect check as they are. The kernels have no barriers and no
   local memory, so the wider group costs nothing but registers; `[[sycl::reqd_sub_group_size(32)]]` is implied by the
   build flag, add it explicitly anyway (the `dq_*` bodies are lane-indexed and must not run at 16).
3. The pass-through bound: `n_sb` is 6,400 for both matrices, divisible by 4 and 8 (6,400 / 8 = 800), so the `i >=
   n_sb` guard never fires on the model's shapes; keep it for `iq_dequant_f16`'s other callers (any `n % 256 == 0`).
4. Switch: `STRATA_DQ_WG=<lanes>` read once (static), values 32 (default: today's kernel, same launch), 128 (SB 4),
   256 (SB 8); any other value falls back to 32 with one stderr line. Both entry points read the same switch.
5. Put the new kernels next to the old ones in `iq_kernels.dp.cpp` (the whole file is port-owned; the CUDA original
   `src/kernels/cuda/iq_kernels.cu` has no wide variant to mirror: this is the port's own change, like its
   vector-store fix). If `sycl/tools/fixups.py` is the house style for mechanical edits to migrated files, this one is
   a hand edit (new code, not a pattern replacement) and goes through the re-migration merge as a conflict to keep.
6. Also try, in the bench only: `SB = 2` and a 1-D grid for gate/up (fold `parity` into the group index) in case
   the 2-D range costs dispatch time. Keep whichever wins; do not ship more than one wide variant.

**Must not change.** Bitwise: every lane computes the same values and writes the same 16-byte run to the same
address as today; `dequant_bench`'s FP16 output hash must be identical for every type at 32, 128 and 256. No VRAM
change (the ring `m.dq_gu[DQ] / m.dq_d[DQ]` at `prefill.cpp:1190` is untouched). The down matrix is written row-major
2560 x 640 and the gate/up interleaved `2r + parity` as before (`prefill.cpp:3980-3983` and `swiglu_il_kernel` at
`kernels.dp.cpp:1775-1785` depend on that layout).

**Tests.** Existing: `iq_parity` and `iq_multi_parity` (`sycl/CMakeLists.txt:227-240`) test the decode dots, not this
kernel; `iq_parity` needs fixtures the port lacks (INTEL.md). New: (a) extend `dequant_bench` to run every type in
`kTypes` (`dequant_bench.cpp:17-18`, plus IQ1_M 29 and Q4_K/Q5_K if `is_iq` admits them) at `STRATA_DQ_WG` 32, 128 and
256 and assert equal hashes (exit 1 on a mismatch, so it can be a ctest); (b) register that as `dequant_wide_parity`
in `sycl/CMakeLists.txt`; (c) the output-identity run on the three standard prompts and the IQ2_XS pair (IQ2_XS and
IQ1_M differ in `dq_*` bodies).

**Bench.** `dequant_bench all 64` at each `STRATA_DQ_WG` (device ms per expert for gate/up and down, derive GB/s from
9.83 MB per expert); then `STRATA_PREFILL_TIMING=1` once for the "dequant" phase share at 2,185 and 8,000 tokens;
then timer off, tok/s at 2,185, 8,000 and 40,000 tokens, interleaved pairs, medians. `unitrace -d`: device time of
`dequant_gu_kernel*` and `dequant_flat_kernel*` summed over a 4K chunk.

**Expected gain, risk, effort.** Dequant 160 to 300 GB/s if the dispatcher is the bound (E), which is 5 to 10% of a
2K to 8K prompt and 3 to 5% at 40K. If the bound is the per-type table reads (IQ grids in global memory as `static
const` arrays, INTEL.md item 3 of "How it was made") the gain is small and the next lever is staging the grid in local
memory, a separate item. Low risk: pure index remap; a wrong guard writes past the ring, which the hash test catches.
1 to 2 days including the bench.

**Depends on / conflicts with.** None. P3 keeps calling the same two entry points per expert, so P2 and P3 compose;
T2c replaces these kernels with int8 writers for its path and would take the same work-group shape.

---

### P3. Group 32 to 64 experts per dequant ring slot and one batched GEMM per group

**Goal and measured motivation.** The FP16 expert path launches five kernels per routed expert: two dequants, the
gate/up GEMM, the SwiGLU and the down GEMM (`prefill.cpp:3964-3984`). At a 4,096-token chunk nearly all 512 experts of
a layer are routed (INTEL.md: "a 4K prompt touches nearly all 512 experts of every layer"), so a layer is about 2,500
launches and a 4K chunk about 120,000 (the review). The GEMMs are small (M = rows routed to the expert, mean 80 at
4,096 x 10 of 512, skewed; author's TODO.md 4) and oneMKL's FP16 GEMM drops to 28 to 32 TFLOP/s at 256 to 512 rows
and lower below (author's `onednn_gemm_bench`, INTEL_PERFORMANCE.md 762), so each call is weight-read bound and
launch-latency bound. The expert GEMMs are 22.8% / 20.2% / 18.3% of the prompt at 2,184 / 8,000 / 40K tokens
(INTEL_PERFORMANCE.md) and 34% of a 4K IQ3_S prompt (INTEL.md). Expectation: launches per layer 2,500 to about 200,
5 to 10% of prompt time (E; depends on how oneMKL schedules a group batch).

**Where.**
- `sycl/src/prefill/prefill.cpp:226` `constexpr int DQ = 2;` (the dequantized-expert ring) and `:1190` its
  allocation (`m.dq_gu[i] = o.take<uint16_t>(1280 * 2560)`, `m.dq_d[i] = o.take<uint16_t>(2560 * 640)`); the
  matching count at `:1868` in the byte-count function (the function at 1835-1884 sizes the prompt's device
  buffers "buffer for buffer").
- `sycl/src/prefill/prefill.cpp:3907-3992` the `compute` lambda; the FP16 path at 3964-3984.
- `sycl/src/prefill/prefill.cpp:3993-4011` the non-stream_all loop (`stage_one` lookahead of `STAGE - 1 = 7`,
  `compute(j, blob, slot)` per expert, `ext_oneapi_submit_barrier({*m.copied[slot]})` before a staged expert).
- `sycl/src/prefill/prefill.cpp:4012-4063` the stream_all walk (same `compute`, in id order through the ring).
- `sycl/src/prefill/prefill.cpp:3347-3370` `m.off` (cumulative row offsets in expert id order) and `order` (the
  routed experts in id order); `:3438` `gather_rows16(m.mixed_h, m.src_dev, m.Xs, T * K, N, m.cs)` fills `m.Xs`
  (FP16, T x K rows of N) in that order.
- `sycl/src/prefill/prefill.cpp:1174-1180` `m.Xs` (T*K*N FP16), `m.GU` (FP32), `m.Hh` (T*K*640 FP16), `m.Dm`
  (T*K*N FP32).
- `sycl/src/prefill/gemm.dp.cpp:773-808` `Gemm::f16` (one `dpct::blas::gemm` per call, 799-806);
  `include/strata/prefill/gemm.hpp:40` its declaration, `:15-85` the class (`handle_` is a
  `dpct::blas::descriptor_ptr`, `gemm.dp.cpp:404-411`).
- `sycl/include/dpct/blas_utils.hpp:1077-1084` `dpct::blas::gemm_batch` (pointer-array form, one m/n/k for the
  batch), `:1207` the `half, half, float, float` instance, `:1256-1265` the strided form.
- `sycl/include/dpct/detail/blas_utils_detail.hpp:372-417` `gemm_batch_impl`: it calls
  `oneapi::mkl::blas::column_major::gemm_batch(q, transpose*, transpose*, int64_t* m, int64_t* n, int64_t* k, Ts*
  alpha, const Ta** a, int64_t* lda, const Tb** b, int64_t* ldb, Ts* beta, Tc** c, int64_t* ldc, int64_t group_count,
  int64_t* group_size)` with `group_count = 1`, the host arrays malloc'ed and freed in a `host_task` after the event
  (so oneMKL reads the host arrays asynchronously; the caller's pointer arrays must outlive the batch).
- `sycl/src/prefill/kernels.dp.cpp:2781-2797` `swiglu_interleaved` (launch over `n * 640` elements, 256 per group)
  and `:1775-1785` `swiglu_il_kernel` (row r, `gu[r * 1280 + 2k]`, `gu[r * 1280 + 2k + 1]`, `hf_sat`).
- `sycl/src/prefill/prefill.cpp:974-979` `m.copied[i]` / `m.used[i]` events of the staging ring; `:3974-3977` the
  `dpct::sync_barrier(m.used[slot])` that releases a ring slot once its blob was read by the dequant.

**Current behaviour.**

```cpp
// prefill.cpp:3964-3984 (the FP16 path; MMQ branches above it are dead on SYCL)
const int q = (int) (j % DQ);
if (lay.native) {
    const auto& f = lay.fmt[(size_t) l];
    strata::kernels::iq_dequant_gu_f16(f.gu_type, blob_dev, blob_dev + f.up_off, f.n_ff, f.n_embd, m.dq_gu[q], m.cs);
    strata::kernels::iq_dequant_f16(f.d_type, blob_dev + f.down_off, f.n_embd * f.n_ff, m.dq_d[q], m.cs);
} else {
    blob_dequant_f16(blob_dev, m.dq_gu[q], m.dq_d[q], m.cs);
}
if (slot >= 0) { dpct::sync_barrier(m.used[slot], m.cs); m.used_of[slot] = slot; }
const int64_t o0 = m.off[(size_t) e], ne = m.cnt[(size_t) e];
pt.mark(kPfGemmGU, cs);
m.gemm.f16(m.Xs + o0 * N, m.dq_gu[q], m.GU + o0 * 1280, ne, 1280, N);
swiglu_interleaved(m.GU + o0 * 1280, m.Hh + o0 * 640, ne, m.cs);
pt.mark(kPfGemmD, cs);
m.gemm.f16(m.Hh + o0 * 640, m.dq_d[q], m.Dm + o0 * N, ne, N, 640);
return true;
```

The ring of `DQ = 2` slots alternates per expert. On one in-order queue nothing overlaps anyway; the ring only lets
expert j+1's dequant be enqueued without a host wait. `m.off` is cumulative in id order and `order` is id order, so
the rows of any run of consecutive routed experts are contiguous in `m.Xs`, `m.GU`, `m.Hh` and `m.Dm`.

**Change.**
1. Ring geometry. Replace the per-expert slot with a per-expert slot inside a group: `G` experts per group
   (`STRATA_PF_GROUP=<G>`, default 0 = today's path; try 32 and 64), `DQ_GROUPS` groups in the ring (default 1; 2 if
   the measurement shows oneMKL leaving the queue idle between batches). The ring is then `DQ_GROUPS * G` dequantized
   experts: `m.dq_gu[s]` / `m.dq_d[s]` for `s < DQ_GROUPS * G`, allocated as today at `prefill.cpp:1190` and counted
   at `:1868` (the count function must see the same number or `--check` undersizes the loan). Bytes: 9.83 MB per
   expert, so G = 32 and DQ_GROUPS = 1 is 315 MB, G = 64 is 629 MB, two groups double that. Default to 315 MB; state
   the number in the PR.
2. Grouping on the host, where `order` is built (`prefill.cpp:3364-3370`): groups are consecutive runs of `order`
   of length `G` (the last shorter). Per group, host arrays (pinned once per `Prefill`, sized `RING_MAX`-like for
   the largest group count per layer): `const void* a_gu[G]`, `const void* b_gu[G]`, `void* c_gu[G]`, `int64_t
   m_g[G]`, and the same for down; `n_g[G]` all 1280 (gate/up) or 2560 (down), `k_g[G]` all 2560 or 640, `lda/ldb/ldc`
   per group as the single call sets them (`ld = K` for A and B in the column-major view, `ldc = N`), `group_size[G]`
   all 1, `alpha[G] = 1`, `beta[G] = 0`. Keep two sets per ring group and reuse them only after the batch that read
   them completed (an event per group, `m.gemm_done[s]`), because oneMKL reads the host arrays asynchronously
   (the deferred free in `blas_utils_detail.hpp:411-416` is the evidence).
3. The per-expert step inside `compute` becomes: dequant into slot `s = (group index % DQ_GROUPS) * G + (j % G)`,
   release the staging slot as today (3974-3977), record `a/b/c/m` for this expert, return. On the group's last
   expert (or the layer's last) issue three launches: (a) gate/up batch, (b) one `swiglu_interleaved(m.GU + o0 *
   1280, m.Hh + o0 * 640, rows_of_group)` where `o0 = m.off[order[j0]]` and `rows_of_group = m.off[order[j]] +
   m.cnt[order[j]] - o0` (contiguous, see above), (c) down batch. `pt.mark(kPfGemmGU)` before (a), `kPfGemmD` before
   (c), as today.
4. The batched call. `dpct::blas::gemm_batch` (pointer form, `blas_utils.hpp:1077`) takes one `m` for the whole
   batch, which does not fit variable rows per expert without padding `m.GU`/`m.Dm` to a per-expert stride. Call
   oneMKL's group API directly instead, exactly as `gemm_batch_impl` does (`blas_utils_detail.hpp:402-410`) but with
   `group_count = G` and `group_size[g] = 1`, so every expert has its own `m[g] = cnt[e]`:
   `oneapi::mkl::blas::column_major::gemm_batch(q, trans_arr, nontrans_arr, n_arr, m_arr, k_arr, alpha_arr,
   (const sycl::half**) b_gu, lda_arr, (const sycl::half**) a_gu, ldb_arr, beta_arr, (float**) c_gu, ldc_arr, G,
   group_size, {})` in the same column-major view as `Gemm::f16` (`gemm.dp.cpp:799-806`: op(A) = W^T with lda = K,
   B = X with ldb = K, C = Y with ldc = N; here per expert "N" = 1280 or 2560 is oneMKL's m and the row count is
   oneMKL's n). The `<half, half, float, float>` type combination exists in the installed oneMKL (dpct instantiates it
   at `blas_utils.hpp:1207`). Add `Gemm::f16_group(...)` in `sycl/src/prefill/gemm.dp.cpp` (port-only; declared in the
   new `sycl/include/strata/prefill/kernels_sycl.hpp` or a `gemm_sycl.hpp` beside it) taking the host arrays and
   returning the `sycl::event`. Whether oneMKL's group batch with G groups of size 1 runs as one kernel or as G
   launches is not verified: `oneapi/mkl/blas.hpp` in the image documents the signature, not the schedule. Measure
   first (step 7).
5. Fallback inside the group when a batch member has `cnt == 0`: cannot happen (`order` holds experts with `cnt > 0`,
   `prefill.cpp:3366-3368`). Peer rows (`on_peer`, multi-GPU) do not exist on SYCL; keep the group path off when
   `!order_peer.empty()`.
6. The stream_all walk (4012-4063): the same `compute`, so groups form in id order there too; an unrouted entry only
   gives its slot back and does not enter a group.
7. Bench first, in isolation: extend `xmx_gemm_bench` (`sycl/src/kernels/xmx_gemm_bench.cpp`, linked with
   `strata_prefill`, `sycl/CMakeLists.txt:290-291`) with a mode that runs 64 experts' gate/up as 64 `Gemm::f16` calls
   against one group batch with the real row distribution (draw 64 counts with mean 80 from a geometric or from a
   dumped histogram, see P4 step 1), M total about 5,000, and reports device ms for both. If the batch is not at
   least 1.3x the loop at G = 32, stop here and file the number.
8. Switch: `STRATA_PF_GROUP=0` (default, today's path), `32`, `64`; `STRATA_PF_GROUP_RING=1|2`. Mirrors nothing
   upstream: upstream's grouped path is the fused int8 MMQ library (`fused::experts`, `mmq_ctx_->run` with
   `MMQ_GROUP = 16`, `prefill.cpp:830, 3936-3961`) that the SYCL build stubs out (`moe_fused_stub.cpp`,
   `sycl/CMakeLists.txt:212-214`); the group bookkeeping here can copy its `bounds_host` idea (3413-3420) but not its
   kernels.

**Must not change.** Rounding order changes, same text, near-tie flips allowed: a batched GEMM may pick a different
oneMKL kernel (different K split or tile) than the single call, so the FP32 sums can differ in the last bits; the
author accepted the same class of change for `STRATA_SELECT_GEMM`. The dequantized bytes and the SwiGLU are bitwise
(P2's hash test still passes; `swiglu_il_kernel` is element-wise). VRAM: the ring grows from 19.7 MB (DQ = 2) to
`DQ_GROUPS * G * 9.83 MB`; it lives in the prompt's device buffers (`borrow` arena, `prefill.cpp:1082-1086`), which
are lent cache slots above 32K (`--prefill-borrow`) or own VRAM below, so every 1.44 MB (IQ2_XS) or 2.0 MB (Coder)
of ring is one fewer resident expert for a model that does not fit: at G = 32 about 220 Coder slots or 440 IQ2_XS
slots during the prompt only (lent slots are refilled after it). Keep `--check`'s count (1835-1884) in step so the
loan is sized right; on the Coder at 32K (1.9 GB free, INTEL.md) 315 MB fits without evicting anything.

**Tests.** Existing: none covers this loop (`gemm_bf16_parity` is a CUDA-only target, top-level `CMakeLists.txt:766`;
no SYCL counterpart is registered). New: (a) `gemm_group_parity` (new, under `sycl/src/prefill/`, built like
`xmx_gemm_bench`): random FP16 X (M = 5,000 rows), 32 random FP16 weight matrices 1280 x 2560, `cnt` drawn as in step
7; compare the group batch against 32 `Gemm::f16` calls, max relative difference per element under 1e-3 (FP32
accumulation of FP16 products at K = 2560; both orders are "correct", the test guards against wrong pointers, strides
or rows, which show as O(1) errors) and the same for down (K = 640); (b) the output-identity run: three standard
prompts plus IQ2_XS, with `STRATA_PF_GROUP=32`, outputs compared by eye and `llm-evals`; (c) `STRATA_DBG_NAN=1`
(INTEL.md: reports the experts' FP16 GEMM inputs) must stay silent.

**Bench.** Step 7's bench first (device ms, 64 experts, gate/up and down, loop vs batch at G 16/32/64). Then
`STRATA_PREFILL_TIMING=1` once: "gemm gate/up" and "gemm down" phase ms at 2,185 and 8,000 tokens; `unitrace -d`:
the count of oneMKL kernels per layer (today about 1,000 at a 4K chunk) and the launch gaps between them. Then timer
off: prompt tok/s at 2,185, 8,000 and 40,000, interleaved pairs, medians; time to first token at 40K; decode tok/s
after the 40K prompt on IQ2_XS (the ring's slot cost shows there, if anywhere).

**Expected gain, risk, effort.** 5 to 10% of prompt time if oneMKL runs a group as one or few kernels; near 0 if
it loops internally (then the launch count is unchanged and only the host call overhead drops). Medium risk: the
rounding-order change needs the eyeball gate; the ring growth costs cache slots on models beyond VRAM; a wrong
`m_arr` row count writes into the next expert's rows (the parity test catches it). 2 to 3 days.

**Depends on / conflicts with.** P1 first (so the A/B does not include the bubble). P2 composes (same entry points
per expert). P4 is the per-group alternative for large-row experts and would sit inside the group's issue step: for
experts with `cnt >= 128` call oneDNN per expert, batch the rest; P4 can be done without P3. T2c replaces the
dequant and the GEMM of this path; P3's grouping and ring bookkeeping carry over to it unchanged.

---

### P4. oneDNN FP16 matmul for expert groups of 128 or more rows

**Goal and measured motivation.** The author's `onednn_gemm_bench` (INTEL_PERFORMANCE.md 762, TODO.md 4,
2026-10-05, B70): on the expert gate/up shape (K 2560, N 1280) at 256 to 512 rows oneMKL FP16 does 28 to 32 TFLOP/s
and oneDNN FP16 54 to 56 (1.8 to 1.9x, same inputs and FP32 accumulation); 1.0 to 1.3x at 64 to 128 rows; slower
below 64 (0.6x at 16 to 32). On the dense 4,096-row projections oneDNN is 1.00 to 1.14x of oneMKL's 128 to 147
TFLOP/s, so this is an expert-shape item. The rows per expert at a 4,096-token chunk average 80 (4,096 x 10 / 512)
and are skewed; the share of rows that sit in experts with 128 or more rows is not measured (TODO.md: "to know the
share above 128"). Expert GEMMs are 18 to 23% of prompt time (above), 34% on IQ3_S 4K. The review's expectation: a
few % at 4K and longer prompts.

**Where.**
- `sycl/src/prefill/gemm.dp.cpp:773-808` `Gemm::f16`; the oneMKL call at 799-806. `include/strata/prefill/gemm.hpp:40`.
- `sycl/src/prefill/prefill.cpp:3978-3983` the two `m.gemm.f16` calls per expert with `ne = m.cnt[e]` rows.
- `sycl/CMakeLists.txt:212-217` the `strata_prefill` target (`-qmkl=sequential`; no oneDNN today), `:1-11` the
  header on how the port keeps upstream files untouched.
- `sycl/include/dpct/dnnl_utils.hpp`: dpct's oneDNN wrapper (engine, conv, rnn, pooling); it has no matmul (grep
  `matmul` finds nothing), so the matmul primitive is called through oneDNN's own API.
- Author's branch: `maxfridbe/intel-arc-0.1.40:sycl/src/kernels/onednn_gemm_bench.cpp` (116 lines): the build line
  (`icpx -fsycl ... -I<dnnl>/include -L<dnnl>/lib -ldnnl -qmkl -lmkl_sycl_blas`, lines 5-8), the shapes (37-38) and
  the oneDNN matmul recipe (src `[M,K]` f16 `tag::ab`, weights `[K,N]` as `W[N,K]` with `tag::ba`, dst `[M,N]` f32,
  `dnnl::matmul::primitive_desc pd(eng, src_md, wei_md, dst_md, attr)`, `sycl_interop::make_memory(..., usm, ptr)`,
  `mm.execute(strm, args)`).
- `sycl/tools/Dockerfile` and `sycl/tools/build.sh`: no oneDNN today (the dev image "carries only oneDNN's runtime",
  bench header line 5). The target VM has `intel-oneapi-dnnl-devel` (oneAPI 2026.1.1 with oneDNN present, per the
  task); the dev image needs the package added for a container build.

**Current behaviour.**

```cpp
// gemm.dp.cpp:799-806
ck(DPCT_CHECK_ERROR(dpct::blas::gemm(
       (dpct::blas::descriptor_ptr)handle_, oneapi::mkl::transpose::trans,
       oneapi::mkl::transpose::nontrans, (int)N, (int)T, (int)K, &alpha, W,
       dpct::library_data_t::real_half, (int)K, X,
       dpct::library_data_t::real_half, (int)K, &beta, Y,
       dpct::library_data_t::real_float, (int)ldy,
       dpct::compute_type::f32)),
   "cublasGemmEx f16");
```

Row-major `Y[T, N] = X[T, K] . W[N, K]^T` as a column-major `N x T = W^T . X`. The author's bench runs the oneDNN form
of the same product with the same memory (no reorder of W): weights `[K, N]` with format `ba` is exactly `W[N, K]`
row-major.

**Change.**
1. Measure the row distribution first (0.25 day, no new kernel): under `STRATA_PREFILL_TIMING=1` add a histogram of
   `m.cnt[e]` over routed experts per layer (buckets <32, 32-63, 64-127, 128-255, 256-511, 512+, with the row totals
   per bucket) printed at the end of the prompt beside the phase table (`prefill.cpp:4447-4460`). The share of rows
   in the 128+ buckets bounds P4's gain: oneDNN wins 1.8x only on those rows.
2. Build plumbing: a CMake option `STRATA_SYCL_ONEDNN` (default OFF) in `sycl/CMakeLists.txt` that does
   `find_package(dnnl CONFIG)` (oneAPI ships `dnnl-config.cmake`; the path on the VM is to be found with `ls
   /opt/intel/oneapi/dnnl/latest/lib/cmake`, not verified) and links `DNNL::dnnl` into `strata_prefill` with
   `-DSTRATA_SYCL_ONEDNN=1`. Add the `-devel` package to `sycl/tools/Dockerfile` for the container build; `build.sh`
   passes `-DSTRATA_SYCL_ONEDNN=ON` when `ONEDNN=1` is in the environment.
3. `Gemm::f16_dnnl(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy)` in
   `sycl/src/prefill/gemm.dp.cpp` under `#ifdef STRATA_SYCL_ONEDNN`: a lazily created `dnnl::engine`
   (`dnnl::sycl_interop::make_engine(q.get_device(), q.get_context())`) and `dnnl::stream`
   (`make_stream(eng, q)`) on `stream_`'s queue (so oneDNN work is ordered on the same in-order queue; check the header
   `oneapi/dnnl/dnnl_sycl.hpp` on the VM for the exact names: the bench uses `dnnl::sycl_interop::make_engine`,
   `make_stream`, `make_memory` with `memory_kind::usm`). A primitive cache keyed by `(T, N, K, ldy)`: `dnnl::matmul`
   primitives are shape-specific and creation costs host time (the bench creates one per shape per timing loop; in
   the engine, cache them; with M varying per expert the cache fills with up to a few hundred shapes per run, which
   is fine; or quantize M up to a multiple of 32 by using a padded `src_md` with runtime dims, `DNNL_RUNTIME_DIM_VAL`,
   check the header). Memory descriptors as the bench: `src_md({T, K}, f16, ab)`, `wei_md({K, N}, f16, ba)`,
   `dst_md({T, N}, f32, ab)` with `ldy` carried as strides `{ldy, 1}` when `ldy != N` (use the strides constructor
   instead of `tag::ab`). `beta` is 0 on both expert calls; the general `beta = 1` form needs a `sum` post-op
   (`attr.set_post_ops` with `append_sum(1.0f)`, check the header) and is out of scope here.
4. Dispatch in `Gemm::f16`: `if (dnnl_on_ && T >= dnnl_min_rows_) { f16_dnnl(...); return; }` before the oneMKL call,
   where `dnnl_min_rows_` comes from `STRATA_PF_DNNL_MIN_ROWS` (default 128; the author measured 1.0 to 1.3x at 64 to
   128 so 128 is the safe edge) and `dnnl_on_` from `STRATA_PF_DNNL=1`. Both expert calls (`prefill.cpp:3980, 3983`)
   pass through `Gemm::f16`, so no caller change. Dense projections go through `Gemm::bf16` and `native`, untouched.
5. The author's bench measured FP16 in, FP32 out for both libraries, so no new precision path; the oneDNN kernel may
   use a different K split than oneMKL's (see Must not change).
6. Mirrors nothing upstream (upstream has cuBLAS and hipBLASLt; `try_hipblaslt` at `gemm.dp.cpp:792-798` is the
   analogous "second library first" hook and a good model for the code shape).

**Must not change.** Rounding order changes, same text, near-tie flips allowed (two GEMM libraries, two summation
orders). Both paths stay FP16 in, FP32 accumulate and out; no BF16 or TF32 downgrade. No VRAM change (oneDNN matmul
with these descriptors needs no scratchpad for f16 to f32 by default; if `pd.scratchpad_desc().get_size()` is non-zero
use `scratchpad_mode::user` with a slice of `workspace_`, 32 MB at `prefill.cpp:802`, check the header). Default
build and default run are unchanged: the option is OFF and the switch is off.

**Tests.** Existing: none for `Gemm::f16`. New: `gemm_dnnl_parity` (under `sycl/src/prefill/`, built only with the
option): random FP16 X (M in {16, 64, 127, 128, 256, 512}) and W (1280 x 2560 and 2560 x 640), oneDNN against
oneMKL, max relative error under 1e-3 per element and the row/column placement checked with a one-hot test (a single
1.0 in X must land in the right row of Y). Output-identity run on the three standard prompts with
`STRATA_PF_DNNL=1`, outputs compared by eye and `llm-evals`.

**Bench.** The author's `onednn_gemm_bench` first, built on the VM with its header's line (confirms the 1.8 to 1.9x on
this driver and oneDNN version; add M = 96, 128, 160, 192 rows to `shapes`, lines 37-38, to find the crossover).
Then step 1's histogram at 2,185, 8,000 and 40,000 tokens (share of rows at 128+). Then `STRATA_PREFILL_TIMING=1`:
"gemm gate/up" and "gemm down" ms; timer off: tok/s at the three sizes, interleaved pairs, medians.

**Expected gain, risk, effort.** If 30 to 50% of routed rows sit in experts with 128+ rows (E; the skew favours this
at 4K chunks), the expert GEMM phase shrinks by about 15 to 25% of itself, 3 to 6% of prompt time at 4K+; less on
short prompts where every expert has few rows. Low risk: a second library on the same queue; the failure modes are
a primitive-creation stall per shape (the cache fixes it) and a stride mistake (the one-hot test). 1 to 2 days plus
the oneDNN build plumbing.

**Depends on / conflicts with.** Independent of P1, P2. With P3, the group issue step would call `f16_dnnl` per
expert for `cnt >= 128` and batch the rest (two issue paths per group); do P3's bench before deciding, because a good
group batch may make P4 unnecessary. T2c builds on P4's plumbing (same engine, stream, primitive cache) and is the
reason to do the plumbing well.

---

### P5. Port the chunked WY DeltaNet recurrence without the 128-SM gate

**Goal and measured motivation.** The GDN recurrence walks every token serially per value head: 192 work-groups of 128
lanes (`HV * NCB = 48 x 4`), with 4 barriers per token (`gdn_rec_cols_kernel`, `gdn_rec_cols_pipe_kernel`). It is
4.0% of the prompt at 2,184 tokens, 6.4% at 8,000, 7.0% at 40K ("DeltaNet recurrence", author's INTEL_PERFORMANCE.md),
9% of a 4K IQ3_S prompt as "GDN" (INTEL.md), 4.8 s of an 80K prompt. Upstream has the recurrence in the chunked WY
form of the gated delta rule (`STRATA_GDN_CHUNKED=1`, CUDA only): on an RTX 5090 1.20x at 128 tokens, 1.67x at 512,
1.81x at 2048+ against `gdn_rec_kh_kernel` (`src/prefill/kernels.cu:726-728`). The SYCL copy has no port of it, and
the port's key-head kernel (`STRATA_GDN_KEYHEAD=1`) measured 8% slower (INTEL.md: 727 vs 788 tok/s), so the SYCL
default is the column-split pipelined kernel. Expectation: 4 to 5% of prompt time at 4K (the review); the risk is
the scan's grid of 128 work-groups on 32 Xe cores.

**Where.**
- SYCL copy, `sycl/src/prefill/kernels.dp.cpp`: `:23-24` the constants (`S = 128, HK = 16, HV = 48, C = 10240`),
  `:626` `RG = 4, RPG = S / RG`, `:761` `CB = 32, NCB = S / CB`; `:768-866` `gdn_rec_cols_kernel`; `:876-993`
  `gdn_rec_cols_pipe_kernel`; `:1003-1008` the `STRATA_GDN_CP_ASYNC 0` fixup ("SYCL port: plain copies (no
  cp.async)"); `:1070-1225` `gdn_rec_kh_kernel`; `:1228-1234` `gdn_keyhead_ok` (SYCL: on only with
  `STRATA_GDN_KEYHEAD=1`); `:2566-2698` `gdn_recurrence_variant` with the launcher at `:2613-2679` and the output
  norm at `:2680-2690`; `:2699` `gdn_recurrence`.
- Upstream, `src/prefill/kernels.cu`: `:714-738` the method and the constants (`kGdnChunkedMin = 128`, `GCH = 32`,
  `GSB = 2048`, `GDV = 16`, `GKP = S + 4`, `GTP = GCH + 1`, `kChunkPrepSmem`, `kChunkScanSmem`); `:740-826`
  `gdn_chunk_prep_kernel` (256 threads, grid `(nch, HK)`, dynamic shared `kChunkPrepSmem` = 41,472 B); `:839-997`
  `gdn_chunk_scan_kernel` (`128 * VPK = 384` threads, grid `(HK, S / GDV) = (16, 8)` = 128 blocks, dynamic shared
  `kChunkScanSmem` = 93,696 B); `:1003-1015` `gdn_chunk_scratch` (25.6 MB per device, allocated once); `:1022-1063`
  `gdn_rec_chunked` with the gate at `:1039` (`cc_major < 8 || sms < HK * (S / GDV) || smem < kChunkScanSmem`;
  `STRATA_GDN_CHUNKED=2` skips the SM-count test); `:1980-1992` the launcher's dispatch.
- Caller: `sycl/src/prefill/prefill.cpp:2862-2876` (`kPfGdn`, `kPfGdnConv`, `kPfGdnRec`, `kPfGdnOut` marks around
  the GDN block). `include/strata/prefill/kernels.hpp:61` `gdn_recurrence`.
- Test: `src/prefill/gdn_rec_parity.cu` (CUDA only, top-level `CMakeLists.txt:761-763`; `sycl/CMakeLists.txt:261`
  says why it is not built in the port: PTX timers and `__nanosleep` in its SM-holding bench).

**Current behaviour.** The SYCL launcher has no chunked branch:

```cpp
// kernels.dp.cpp:2634-2636 and 2652-2679 (abridged)
static const bool pipe = [] { const char* v = std::getenv("STRATA_GDN_PIPELINE"); return v == nullptr || std::atoi(v) != 0; }();
if (pipe && gdn_keyhead_ok())   // the value heads of a key head in one thread (same bits)
    ... gdn_rec_kh_kernel(state, h, gate, beta, y, T) over HK * NCB groups of (RG, CB)
else if (pipe)                 // the software-pipelined loads (same bits)
    ... gdn_rec_cols_pipe_kernel(state, h, gate, beta, y, T) over HV * NCB groups of (RG, CB)
else
    ... gdn_rec_cols_kernel(state, h, gate, beta, y, T)
... gdn_out_norm_kernel over (HV, T) groups of S
```

Upstream's dispatch that the port must mirror:

```cpp
// src/prefill/kernels.cu:1980-1985
static const bool chunked = [] { const char* v = std::getenv("STRATA_GDN_CHUNKED"); return v != nullptr && std::atoi(v) != 0; }();
if (chunked && T >= kGdnChunkedMin && gdn_rec_chunked(state, h, gate, beta, y, T, (cudaStream_t) stream) == cudaSuccess) {
} else if (pipe && gdn_keyhead_ok())
    gdn_rec_kh_kernel<<<HK * NCB, dim3(CB, RG), 0, (cudaStream_t) stream>>>(state, h, gate, beta, y, T);
```

The per-token kernel's inner step, for scale (every token, every column, 4 barriers):

```cpp
// kernels.dp.cpp:815-838
const float g = sycl::native::exp(gate[t * HV + head]);
float kv = 0.0f;
for (int r = 0; r < RPG; ++r) kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
red[rg][c] = kv;
item_ct1.barrier();
const float kv_col = red[0][c] + red[1][c] + red[2][c] + red[3][c];
const float delta = (ht[2 * HK * S + head * S + col] - g * kv_col) * beta[t * HV + head];
```

**Change.**
1. Port `gdn_chunk_prep_kernel` and `gdn_chunk_scan_kernel` by hand into `kernels.dp.cpp` next to `gdn_rec_kh_kernel`
   (dpct does not run on this tree for a single kernel; the port's other hand ports are `verify.cpp`, `mtp.cpp`).
   Translation rules, all with precedents in the file: `extern __shared__` dynamic arrays become one
   `sycl::local_accessor<float, 1>` of `kChunkPrepSmem / 4` or `kChunkScanSmem / 4` floats passed to the kernel (not
   `group_local_memory_for_overwrite`, which is for fixed sizes); `__syncthreads()` -> `item.barrier()`;
   `__shfl_up_sync` / `__shfl_sync` -> `dpct::experimental::shift_sub_group_right` / `select_from_sub_group` as the
   file's `warp_sum` uses `permute_sub_group_by_xor` (`kernels.dp.cpp:26-38`); `__syncwarp()` ->
   `sycl::group_barrier(sg)`; `float4` -> `sycl::float4` with `.x() .y() .z() .w()`; `__ldg` -> plain loads;
   `expf` -> `sycl::exp` (not `sycl::native::exp`: the kernel's correctness rests on `exp(gamma_t - gamma_i)` for
   exponents <= 0, keep full precision; the per-token kernel uses `native::exp` for `g`, a different quantity);
   `cp.async` (`gdn_cp16`, `gdn_pf_l2`, `gdn_cp_commit`, `gdn_cp_wait_*`) -> plain copies and no-ops, exactly as the
   port did for `gdn_rec_kh_kernel` (`:1003-1030`). `__launch_bounds__(128 * VPK)` -> `[[sycl::reqd_work_group_size(1,
   1, 384)]]`; `[[sycl::reqd_sub_group_size(32)]]` on both (the prep's "a lane per token" and the `warp + 8m` row
   mapping assume 32).
2. Scratch: `gdn_chunk_scratch()` becomes a `sycl::malloc_device` of `(GSB / GCH) * HV * (2 * GCH * GCH + GCH)` floats
   (25.6 MB) once per device, kept (upstream's comment at `:999-1002` on why: a per-call allocation cost 78 vs 34 ms
   on a 2K prompt). Take it from the prompt's device buffers instead if the 25.6 MB must be counted against the
   loan: add it to the count at `prefill.cpp:1835-1884` and pass the pointer through `gdn_recurrence`'s `stream`
   side channel or a port-only setter. Simplest first: a static per-device allocation, 25.6 MB, stated in the PR.
3. `gdn_rec_chunked` on SYCL: drop the compute-capability and SM-count tests; keep the shared-memory test as
   `q.get_device().get_info<sycl::info::device::local_mem_size>() >= kChunkScanSmem` (Xe2 reports 128 KB on the B70
   class per INTEL.md's "120 KB of local memory allows one work-group per Xe core"; not verified by this review, the
   check protects the A-series). Launch the prep over `nd_range(range(1, HK, nch) * range(1, 1, 256), range(1, 1,
   256))` and the scan over `nd_range(range(1, S / GDV, HK) * range(1, 1, 384), range(1, 1, 384))`, per super-block
   of `GSB` tokens as upstream (`:1057-1061`).
4. The dispatch in `gdn_recurrence_variant` at `kernels.dp.cpp:2634`: insert upstream's `chunked && T >=
   kGdnChunkedMin && gdn_rec_chunked(...)` branch before the key-head test, the same env name `STRATA_GDN_CHUNKED`
   (1 on; default off), `STRATA_GDN_CHUNKED_MIN=<tokens>` as a port-only override of `kGdnChunkedMin` for the
   crossover search (upstream's 128 was measured on a 5090).
5. The waves problem, which is the real design question on this card: the scan has 128 work-groups of 384 lanes
   each with 93.7 KB of local memory, so one per Xe core; on 32 cores that is 4 waves, and each work-group walks every
   chunk of the super-block, so the kernel time is about 4x one work-group's walk. Upstream's gate refuses exactly
   this (`:689-695` and `:1019-1021`: "a second wave would double the time"). The per-token kernel it replaces runs
   192 work-groups of 128 lanes with 1.5 KB of local memory each, so several per core, and its time is one walk of T
   tokens with 4 barriers each. Whether 4 waves of the chunked scan beat 1 wave of the serial walk is the measurement;
   the arithmetic favours the chunked form (its per-token cost is a 32-wide register-tiled matmul instead of 4
   barriers), the wave count works against it. Two re-tilings to try if 4 waves lose, both local to the scan kernel:
   (a) `GDV = 32` (64 work-groups, 2 waves; `s[4][4]` per thread becomes `s[4][8]` or the threads per head double to
   256, local memory grows by `VPK * (S + 3 * GCH) * 16 * 4` = 43 KB to about 137 KB, which does not fit, so it must
   be the register route: 8 columns per thread, about 32 more live floats, check for spills with
   `IGC_ShaderDumpEnable` or the `-fsycl-device-code-split` build's `.spv` disassembly; not verified); (b) `GDV = 64`
   (32 work-groups, 1 wave on the B70, 16 columns per thread: too many registers at SIMD32, probably needs SIMD16
   for this kernel alone, which the file's sub-group-32 reductions in the prep would not share). Spec (a) as the
   second step, (b) as a note.
6. Mirrors upstream exactly except the gate; the arithmetic order inside each kernel must be kept line for line so
   that the CUDA `gdn_rec_parity`'s tolerance (1e-4 of the largest value against `gdn_rec_kh_kernel`, and against an
   FP64 reference up to 8192 tokens, `gdn_rec_parity.cu:8-11`) carries over.

**Must not change.** Rounding order changes (upstream's own words at `:721`: "the same state and outputs as the
token-by-token kernels up to FP32 rounding (another order of the sums)"): same text, near-tie flips allowed; the
tolerance test below is the gate. The output norm (`gdn_out_norm_kernel`) and the per-token kernels stay as they are
and remain the default. The state layout `state[(row * HV + head) * S + col]` and the `y` layout `[T][HV][S]` are
shared with the decode path's GDN kernels (`strata/kernels/gdn.hpp`), so the scan's final state write-back
(`kernels.cu:993-996`) must land in the same places. VRAM: 25.6 MB of scratch (state the number); local memory
93.7 KB per scan work-group.

**Tests.** Existing: `gdn_parity` (`sycl/src/kernels/gdn_parity.cpp`) tests the decode GDN kernels (conv, norms,
recurrence of `strata/kernels/gdn.hpp`), not the prompt recurrence. New: port `src/prefill/gdn_rec_parity.cu` to
`sycl/src/prefill/gdn_rec_parity.dp.cpp` with the bench half stubbed (the PTX timers and `__nanosleep` of its
"fewer_sms" mode have no SYCL counterpart, `sycl/CMakeLists.txt:261`): the bit checks between `gdn_rec_kh_kernel`
and `gdn_rec_cols_pipe_kernel` (T from 1 to 4099, non-zero initial state) and the 1e-4 check of the chunked
recurrence against `gdn_rec_kh_kernel` at T = 8192 and 32768, plus the FP64 reference up to 8192, exactly as the
file's header states. Register it as `gdn_rec_parity` in `sycl/CMakeLists.txt`. Then the output-identity run on the
three standard prompts with `STRATA_GDN_CHUNKED=1`, outputs by eye and `llm-evals`. `STRATA_VERIFY_DEBUG=1`'s
residual ladder should show the first divergence at a GDN layer and stay small (1e-4 class), not NaN.

**Bench.** The ported `gdn_rec_parity --bench` timing mode (keep only the alternating-launch timing at T = 2048,
8192, 32768, one layer, all 48 value heads, drop the SM-holding modes): device ms of the per-token kernel against
the chunked pair, which gives the crossover for `STRATA_GDN_CHUNKED_MIN`. `unitrace -d`: `gdn_chunk_prep_kernel` +
`gdn_chunk_scan_kernel` against `gdn_rec_cols_pipe_kernel` per layer. `STRATA_PREFILL_TIMING=1` once: "gdn
recurrence" phase ms at 2,185, 8,000 and 40,000 tokens. Timer off: tok/s at the three sizes, interleaved pairs,
medians.

**Expected gain, risk, effort.** If the 4-wave scan still beats the serial walk: 4 to 5% of prompt time at 4K, 5 to
6% at 40K (the phase is 7% there). If it does not, the GDV = 32 re-tiling is another day and may still lose; then the
item is a documented negative like `STRATA_GDN_KEYHEAD`. Medium risk: the FP32 order change needs the tolerance test
and the eyeball gate; the hand translation of two 250-line kernels with shuffles and dynamic shared memory is where
bugs hide, and the parity test is the only net. 2 to 3 days, 4 with the re-tiling.

**Depends on / conflicts with.** Independent of P1 to P4, P6. Conflicts with nothing on the author's branch (his
TODO has no GDN item). Upstream PR #413's per-key-head kernel is the `gdn_rec_kh_kernel` the port already carries.

---

### P6. Fuse the two FP32 passes over the residual R: the hyper-connection read

**Goal and measured motivation.** The hyper-connection read of each half runs `gr_norm_rs` (reads R, writes the
BF16 image `xn16` and the row scales `rs`), the down GEMM (K 10240 to 320), SiLU, the up GEMM (320 to 10240, writes
`gated` FP32), the inject GEMM, then `gr_mix_r` (reads R again, `rs`, the norm weights and `gated`, writes `mixed`
FP32 plus its BF16 and FP16 images). Per token per half that is about 40 KB of R read twice, 40 KB of `gated`
written and read, 20 KB of `xn16` written and read, and 20 KB of `mixed*` written: about 220 KB, 96 halves per
token, about 86 GB per 4,096-token chunk, 0.14 s at 608 GB/s against a chunk of about 4 s: the "hyper-connection
reads" phase at 4.6 to 5.7% of the prompt (INTEL.md 4K IQ3_S; author's table at 2,184 / 8,000 / 40K: 4.9 / 5.7 /
5.7%). The write-back side is already fused (F-2: `gr_write_norm_rs_kernel` writes R and norms it for the next half
in one pass, `kernels.dp.cpp:219-250`, used at `prefill.cpp:4234-4238`), so the second read of R is now on the
mix side. The HIP build removes the `gated` round trip with `gr_upmix_kernel` (the up GEMM's epilogue does the mix,
`src/prefill/kernels.cu:1749-1795`); on SYCL `gr_upmix` returns false (`kernels.dp.cpp:2303-2322`). Expectation:
about 2% of prompt time (the review): the `gated` write and read plus the second read of R are 120 of the 220 KB.

**Where.**
- `sycl/src/prefill/kernels.dp.cpp:160-186` `gr_norm_rs_kernel`; `:187-214` `gr_mix_r_kernel`; `:2260-2284`
  `gr_norm_rs` (launch: `T * HC` groups of 256); `:2285-2302` `gr_mix_r` (launch: `blocks_for(T * N)` groups of 256,
  one element per lane); `:2303-2322` `gr_upmix` (HIP body, SYCL stub returning false); `:219-250`
  `gr_write_norm_rs_kernel` (F-2, the model for "same bits" fusion in this file).
- `sycl/src/prefill/prefill.cpp:2779-2852` the hyper-connection read of one half (the `gr_norm_rs` call at 2784, the
  down projection 2788-2801, `gr_silu` 2802, the `gr_upmix` attempt 2803-2843 behind `hc_upmix()`, the up GEMM 2844,
  inject 2845, `gr_mix_r` 2850); `:1137-1141` the buffers (`m.gated = o.take<float>(T * D)`, `m.mixed`,
  `m.mixed_bf`, `m.mixed_h`).
- `include/strata/prefill/kernels.hpp:21, 24, 30` the declarations of `gr_norm_rs`, `gr_mix_r`, `gr_upmix`.
- Upstream `src/prefill/kernels.cu:1741-1795` `gr_upmix_kernel` (gfx11 WMMA: a 128-token x 16-column block, the four
  streams' `w_up` rows staged in LDS once, the mix in the epilogue "in gr_mix_r_kernel's order").

**Current behaviour.**

```cpp
// kernels.dp.cpp:197-213 (gr_mix_r_kernel)
if (i >= T * N) return;
const int64_t t = i / N, d = i % N;
float s = 0.0f;
for (int c = 0; c < HC; ++c) {
    const int64_t j = t * D + c * N + d;
    const float x = R[j] * rs[t * HC + c] * w[c * N + d];   // gr_norm_kernel's value, bit for bit
    s = sycl::fma((float)x, sigm(g[j]), s);
}
s /= (float) HC;
mixed[i] = s;
if (mixed16) { const uint16_t h = act16(s); mixed16[i] = h; if (mixed16_lo) mixed16_lo[i] = bf_lo(s, h); }
if (mixed_h) mixed_h[i] = hf(s);
```
```cpp
// prefill.cpp:2844-2851
if (!upmixed && !bf16_proj(m.gemm, wu, m.lo16, m.gated, T, su, err, 0, m.lo16_lo)) return false;
if (!hcd && !bf16_proj(m.gemm, wi, m.xn16, m.inj, T, si, err, 0, m.xn16_lo, ldx != D ? ldx : 0)) return false;
...
} else {
    gr_mix_r(m.R, m.grs, (const float*) wn->data, m.gated, m.mixed, m.mixed_bf, T, m.cs, m.mixed_h, m.mixed_bf_lo);
}
```

What cannot be fused: `gr_norm_rs` and `gr_mix_r` are separated by the down, up and inject GEMMs, and `gr_mix_r`
needs `gated`, so the two kernels cannot become one launch. What can: the `gated` round trip, by computing the up
projection inside the mix kernel (the HIP route), and the second read of R, by reading the BF16 image instead
(changes bits: rejected below).

**Change.**
1. Port `gr_upmix` to SYCL as a plain FP32 kernel first, not a joint_matrix one: per (token t, column d) the up
   projection is `gated[t][c * N + d] = sum_k lo16[t][k] * w_up[(c * N + d) * LR + k]` over `LR = 320` BF16 products for
   each of the `HC = 4` streams, then the mix exactly as `gr_mix_r_kernel` (lines 199-213, same order). Work shape as
   upstream (`UM_T = 128` tokens x `UM_D = 16` columns per work-group, `kernels.cu:1741`): stage the block's `w_up`
   rows (4 streams x 16 columns x 320 BF16 = 40 KB) in local memory once, each lane owns (token, column) pairs and
   accumulates 4 dots of 320 in FP32 registers. Arithmetic: 4,096 tokens x 10,240 x 320 x 2 = 27 GFLOP per half, so
   a 4K chunk's 96 halves are 2.6 TFLOP in FP32 ALU, about 0.1 s at the B70's FP32 rate (E, from 32 cores x 128 lanes
   x 2 x about 2.5 GHz = 20 TFLOP/s peak, half of it reached), which is the same order as the 0.14 s of traffic the
   whole phase costs today. So the plain kernel is a wash or a loss and exists to validate the data path; the gain
   needs the matrix engine (step 2). Switch `STRATA_HC_UPMIX=1` (`hc_upmix()` already exists in the caller at
   2804, with `STRATA_HC_UPMIX_CHECK=<n>` comparing against the default pair for n layers and printing the relative
   RMS: reuse both, `prefill.cpp:2806-2838`).
2. The XMX version: the up projection as a `joint_matrix` product per (16 tokens x 16 columns) tile with K = 320 in
   steps of 16, A = `lo16` rows (BF16, row-major, ld LR), B = the staged `w_up` tile, FP32 accumulator, then the same
   epilogue. The port's two joint_matrix kernels (`xmx_gemm_iq`, `qsa_prompt_attn_xmx`) are correct but lose to
   oneMKL on this card (INTEL.md "XMX"); here the comparison is not against oneMKL's GEMM alone but against GEMM +
   40 KB/token written and read, so the bar is lower. Only build step 2 if step 1's traffic saving shows in the
   phase timer with the ALU cost still hiding it (that is, if step 1 is within 1.2x of the default pair).
3. Alternative that stays bitwise and is cheap, as a separate switch: `gr_mix_r` reads `gated` through a
   `sycl::vec<float, 4>` per lane (4 columns per lane: `blocks_for(T * N / 4)` groups) and R likewise, cutting the
   kernel's instruction count; the sums per element are unchanged. Expected: a few % of the kernel, not of the
   prompt. `STRATA_GR_MIX_VEC4=1`.
4. Rejected: reading `xn16` (BF16) instead of R in the mix (saves 20 KB/token but changes `x` from the FP32 product
   to its BF16 rounding, a real precision loss on the residual stream, not a reorder).
5. Where: the kernels in `kernels.dp.cpp` beside `gr_upmix` (replace the stub body under `#else` at 2317-2321 with
   the SYCL launch under the switch); no new files. Mirrors `src/prefill/kernels.cu:1749-1795` with the WMMA
   intrinsics replaced.

**Must not change.** Step 1 and 2: rounding order changes in `gated` (the up GEMM's FP32 sums in a different order
than oneMKL's), same text, near-tie flips allowed; the mix epilogue itself is bitwise given `gated`
(`STRATA_HC_UPMIX_CHECK` prints the relative RMS of `mixed`, expect 1e-6 class). Step 3: bitwise. `m.gated` stays
allocated (the default path and the inject GEMM's neighbours do not move; `prefill.cpp:1138`). No new VRAM beyond
40 KB of local memory per work-group. `gr_norm_rs` is untouched; `xn16`'s padded stride `ldx` (`prefill.cpp:2781`)
is not involved in the mix.

**Tests.** Existing: `gr_parity` and `gr_multi_parity` (`sycl/CMakeLists.txt:227, 252`) cover the decode
hyper-connection kernels (`strata/kernels/gr.hpp`), not these prompt kernels; `gr_bench` (`:288-289`) times the
decode read. New: (a) `STRATA_HC_UPMIX_CHECK=48` on the 2,184-token prompt (every layer, both halves, relative RMS
of `mixed` printed per call; gate: under 1e-5 for step 1, which is FP32 sums of BF16 products in another order);
(b) a `gr_mix_parity` (new, small): random R, rs, w, gated for T = 257, `gr_mix_r` against the vec4 variant bitwise
and against a CPU double-precision mix within 1e-6; (c) output identity on the three standard prompts for step 3,
eyeball and `llm-evals` for steps 1 and 2.

**Bench.** `STRATA_PREFILL_TIMING=1`: "hc read" phase ms at 2,185, 8,000 and 40,000 tokens with each switch;
`unitrace -d`: `gr_mix_r_kernel` + the up GEMM's oneMKL kernel against the fused kernel per half. Timer off: tok/s
at the three sizes, interleaved pairs, medians.

**Expected gain, risk, effort.** Step 3: under 1% of prompt. Step 1: 0 to 1% (ALU bound). Step 2: about 2% if the
joint_matrix tile keeps up with the saved traffic, which on this card is not a given. Low risk for correctness
(the check switch exists); the risk is spending the time for a wash, so stop after step 1's measurement if it is a
loss. 1 to 2 days for steps 1 and 3; step 2 another 2.

**Depends on / conflicts with.** Independent of P1 to P5. Overlaps the author's TODO 4 "fused small kernels for
decode" only in spirit. If T2c or P4 bring oneDNN in, a oneDNN matmul with a binary post-op cannot express the
4-stream sigmoid mix, so this stays a hand kernel.

---

### T2c. Tier 2: INT8 expert weights with per-block scales through oneDNN matmul (f16 source, s8 weights, grouped scales)

**Goal and measured motivation.** The prompt expert path dequantizes every routed expert to FP16 (9.83 MB written,
then read by the GEMMs) and the GEMM at M of about 80 rows is bound by reading those weights (28 to 32 TFLOP/s at 256
to 512 rows, less below; author's bench). The i-quant codes are small integers times a block scale, so an int8 image
plus one scale per block is exact and half the bytes: 4.9 MB per expert. oneDNN's matmul can take f16 activations
and s8 weights with scales grouped along K and decompress inside the kernel ("weights decompression" matmul; the
author's bench measured oneDNN int8 x int8 with per-tensor and per-column scales at 1.6 to 2.4x of oneMKL f16 on
dense shapes and 1.3 to 3.2x on expert shapes, INTEL_PERFORMANCE.md 762, which is a different primitive: the f16 x s8
decompressing form runs the f16 DPAS rate and gains from bytes, not from int8 math). Expectation (the review): about
20 points of 4K prompt time, about 1.25x (E), from dequant 20% + GEMMs 34% at IQ3_S 4K (INTEL.md) both halving their
weight bytes. The author's own open item (TODO.md 4: "int8 for the dense projections ... Strata's earlier int8 DPAS
kernel lost because it rescaled after every 32-element block inside the GEMM; this rescales once").

**Where.**
- `sycl/src/kernels/cuda/iq_kernels.dp.cpp:2723-2753` `dq_dispatch` and the per-type bodies: IQ2_XS `:2403-2415`,
  IQ2_S `:2417`, IQ3_S `:2450-2466`, IQ1_M `:2468-2485`, IQ4_NL `:2487-2500`, IQ2_XXS `:2387-2401`, IQ3_XXS
  `:2430-2448` (scale lines at 2395 and 2439); `:2755-2774` the two dequant kernels; `:4292-4317`, `:4390-4418` the
  entry points. `third_party/ggml/ggml-common.h:381-451` the block layouts (`block_iq2_xs` scales `[QK_K/32]` bytes of
  two nibbles; `block_iq1_m` scales `[QK_K/32]` of 3-bit fields; `block_iq3_s` scales `[QK_K/64]` of two nibbles;
  `block_iq4_nl` 32 values per `d`).
- `sycl/src/prefill/prefill.cpp:3964-3984` the FP16 path (to be mirrored by an int8 path), `:226, 1190, 1868` the
  ring, `:3978-3983` the GEMM calls.
- `sycl/src/prefill/gemm.dp.cpp:773-808` `Gemm::f16`; P4's `f16_dnnl` plumbing (engine, stream, primitive cache).
- `sycl/include/dpct/dnnl_utils.hpp`: no matmul wrapper; use oneDNN's API directly (`oneapi/dnnl/dnnl.hpp`,
  `oneapi/dnnl/dnnl_sycl.hpp`, present on the VM with `intel-oneapi-dnnl-devel`, not in the dev image).
- Author's `onednn_gemm_bench.cpp:84-101`: the matmul construction (`primitive_attr`, `set_scales_mask(DNNL_ARG_SRC,
  0)`, `set_scales_mask(DNNL_ARG_WEIGHTS, 1 << 1)`, the `DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS` argument as a
  `{N}` f32 memory). It does not use grouped scales; the grouped form is `primitive_attr::set_scales(int arg, int
  mask, const memory::dims& groups, memory::data_type dt)` in oneDNN 3.5+ (check the header: the exact overload and
  whether `dt` may be f16).

**Current behaviour.** The scale granularity per format, read from the dequant bodies (this matters: the review says
"per-32 scales", the code says per 16 for two of the formats in use):

```cpp
// IQ1_M (the Coder's gate/up), iq_kernels.dp.cpp:2475-2476: ib16 indexes 16-value halves, so one scale per 16 values
const int64_t ib16 = 2 * ib + il / 2;
const float d = (float) scale.f16 * (2 * ((sc[ib16 / 4] >> 3 * (ib16 % 4)) & 0x7) + 1);
const float delta = x[ibs].qh[2 * ib + il / 2] & (0x08 << 4 * (il % 2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
... y[j] = cvt<dst_t>(d * (q[j] + delta));
```
```cpp
// IQ2_XS, :2409: the nibble is selected by il / 2, one scale per 16 values
const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
// IQ3_S, :2457: one scale per 32 values (ib indexes 32-value blocks)
const float d = (float) x[ibs].d * (1 + 2 * ((x[ibs].scales[ib / 2] >> 4 * (ib % 2)) & 0xf));
// IQ4_NL (the Coder's down in 39 of 48 layers), :2493: d per 32-value block
const float d = (float) x[ib].d;
```

So the exact integer decomposition is: IQ1_M `code = 8 * (q - 1) + (delta < -1 ? -1 : +1)` in {-9, -7, -1, 1, 7, 9}
with scale `scale.f16 * (2 s + 1) / 8` per 16 values (the delta sign varies per 8 values but is folded into the code);
IQ2_XS `code = grid[j] * sign` with grid bytes in {8, 25, 43} (`iq2xs_grid`), scale `d * (2 s + 1) / 8` per 16;
IQ2_XXS the same codes, scale `d * (2 s + 1) / 8` per 32 (`aux32 >> 28`); IQ3_XXS codes `iq3xxs_grid` bytes
(4 to 62) with scale `d * (2 s + 1) / 4` per 32; IQ3_S codes 1 to 15 with sign, scale `d * (1 + 2 s)` per 32;
IQ4_NL codes `kvalues_iq4nl` (-127 to 113), scale `d` per 32; IQ2_S like IQ2_XS, per 16. Every code fits int8; no
code set can be rescaled to a coarser group exactly (odd multipliers up to 15 push 43 x 15 past 127). Therefore the
oneDNN group must be 16 along K for IQ1_M, IQ2_XS and IQ2_S, and 32 for the rest; the probe tests both.

**Change (design to start; open questions flagged).**
1. Probe first (half a day, on the VM, about 50 lines, modelled on the author's bench): build with the bench's icpx
   line plus `-I/opt/intel/oneapi/dnnl/latest/include -L.../lib -ldnnl`. Create `src_md({M, K}, f16, ab)`,
   `wei_md({K, N}, s8, ba)` (so `W[N, K]` row-major int8 as our writer lays it out), `dst_md({M, N}, f32, ab)`;
   `primitive_attr attr; attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1), {G, 1}, memory::data_type::f32);
   attr.set_fpmath_mode(fpmath_mode::f16, true);` (the second argument "apply to int" is what turns on weight
   decompression in oneDNN 3.5+; check the header for the exact spelling), `matmul::primitive_desc pd(eng, src_md,
   wei_md, dst_md, attr)` inside a try: a `dnnl::error` means unsupported for this `(G, dt)` on this device. Scales
   memory `{K / G, N}` f32 (also try f16). Run with M in {16, 80, 256}, K = 2560, N = 1280, G in {16, 32}, random
   int8 codes and scales, compare against a CPU double reference (max relative error, expect f16-rounding class
   about 1e-3 because the decompressed weights are rounded to f16 before the DPAS: see Must not change), and time it
   against the author's oneMKL f16 line at the same M. Also run `ONEDNN_VERBOSE=1` once: the implementation name
   tells whether a real decompressing kernel ran (`jit:gemm:any` or `ocl:gemm_with_po:any`) or a reference path
   (`ref:any`, which would be 100x slower and kills the item). This probe is the gate for everything below; its
   output (supported groups, implementation name, ms per shape) goes in the PR.
2. Int8 writer kernels, `iq_dequant_gu_s8(type, gate, up, n_ff, n_embd, int8_t* dst, void* scales, int group,
   stream)` and `iq_dequant_s8(type, src, n, dst, scales, group, stream)` in `iq_kernels.dp.cpp`, same grid as the
   FP16 kernels (and P2's wide groups), one `dqi_*` body per format mirroring `dq_*` but emitting `int8_t code` per
   value and, from lane `il % (group / 8) == 0`, the block scale into `scales[(k / group) * ldn + row]` in the `{K / G,
   N}` layout oneDNN wants (row-major over K-blocks, N fastest; verify against the probe's working layout). Codes are
   exact integers, so the writer is bitwise testable against `code * scale == dq_*`'s float value (both in double).
   Layout of `dst`: the same interleaved `2 r + parity` rows for gate/up and row-major for down, int8, so `swiglu_il_kernel`
   and the row offsets are unchanged.
3. Ring: `m.dq_gu8[s]` (1280 x 2560 int8 = 3.28 MB) and scales (160 x 1280 f32 = 0.82 MB at G = 16, 0.41 MB at G =
   32), `m.dq_d8[s]` (2560 x 640 = 1.64 MB) and scales (40 x 2560 f32 = 0.41 MB at 16): about 6.1 MB per expert at
   G = 16 against 9.83 MB FP16. Allocate beside the FP16 ring (both exist while the switch is an A/B) at
   `prefill.cpp:1190` and count at `:1868`; with P3's grouping of 32 experts that is 196 MB.
4. GEMM: `Gemm::f16_s8_grouped(X f16, W s8, scales, group, Y, T, N, K)` in `gemm.dp.cpp` on P4's oneDNN plumbing,
   primitive cached by `(T, N, K, G)`; arguments `DNNL_ARG_SRC`, `DNNL_ARG_WEIGHTS`, `DNNL_ARG_DST`,
   `DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS`. Called from the `compute` lambda in place of the two `m.gemm.f16` calls
   when `STRATA_PF_INT8W=1` and the layer's `(gu_type, d_type)` have writers; the group per matrix comes from the
   type (16 for 17, 22, 29; 32 for 16, 18, 20, 21). Layers whose types have no writer keep the FP16 path (as the
   port does today for `is_iq`'s list).
5. Switch: `STRATA_PF_INT8W=1` (default off), needs the `STRATA_SYCL_ONEDNN` build. `STRATA_DBG_NAN=1` should also
   dump the int8 images' scales when it reports GEMM inputs (`prefill.cpp` reports "the experts' fp16 GEMM inputs"
   today, INTEL.md).
6. Open questions, in the order the probe answers them: (a) does oneDNN 2026.1.1 on Xe2 accept `{16, 1}` and `{32,
   1}` weight-scale groups with s8 weights and f16 source, and with which implementation; (b) f16 or f32 scales
   (f16 halves the scale bytes; the per-16 scales for IQ1_M are already 25% of the int8 image at f32); (c) does the
   decompressing kernel reach the f16 DPAS rate at M = 80, or is the gain only the dequant's halved writes (the
   probe's ms against oneMKL f16 at the same M says); (d) how the decompressed weight is rounded (f16 before the
   DPAS, which is what the FP16 path stores anyway, so the products are the same numbers in a different order; or
   f32, which would be more exact than today); (e) whether a true int8 x int8 DPAS path with per-row activation
   scales is worth a second item (the author's TODO says the per-32 rescale inside the GEMM is what killed Strata's
   earlier kernel; with oneDNN the rescale would be oneDNN's problem, but s8 activations need a quantizer and are
   not exact). Mark (e) as a follow-up Plane item, not part of this.
7. Mirrors nothing upstream: upstream's int8 prompt path is the MMQ library (`moe_mmq.cu`, `moe_fused.cu`) built on
   llama.cpp's CUDA kernels, which the port does not compile.

**Must not change.** Rounding order changes, same text, near-tie flips allowed (a different GEMM kernel; if oneDNN
rounds the decompressed weight to f16 the products are today's numbers in another order, if to f32 they are slightly
more exact: either is the SELECT_GEMM class). The int8 image is exact: `code * scale` reproduces `dq_*`'s FP32 value
bit for bit before the FP16 rounding, and the writer test must prove it per format. VRAM: 6.1 MB per expert of ring at
G = 16 beside the FP16 ring's 9.83 MB while both exist (12.3 MB more than today at DQ = 2; 196 MB with P3's 32-expert
groups); state the bytes. Decode is untouched (the i-quant decode dots stay on the GGUF blocks in the expert cache;
nothing here changes what is resident).

**Tests.** New: (a) `iq_s8_parity`: for every supported type, random blocks, `code * scale` (double) equals the FP16
writer's pre-rounding value (double) exactly, and the FP16 writer's output equals `hf(code * scale)` bitwise; (b) the
probe as a ctest (`onednn_s8_probe`) that fails if `primitive_desc` throws for the groups the engine will use or if
the implementation name is a `ref` path; (c) `gemm_s8_parity`: the oneDNN grouped matmul against `Gemm::f16` on the
FP16 image of the same codes, max relative error under 2e-3 at K = 2560 (f16 products, f32 sums, two orders); (d)
output identity on the three standard prompts and the IQ2_XS pair, eyeball and `llm-evals`.

**Bench.** The probe's table first (ms per shape, oneMKL f16 vs oneDNN f16 vs oneDNN f16 x s8 grouped, M in {16, 80,
256, 512}). Then `STRATA_PREFILL_TIMING=1`: "dequant", "gemm gate/up", "gemm down" phase ms at 2,185, 8,000 and
40,000 tokens; `unitrace -d` for the writer kernels' GB/s (4.9 MB per expert now). Timer off: tok/s at the three
sizes and time to first token at 40K, interleaved pairs, medians; the 4K IQ3_S prompt from INTEL.md's 2026-10-07
table (980 to 1,002 tok/s with `--prefill 4096`) as the headline comparison, since that is where dequant + GEMMs are
54% of the time.

**Expected gain, risk, effort.** If the probe shows a real decompressing kernel at the f16 rate: dequant writes halve
(about 10 points of its 20%) and the weight-read-bound GEMMs speed up by up to 1.5x at low M (10 to 15 points of 34%):
about 1.2 to 1.3x on a 4K prompt (E, matching the review's 1.25x), less at 40K where attention and selection grow.
If oneDNN falls to a reference implementation for grouped s8 on Xe2, the gain is only the dequant's halved writes and
the item shrinks to P2-class (5%). Medium to high risk: the oneDNN feature's device support is unverified, the
per-16 requirement for IQ1_M and IQ2_XS was not in the review's framing, and the writer kernels are one `dqi_*` per
format (seven bodies). 5 to 8 days; 1 day for the probe decides whether to spend the rest.

**Depends on / conflicts with.** Needs P4's oneDNN build plumbing (do P4 first or fold its plumbing in). Composes
with P2 (the writers take the wide work-groups) and P3 (grouping and the ring bookkeeping; a oneDNN matmul has no
grouped-batch form with per-member M, so under P3 the int8 path issues one matmul per expert, which argues for doing
P3's oneMKL batch for the small experts and oneDNN for the large ones, as in P4). Replaces the FP16 dequant + oneMKL
path per layer when on; the author's `xmx_gemm_iq` (fused dequant + joint_matrix, 0.22x) and `xmx_int8_bench`
(int8 DPAS straight from IQ4_NL, 0.25 to 0.98x) are the two measured negatives this design avoids by letting oneDNN
own the kernel.

---

## What this review could not pin down (check on the card)

- Whether Xe2's work-group dispatch is the dequant kernel's bound (P2): the `dequant_bench` A/B at 32/128/256 lanes.
- Whether oneMKL's group `gemm_batch` with `group_count = G, group_size = 1` runs as one kernel (P3): `unitrace`
  kernel count on the step-7 bench.
- The share of routed rows in experts with 128+ rows at 4K and 40K chunks (P4): the histogram of step 1.
- Whether a 4-wave `gdn_chunk_scan_kernel` beats one wave of the serial recurrence on 32 cores (P5): the ported
  `gdn_rec_parity --bench`.
- Whether the plain FP32 `gr_upmix` is ALU bound on this card (P6): the "hc read" phase ms with the switch on.
- oneDNN 2026.1.1's support for s8 weights with `{16, 1}` / `{32, 1}` scale groups and f16 source on Xe2, the
  implementation it picks, and its rounding of the decompressed weight (T2c): the 50-line probe.
- The local memory size oneDNN and the SYCL runtime report for a B70 work-group (P5 assumes 128 KB from INTEL.md's
  120 KB attention kernel; `sycl-ls --verbose` or `get_info<local_mem_size>` confirms).

# Part D. Build, launch, bench and operations


Read-only specs for an engineer, written 2026-10-09 against `/home/user/strata` (fork of Niko1221/Strata at `fb58e0d`, engine 0.1.41, port under `sycl/`) and `/home/user/homelab`. Every `path:line` below was read in those checkouts; nothing was run. "Not verified" marks what could not be confirmed from the repos. Governing documents: `docs/research/strata-b70-flash-next-20261009-022842.md` sections 4 and 5 (house rules, the five steps) and `docs/research/strata-b70-engine-plan-20261009-025812.md` sections 2, 6 and 7.

Target machine: arc-llm VM (VM 800, `ubuntu@192.168.0.211`), Ubuntu, kernel 7.0, `xe` driver, oneAPI 2026.1.1 at `/opt/intel/oneapi`, no Docker, no `ocloc` yet, cmake 4.2.3 and ninja present, models under `/opt/llm/models`, 8 vCPUs, 48 GB RAM, the B70 and an RTX 5060 Ti passed through. The Strata checkout is assumed at `/home/ubuntu/src/strata` (the research doc's `~/src/strata`); replace throughout if another path is chosen.

Standing constraints that shape every section: new work on the VM is units, not containers (`infra/arc-llm-vm/AGENTS.md:11`, "Everything is a systemd unit (no Docker)"); one workload on the B70 at a time; never `setup.sh`, `update.sh` or the MCP installer on the VM (research doc section 4, supply chain); the engine binds `127.0.0.1` only; the API key is created by the passphrase tool and placed by stdin.

---

## I1. Native wrapper replacing `sycl/serve/strata-sycl.sh`, and `strata-flash.service`

### Where

- `sycl/serve/strata-sycl.sh:1-41` (the Docker wrapper this replaces).
- `sycl/serve/server_intel.py:1-16` (usage), `:289-294` (main: installs the xe reader, narrator, switcher, logprobs, then `serve.server.main()`).
- `serve/server.py:5613-5651` (argparse: `--engine`, `--config`, `--host`, `--port`, `--api-key`, ...), `:5709-5711` (how `exe` is resolved), `:2352-2369` (`child_env`: the config's `"env"` block goes into the engine's environment), `:4531-4534` (`/health`), `:4563` and `:4740-4754` (`/props`, needs the API key), `:5414-5424` (API key forms).
- `sycl/setup_intel.py:34` (`MOUNT`), `:92-100` (`sycl_path`: host path to `/work/...`), `:127-186` (`to_sycl`: what setup writes for an `xe` card).
- Engine env reads: `sycl/src/core/expert_source.cpp:2709` (`STRATA_VERIFY_DEVICE_PLAN`), `sycl/src/core/verify.cpp:1941` and `sycl/src/program/generate.cpp:4737` (`STRATA_VERIFY_NO_HOST`), `sycl/src/prefill/prefill.cpp:984` (`STRATA_STAGER_THREADS`), `sycl/src/platform/direct_file.cpp:61-64` (`STRATA_IO_THREADS`, default 16 on Linux).
- Homelab units: `infra/arc-llm-vm/systemd/qwen38.service:1-24`, `cyber-llm.service:1-23`, `shieldgemma.service:1-18`, `b70-power-cap.service:1-14`; deploy mapping `infra/arc-llm-vm/deploy.sh:27-30` (`systemd/` to `/etc/systemd/system/`, `sbin/` to `/usr/local/sbin/`).

### Current behaviour

`strata-sycl.sh` is the `exe` serve/server.py spawns for `--engine strata`. It needs Docker: it `exec docker run --rm -i ... --device /dev/dri --oom-score-adj 1000 -v "$root:/work"` with the engine's stdin/stdout attached (`:37-41`), so every path in the config's `args` is a container path under `/work`. Its environment rules (`:22-35`):

```
declare -A setting=([STRATA_VERIFY_DEVICE_PLAN]=1 [STRATA_VERIFY_NO_HOST]=1 [STRATA_STAGER_THREADS]=12)
...  STRATA_*|ONEAPI_*|UR_*|IGC_*|SYCL_*|ZES_*|NEOReadDebugKeys|OverrideDefaultFP64Settings) setting[$k]=${!k} ;;
...
    # the engine tests the switches by presence: a value of 0 (or empty) means "not set", so it is not passed on
    case "${setting[$k]}" in 0|"") ;; *) envs+=(-e "$k=${setting[$k]}") ;; esac
```

The engine confirms the presence rule: `expert_source.cpp:2709` takes DEVICE_PLAN as `v != nullptr && std::atoi(v) != 0`; `verify.cpp:1941` takes NO_HOST as `v && *v && std::strcmp(v, "0") != 0`. So "0" and unset are equivalent for the `STRATA_VERIFY_*` switches, which is why the wrapper may simply not export a zero.

`serve/server.py:2361-2362` copies the config's `"env"` block into the engine's environment (`for k, v in (cfg.get("env") or {}).items(): env[str(k)] = str(v)`), so a config can override any wrapper default the same way as today (INTEL.md:72-75).

`sycl/setup_intel.py:92-100` rewrites every path argument to the container's view (`"/work/" + p.relative_to(root)`) and fails if a path is outside `MOUNT` (`:97-100`); `to_sycl` (`:127-186`) also sets `"exe": str(SYCL_WRAPPER)` (the Docker script) and `"sycl_root"`. Nothing in the repo writes a host-path config for a native engine, and `setup_intel.py` must not run on the VM anyway (it goes through `setup.py`, which the supply-chain rule excludes). So the serve config is written by hand (I3 gives it).

`b70-power-cap.service` is `Type=oneshot`, `RemainAfterExit=yes`, `Environment=CAP_W=180`, `ExecStart=/usr/local/sbin/b70-power-cap.sh` (`:7-10`). The script itself is not in the homelab repo (only the unit; `find` for `b70-power-cap.sh` returns nothing), so the sysfs attribute is taken from `skills/llm-vm-ops/SKILL.md:46`: "read `power1_cap` at the hwmon whose `name` is `xe` (180000000 = 180 W)". Sentinel finds that hwmon the same way (`apps/sentinel/sentinel/collectors/llm.py:16`). If `b70-power-cap.sh` on the VM writes another attribute (`power1_max`), take the attribute name from the script; the check below reads the one the script writes.

### Change

1. Add `sycl/serve/strata-native.sh` to our fork (new file; `strata-sycl.sh` stays for Docker hosts and upstream). Full script:

```bash
#!/usr/bin/env bash
# strata-native.sh: the SYCL-built engine as a drop-in `exe` for serve/server.py (--engine strata) on a host that has
# oneAPI installed natively (no Docker). The serve protocol is lines on stdin/stdout, so the binary is exec'd with the
# pipes untouched; stderr goes to the server's log. Paths in the config's args are host paths.
#   STRATA_NATIVE_BIN   the engine binary                       (default: <repo>/build-sycl/strata, else build-sycl-aot/strata)
#   ONEAPI_ROOT         where setvars.sh lives                   (default: /opt/intel/oneapi)
# Defaults below are overridden by the environment (the config's "env" block arrives that way: serve/server.py
# child_env). The engine tests the STRATA_* switches by presence, so a value of 0 (or empty) means "not set" and the
# variable is removed (strata-sycl.sh:34-35). SYCL_CACHE_PERSISTENT is exported as a literal 0 on purpose: the
# persistent kernel cache segfaults on Xe2 (docs/INTEL_ARC.md), so it must be off whatever the toolchain's default.
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)                 # the repo
bin=${STRATA_NATIVE_BIN:-}
if [ -z "$bin" ]; then
    for b in "$here/build-sycl-aot/strata" "$here/build-sycl/strata"; do [ -x "$b" ] && { bin=$b; break; }; done
fi
[ -n "$bin" ] && [ -x "$bin" ] || { echo "strata-native.sh: no engine binary (build-sycl-aot/strata or build-sycl/strata; docs/INTEL_ARC.md, Build without Docker)" >&2; exit 2; }
# setvars.sh trips `set -u` (OCL_ICD_FILENAMES, docs/INTEL_ARC.md:147)
set +u; source "${ONEAPI_ROOT:-/opt/intel/oneapi}/setvars.sh" >/dev/null 2>&1 || { echo "strata-native.sh: setvars.sh failed" >&2; exit 2; }; set -u
declare -A setting=([STRATA_VERIFY_DEVICE_PLAN]=1 [STRATA_VERIFY_NO_HOST]=1 [STRATA_STAGER_THREADS]=8)
for k in "${!setting[@]}"; do [ -n "${!k+x}" ] && setting[$k]=${!k}; done
for k in "${!setting[@]}"; do
    case "${setting[$k]}" in 0|"") unset "$k" ;; *) export "$k=${setting[$k]}" ;; esac
done
export SYCL_CACHE_PERSISTENT=0                             # literal 0, not the presence rule (see above)
export ZES_ENABLE_SYSMAN="${ZES_ENABLE_SYSMAN:-1}"         # sysman: the Monitor tab's xe readings (sycl/serve/xe_telemetry.py)
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"   # the Intel card(s) under Level Zero only
echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true   # the OOM killer's first pick, as the Docker wrapper's --oom-score-adj 1000
exec "$bin" "$@"
```

   Why each variable:
   - `SYCL_CACHE_PERSISTENT=0`: the persistent JIT cache segfaults on Xe2 (research doc step 2; INTEL_ARC.md:149 says the native run exports it). Exported literally, not through the presence rule, because here "0" is the meaningful value.
   - `ZES_ENABLE_SYSMAN=1`: lets `sycl/serve/xe_telemetry.py` read the card through Level Zero sysman for the Monitor tab (server_intel.py:8, 37-48). Harmless for the engine.
   - `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`: the VM exposes two cards. The RTX 5060 Ti is not a Level Zero device, but without a selector the SYCL runtime also enumerates the OpenCL CPU device and the OpenCL view of the B70, and a two-backend device list is what the port's single-card assumptions trip on. `level_zero:gpu` keeps exactly the Intel GPU(s) under Level Zero; `perf_matrix.py:232` on the author's branch uses the same value for a layer split, and the research doc's `level_zero:0` is equivalent while one Intel card is present. Overridable through the environment or the config's `"env"`.
   - `STRATA_VERIFY_DEVICE_PLAN=1`, `STRATA_VERIFY_NO_HOST=1`: the port's run-time switches for an `xe` card (strata-sycl.sh:22-25; `NO_HOST` because the host-GPU flag stores are not visible on xe). Same defaults as the Docker wrapper.
   - `STRATA_STAGER_THREADS=8`: the Docker wrapper says 12 for the author's 12-thread hosts; the VM has 8 vCPUs. `prefill.cpp:984` reads it as the count of host copy threads for unpinned blobs. The engine plan (section 2) also suggests `STRATA_IO_THREADS=64` for the prompt path; leave that to the config's `"env"` for the bench, not the wrapper.
   - The presence rule is kept so `"env": {"STRATA_VERIFY_NO_HOST": "0"}` in a config still turns a switch off (INTEL.md:75).

2. The serve config's `exe` points at the wrapper by absolute path: `"exe": "/home/ubuntu/src/strata/sycl/serve/strata-native.sh"`; `serve/server.py:5711` makes a relative `exe` absolute against `cwd`, so an absolute path is the simplest. Every path in `args` is a host path (`/opt/llm/models/...`, `/opt/llm/strata/...`): nothing rewrites them because `setup_intel.py` is not used. Full config in I3.

3. Python for the server. `setup.sh` is excluded on the VM, so make the venv by hand from the pinned `requirements.txt` (`requirements.txt:1-20`: numpy, jinja2, regex, pyyaml, tqdm, requests, cmake, ninja, pillow, psutil, pinned; "Python 3.10+"): `uv venv --python 3.12 /home/ubuntu/src/strata/.venv && uv pip install --python /home/ubuntu/src/strata/.venv/bin/python -r /home/ubuntu/src/strata/requirements.txt`. Whether the pins install on 3.14 is not verified (numpy 2.5.3 is pinned for `>= 3.12`), so 3.12 is the safe pick; `setup.sh:24` itself picks `PY=3.12`.

4. Add `infra/arc-llm-vm/sbin/b70-power-cap-check.sh` (deployed to `/usr/local/sbin/` by `deploy.sh:30`):

```bash
#!/bin/bash
# ExecStartPre for the B70 units that must not start without the power cap: read the cap from the xe hwmon and fail
# unless it is exactly the wanted watts. b70-power-cap.service is a oneshot; Wants= alone does not prove it ran.
# Attribute: power1_cap at the hwmon whose name is xe (skills/llm-vm-ops/SKILL.md). If b70-power-cap.sh writes
# another attribute, change ATTR here to the one it writes.
set -u
want_w=${1:-180}; ATTR=power1_cap
for h in /sys/class/hwmon/hwmon*; do
  [ "$(cat "$h/name" 2>/dev/null)" = xe ] || continue
  cap=$(cat "$h/$ATTR" 2>/dev/null) || { echo "b70-power-cap-check: $h/$ATTR unreadable" >&2; exit 1; }
  [ "$cap" = "$((want_w * 1000000))" ] && exit 0
  echo "b70-power-cap-check: cap is $cap uW, want $((want_w * 1000000)) (${want_w} W); refusing to start" >&2; exit 1
done
echo "b70-power-cap-check: no xe hwmon found" >&2; exit 1
```

5. Add `infra/arc-llm-vm/systemd/strata-flash.service` (deployed by `deploy.sh:27`; not enabled at boot, like `cyber-llm.service:21`):

```ini
[Unit]
Description=Strata SYCL engine: Qwen3.8-Flash-Next IQ2_XS on the Arc Pro B70, 127.0.0.1:8097 (docs/research/strata-b70-flash-next-20261009-022842.md, step 5)
After=network-online.target b70-power-cap.service
Wants=network-online.target b70-power-cap.service
# One workload on the B70 at a time: every unit that can own the card. shieldgemma stays listed until it moves to its
# own Arc A380 (owner, 2026-10-09). cyber-llm itself conflicts only with qwen38, so without this line a cyber start
# would put two models on the card.
Conflicts=qwen38.service cyber-llm.service llama-swap.service llama-server.service llama-gemma.service shieldgemma.service
RequiresMountsFor=/opt/llm

[Service]
User=ubuntu
SupplementaryGroups=render
WorkingDirectory=/home/ubuntu/src/strata
Environment=MODEL_LOG_NAME=strata-flash
# STRATA_API_KEY=... only; 0600 root, placed by stdin, never printed (root AGENTS.md, Secrets). server.py reads it
# (serve/server.py:5631) and requires it on /v1/*, /props and /slots.
EnvironmentFile=/etc/homelab/secrets/strata-flash.env
# Fail closed: the cap must read 180 W before the engine touches the card.
ExecStartPre=/usr/local/sbin/b70-power-cap-check.sh 180
ExecStart=/home/ubuntu/src/strata/.venv/bin/python /home/ubuntu/src/strata/sycl/serve/server_intel.py \
  --engine strata --config /opt/llm/strata/strata-flash.json --host 127.0.0.1 --port 8097
# The engine pins the expert mirror and the KV (several GiB of locked host memory).
LimitMEMLOCK=infinity
# A runaway mirror kills Strata, never fast-llm (the 5060 Ti, kid chat).
OOMScoreAdjust=1000
MemoryMax=30G
# AOT start ~45 s, JIT ~90 s, plus the mirror fill; qwen38 and cyber-llm use the same bound.
TimeoutStartSec=300
TimeoutStopSec=60
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

   Notes: `MemoryMax=30G` is from the research doc's step 5 and bounds the pinned mirror plus the server; with 48 GB in the VM and fast-llm's ~15 GB (`skills/llm-vm-ops/SKILL.md:46`) it leaves headroom; the engine sizes its mirror from `MemAvailable` minus 4 GiB at start (engine plan, `STRATA_MIRROR_MIB`), so set `"env": {"STRATA_MIRROR_MIB": "<explicit>"}` in the config once the bench shows the real need. `Restart=on-failure` matches `qwen38.service:19`; Sentinel's `strata_heal` (I6) is a separate item, so until it exists the unit's own restart is the only automatic recovery. The engine's `ulimit -l` need is covered by `LimitMEMLOCK`. The secret file is per-stack like `/etc/homelab/secrets/<stack>.env` on docker-vm (root AGENTS.md, Secrets): `STRATA_API_KEY=<passphrase tool output>`, placed with `sudo install -m 0600 -o root -g root /dev/stdin /etc/homelab/secrets/strata-flash.env`; `User=ubuntu` cannot read a root 0600 file but systemd reads `EnvironmentFile=` as root before dropping privileges, so the mode stays 0600 root.

6. `deploy.sh` needs no change: `systemd/` and `sbin/` are already mapped (`deploy.sh:27,30`). Run `infra/arc-llm-vm/deploy.sh` (dry run), then `--apply`; it only daemon-reloads, never starts (`infra/arc-llm-vm/AGENTS.md:59`). Starting the unit is the Steward `llm-mode strata` path (I6), or by hand under an `llm:*` lease during the bench window.

### Tests

- Wrapper, on the VM without the card: `STRATA_NATIVE_BIN=/bin/true bash sycl/serve/strata-native.sh --help` exits 0; `STRATA_VERIFY_NO_HOST=0 STRATA_NATIVE_BIN=/usr/bin/env bash sycl/serve/strata-native.sh` prints an environment without `STRATA_VERIFY_NO_HOST` and with `STRATA_VERIFY_DEVICE_PLAN=1`, `SYCL_CACHE_PERSISTENT=0`, `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`. Add these two as a shell test `sycl/serve/test_strata_native.sh` (bash, no card) and, for the homelab side, a `node --test` file in the style of `infra/arc-llm-vm/tests/inference-power-vm.test.ts:15-25` that replaces `/sys/class/hwmon` with a temp tree and asserts `b70-power-cap-check.sh 180` exits 0 on `180000000` and 1 on `230000000` and on a missing xe hwmon.
- Unit: `systemd-analyze verify infra/arc-llm-vm/systemd/strata-flash.service` on the VM (dry); `systemctl show strata-flash -p Conflicts` lists the six units after `--apply`.
- Serve: `curl -s 127.0.0.1:8097/health` returns `{"status": "ok", ..., "service": "strata"}` (server.py:4532-4534; needs no key); `/props` with the key returns `default_generation_settings` (server.py:4754).

### Done when

The unit starts under a lease with `ExecStartPre` passing, refuses to start when the cap reads anything but 180 W, `qwen38` and `shieldgemma` are stopped by `Conflicts=` when it starts (and it is stopped when they start), `/health` answers on `127.0.0.1:8097` and nothing listens on `0.0.0.0`.

### Effort

0.5 day for the wrapper and unit, 0.5 day for the checks and tests.

---

## I2. Build recipe on the VM

### Where

`sycl/CMakeLists.txt:1-11` (header: a separate project; "cmake -S sycl -B build-sycl -G Ninja -DCMAKE_CXX_COMPILER=icpx"), `:23` (`STRATA_SYCL_AOT` cache variable), `:40-54` (AOT flags and the divide/sqrt rounding option), `:134-156` (`STRATA_GGML_DIR` and FetchContent), `:224-258` (`STRATA_SYCL_PARITY`, the parity test list), `:220` (`add_executable(strata ...)`). `sycl/tools/build.sh:4-12` (the flags the image build uses). `docs/INTEL_ARC.md:88-124` ("Build without Docker (Linux)"). `sycl/setup_intel.py:85-90` (`sycl_engine` looks for `build-sycl-aot/strata` then `build-sycl/strata`).

### Current behaviour

`sycl/CMakeLists.txt:134-156`:

```
set(STRATA_GGML_DIR "" CACHE PATH "llama.cpp checkout to take ggml from (empty: fetch the pinned commit)")
...
  if(STRATA_GGML_DIR)
    add_subdirectory(${STRATA_GGML_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
  else()
    include(FetchContent)
    FetchContent_Declare(strata_llamacpp
      GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
      GIT_TAG 3cf03257f219afbe7334045ff7c6a06ac68c627d
      GIT_SHALLOW FALSE)
```

Without `STRATA_GGML_DIR`, configure clones the full llama.cpp history from GitHub at configure time (`GIT_SHALLOW FALSE`): a network fetch inside the build, which the house rules want pinned and visible. With it, CMake takes `<dir>/ggml` as a subdirectory (static, CPU only, no OpenMP: `:138-146`). AOT (`:40-47`) adds `-fsycl-targets=spir64_gen` and passes `-device ${STRATA_SYCL_AOT} -options -cl-fp32-correctly-rounded-divide-sqrt` to ocloc; the JIT build passes the rounding option to the spir64 backend instead (`:53-55`). `build.sh:7-8` configures with `-DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DSTRATA_SYCL_AOT="${AOT:-}"`. Parity tests are built by default (`:224`, `option(STRATA_SYCL_PARITY ... ON)`) and registered with `add_test` (`:233, :241, :245, :250, :257, :267`); INTEL_ARC.md:117-118 says run them with `ctest --test-dir build-sycl` on the card.

### Change

1. Clone llama.cpp once, pinned, outside the build tree: `git clone https://github.com/ggml-org/llama.cpp.git /home/ubuntu/src/llama.cpp && git -C /home/ubuntu/src/llama.cpp checkout 3cf03257f219afbe7334045ff7c6a06ac68c627d`. Record the sha in `infra/arc-llm-vm/VERSIONS.md`. The same checkout's `gguf-py` serves the pack tools in I3 (`tools/_paths.py:15-22` reads `STRATA_GGUF_PY`).

2. JIT build (today, no ocloc):

```sh
cd /home/ubuntu/src/strata && git checkout fb58e0d      # or the b70 branch head once the port lands; record the sha
set +u; source /opt/intel/oneapi/setvars.sh; set -u
export SYCL_CACHE_PERSISTENT=0
cmake -S sycl -B build-sycl -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
      -DSTRATA_GGML_DIR=/home/ubuntu/src/llama.cpp
cmake --build build-sycl --target strata -j 8 2>&1 | tee build-sycl/build.log
grep -c 'error:' build-sycl/build.log      # must print 0 (build.sh:11 uses the same check)
```

   `-j 8` for 8 vCPUs (build.sh defaults to 12). Expect several minutes; the engine's first start then JIT-compiles about 47 s of kernels (INTEL_ARC.md:115-116).

3. AOT build (after the owner's yes to `apt install intel-ocloc`, research doc step 2; add the package to `infra/arc-llm-vm/apt-holds.txt`):

```sh
ocloc ids bmg-g31                                        # must list the B70's device id (INTEL.md:96)
cmake -S sycl -B build-sycl-aot -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
      -DSTRATA_GGML_DIR=/home/ubuntu/src/llama.cpp -DSTRATA_SYCL_AOT=bmg-g31
cmake --build build-sycl-aot --target strata -j 8 2>&1 | tee build-sycl-aot/build.log
```

   Keep both trees: the wrapper prefers `build-sycl-aot/strata` when it exists (I1 step 1), as `setup_intel.py:85-90` does.

4. Parity tests (card time; only inside the bench window with the B70 free, research doc step 4):

```sh
ONEAPI_DEVICE_SELECTOR=level_zero:gpu SYCL_CACHE_PERSISTENT=0 ctest --test-dir build-sycl --output-on-failure
```

   Expected set on a B70 (INTEL.md:191-199 and later entries): the 19 kernel tests pass; `iq_parity` and `ple_parity` need fixtures (`iq_parity` is only registered when `STRATA_IQ_FIXTURE`-style inputs exist on the author's branch, `maxfridbe/intel-arc-0.1.40:sycl/CMakeLists.txt:218-230`; our fork's `:226-233` loop includes `iq` and `ple` with `--selftest`, so expect those two to report missing fixtures); `qsa_prompt_attn_parity` and `kv_hybrid_parity` pass since the XMX kernel (INTEL.md:566). Record the pass list in the bench report.

5. Binary location: `build-sycl/strata` (JIT) or `build-sycl-aot/strata` (AOT), because the configure uses `-S sycl`; with the top-level option the binary would be at `build-sycl/sycl/strata` (INTEL_ARC.md:122-124), which the wrapper does not look for. `build-sycl/strata --help` prints the flags; `strings build-sycl/strata | grep -m1 STRATA_VERSION` is not needed: the engine's start log line says the version (`session is up (engine ...)`, perf_matrix.py:206).

### Tests

The build log has `errors: 0`; `ctest` in the window; a 64-token greedy run with the research doc's run A command. No homelab test.

### Done when

Both trees build from the recorded sha with the pinned ggml checkout, `ctest` matches the expected set, and `VERSIONS.md` records the Strata sha, the llama.cpp sha, the oneAPI version and (later) the ocloc version.

### Effort

0.5 day (JIT), plus 0.25 day for AOT once ocloc is installed.

---

## I3. Model fetch and pack by hand, without `setup.py`

### Where

`setup.py:63-70` (`HF_REVISIONS`, pinned commits), `:84-85` (`hf()`: `.../resolve/<rev>/`), `:205-206` (`MODELS["IQ2_XS"]`), `:268-273` (family `qwen`: `"hf": hf(repo) + "{q}/"`, `"file": "Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf"`, `"mmproj": ...`), `:302` (`MMPROJ`), `:5296-5318` (the pack step), `:5330-5334` (the MTP chain), `:5351-5358` (the engine args setup writes), `:5453-5455` (the config keys), `:5281-5282` and `:5490-5512` (`verify_sha256`: only for families with a `sha256` table, which is Unsloth; the ISTA files get no hash check from setup). `data/gguf_fingerprints.json` (`qwen/IQ2_XS`: the two file names, 68,015,068,160 bytes in all). `tools/iq_pack.py:1-39` (pack layout), `:493-503` (argparse), `:565-566` (it runs `strata_tokenizer.py` itself when `tokenizer/` is missing). `tools/mtp_fetch.py:29-32` (`PINNED_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"`), `:298-302` (argparse: `cmd` in `inventory|fetch|verify`, `--out` required, `--only`), `:47-50` (built-in sha256 per tensor). `tools/mtp_pack.py:134-139` (`--src`, `--experts` in `QUANT` keys, default `q2_0`, `--out`, `--check-experts`), `:36` ("the pinned llama.cpp gguf-py, which knows Q2_0 = type 42"). `tools/mtp_rt.py:61-64` (`--gguf`, `--out`). `tools/_paths.py:15-22` (`STRATA_GGUF_PY`). `data/expert-profile.bin`, `data/expert-profile-coder.bin` (present in the checkout).

### Current behaviour

`setup.py` downloads at the pinned revision, packs with `iq_pack.py`, fetches and packs the MTP block, then writes the config with these args (`:5351-5358`):

```
args = ["--pack", str(pack), "--native", str(shards[0]), *(["--ple-gguf", str(ple)] if len(shards) <= 2 else []),
        "--expert-profile", str(ROOT / "data" / fam.get("profile", "expert-profile.bin")), "--expert-cache", "auto",
        "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5", "--mtp", str(rt),
        "--max-context", str(ctx)]
...     args += ["--kv", kv]
```

and `setup_intel.py:127-186` then adds `--stream-experts`, `--vram-reserve-mib 1024` (32K context) and, for a 24 GB+ card, replaces `--prefill auto` with `--prefill 4096`. The pack step runs with `STRATA_GGUF_PY` pointing at setup's own llama.cpp clone (`:5299`).

### Change

Target layout on the VM (all under `/opt/llm`, owned by `ubuntu`): models in `/opt/llm/models/strata/IQ2_XS/`, the pack in `/opt/llm/strata/packs/iq2_xs/`, MTP in `/opt/llm/strata/mtp-bf16/` and `/opt/llm/strata/mtp-bf16/rt/`, config `/opt/llm/strata/strata-flash.json`. About 73 GB of downloads (research doc step 3); the VM needs egress to huggingface.co once.

1. Files, at the pinned revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09` of `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` (`setup.py:67`); the family's URL puts each quant in its own folder (`:270`, `hf(...) + "{q}/"`):

   - shard 1: `https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf` (about 39.2 GB)
   - shard 2: same path with `-00002-of-00002.gguf` (about 28.8 GB; the `per_layer_token_embd` table, shared by every size: hard-link it if another size is ever downloaded)
   - mmproj `mmproj-Qwen3.8-Flash-Next-BF16.gguf` (`setup.py:302`): skip; vision is off by the audit's conditions (research doc section 4).

   Download with `curl -L --fail --retry 5 -C - -o <file> <url>` (resumable).

2. Hash check against the Hub's LFS oid (setup does none for this family, so this is ours). The oid is the file's sha256. Two ways to read it, neither verified live from these repos: (a) the tree API `https://huggingface.co/api/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ2_XS` returns one entry per file with `lfs: {oid, size}`; (b) a `HEAD` on the resolve URL returns the sha256 in the `X-Linked-Etag` header. Then `sha256sum <file>` must equal the oid and `stat -c %s` the size; write `<file>.sha256` beside the file with the oid and the revision, as setup's `.done` mark does (`setup.py:5494, 5512`). Also check the two sizes sum to `68015068160` (`data/gguf_fingerprints.json`, `qwen/IQ2_XS.bytes`). A mismatch deletes the file and refetches (`verify_sha256`'s rule, `:5507-5510`).

3. Pack tools' Python: the same venv as I1 step 3, plus `export STRATA_GGUF_PY=/home/ubuntu/src/llama.cpp/gguf-py` (the I2 checkout at `3cf03257`). Gate before packing: `python -c "import sys; sys.path.insert(0, '$STRATA_GGUF_PY'); import gguf; gguf.GGMLQuantizationType.Q2_0"`. `mtp_pack.py:36` says the pinned gguf-py knows Q2_0 (type 42); whether ggml-org's `3cf03257` has it is not verified. If it fails, clone the commit `setup.py` pins for its own `third_party/llama.cpp` (`LLAMA_CPP_COMMIT`, `setup.py:3138`; read its value there) into a second directory and point `STRATA_GGUF_PY` at that one.

4. The pack (`iq_pack.py:495-503`; seconds, the experts stay in the GGUF):

```sh
cd /home/ubuntu/src/strata && .venv/bin/python tools/iq_pack.py \
  --gguf /opt/llm/models/strata/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf \
  --out /opt/llm/strata/packs/iq2_xs
```

   No `--compat-bf16` (that is for Unsloth files, `setup.py:5316-5318` comment), no `--experts-bin` (the engine reads experts from the GGUF with `--native`), no `--base`. The tool reads shard 2 beside shard 1 by name (`iq_pack.py:27-29`), so both must sit in one folder with the published names (`:507-509`: it keeps the filename, do not rename). It writes `index.txt`, `dense.bin`, `conversions.json`, `native_experts.txt` (last; a pack without it is unfinished) and `tokenizer/` (via `strata_tokenizer.py`, `:565-566`).

5. The MTP draft layer, exactly as `setup.py:5330-5334` chains it:

```sh
.venv/bin/python tools/mtp_fetch.py fetch --out /opt/llm/strata/mtp-bf16          # ~4.9 GB from Qwen/Qwen3.8-Flash-Next at de4b8e4d; sha256 per tensor built in
.venv/bin/python tools/mtp_fetch.py verify --out /opt/llm/strata/mtp-bf16         # exit 3 = a tensor missing or corrupt (mtp_fetch.py:45)
.venv/bin/python tools/mtp_pack.py --src /opt/llm/strata/mtp-bf16 --experts q2_0 --out /opt/llm/strata/mtp-bf16/mtp-q2_0.gguf
.venv/bin/python tools/mtp_rt.py --gguf /opt/llm/strata/mtp-bf16/mtp-q2_0.gguf --out /opt/llm/strata/mtp-bf16/rt
```

   `fetch` writes `mtp-manifest.json` with each tensor's sha256 and checks them (`mtp_fetch.py:47-50`); `STRATA_MTP_REVISION` must stay unset so the pinned revision is used (`:30-32`). `--experts q2_0` is setup's choice (`:5332`). `rt/` holds `experts.bin`, `dense.bin`, `dense.txt` (`mtp_rt.py:5-13`). Setup also copies a draft vocabulary into the MTP dir (`refresh_draft_vocab`, `setup.py:4022-4023, 4084-4085`, `data/draft_vocab*.bin`); whether the engine needs it beside `--mtp` is not verified: run the engine once and read its start log for a draft-vocab line before copying `data/draft_vocab.bin` into `rt/`.

6. Expert profile: `/home/ubuntu/src/strata/data/expert-profile.bin` (the original model's; `setup.py:5352` picks `fam.get("profile", "expert-profile.bin")`, and only the Coder family names another). Add `--expert-profile-save /opt/llm/strata/expert-profile-live.bin` from day one (engine plan, Tier 0) so the resident set follows our traffic; the flag exists (`sycl/src/program/generate.cpp` has `"--expert-profile-save"`).

7. The serve config, `/opt/llm/strata/strata-flash.json`, written by hand with host paths (fields from INTEL.md:381-386 and the keys setup writes, `setup.py:5453-5455`; `exe`, `args`, `cwd`, `tokenizer`, `model_name` are what `--config` expects, `serve/server.py:5614`):

```json
{"engine": "strata", "backend": "sycl",
 "exe": "/home/ubuntu/src/strata/sycl/serve/strata-native.sh",
 "cwd": "/home/ubuntu/src/strata",
 "tokenizer": "/opt/llm/strata/packs/iq2_xs/tokenizer",
 "model_name": "flash",
 "log": "/opt/llm/strata/strata-flash.log",
 "args": ["--pack", "/opt/llm/strata/packs/iq2_xs",
          "--native", "/opt/llm/models/strata/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf",
          "--ple-gguf", "/opt/llm/models/strata/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf",
          "--expert-profile", "/home/ubuntu/src/strata/data/expert-profile.bin",
          "--expert-profile-save", "/opt/llm/strata/expert-profile-live.bin",
          "--expert-cache", "auto", "--stream-experts",
          "--prefill", "4096", "--spec", "4", "--spec-min-p", "0.5",
          "--mtp", "/opt/llm/strata/mtp-bf16/rt",
          "--max-context", "32768", "--kv", "int8", "--vram-reserve-mib", "1024"],
 "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.05},
 "env": {"STRATA_STAGER_THREADS": "8"}}
```

   Every flag above exists in `sycl/src/program/generate.cpp` (checked by string: `--pack`, `--native`, `--ple-gguf`, `--expert-profile`, `--expert-profile-save`, `--expert-cache`, `--stream-experts`, `--prefill`, `--spec`, `--spec-min-p`, `--mtp`, `--max-context`, `--kv`, `--vram-reserve-mib`; `--stream-experts` is port-only, absent from upstream's `src/program/generate.cpp`). `--prefill 4096` is what `setup_intel.py:165-167` writes for a 24 GB+ card; `--vram-reserve-mib 1024` is the 32K value (`:154-156`). No `"api_key"` in the file: the key comes from `STRATA_API_KEY` in the unit's `EnvironmentFile` (`serve/server.py:5631`). No `"vision"`, no `"port"`/`"host"` (given on the command line), no `mcp` keys, no `before_load` (audit conditions). `model_name` is `flash`, the alias arc-queue routes (I6); `"aliases"` can add `flash-next` if the server supports a list (`server.py:4559-4562` lists `svc.aliases`; how the config sets them is not verified: grep `aliases` in `serve/server.py` before relying on it).

### Tests

Hash equality for both shards; `mtp_fetch.py verify` exit 0; `iq_pack.py` ends with `native_experts.txt` present and `tokenizer/vocab.json` and `tokenizer/chat_template.jinja` present (`iq_pack.py:565`); `python -c "import json; json.load(open('/opt/llm/strata/strata-flash.json'))"`; then the engine's run A (research doc step 4) and the start lines `N experts resident`, `MiB of VRAM free with everything loaded` not `LOW`.

### Done when

The pack, MTP runtime and config exist with recorded hashes and revisions in `infra/arc-llm-vm/VERSIONS.md` (model commit `ed59f92`, MTP checkpoint `de4b8e4d`, the two oids), and a served request through `server_intel.py` on `127.0.0.1:8097` answers.

### Effort

0.5 day of hands-on time plus download time (73 GB; the engine plan says a fetch at the VM's link speed is most of the window). The gguf-py Q2_0 gate may add an hour.

---

## I4. Bench harness

### (a) A `native` runner mode for `sycl/benchy.sh` and `sycl/tools/perf_matrix.py`

#### Where (author's branch `maxfridbe/intel-arc-0.1.40`; neither file exists in our fork)

`sycl/benchy.sh:27-43` (the runner `case`: `docker`, `podman`, `distrobox:?*`, else usage error), `:44-47` (re-exec with `sudo --preserve-env=STRATA_SYCL_ROOT,STRATA_SYCL_IMAGE,STRATA_SYCL_RUNNER,STRATA_SYCL_BIN,ONEAPI_DEVICE_SELECTOR`), `:49` (runs `perf_matrix.py`). `sycl/tools/perf_matrix.py:31-38` (`ROOT`, `IMAGE`, `BIN`, `RUNNER`, `BOX`), `:42` (`ENV`, the three wrapper defaults), `:115-133` (`machine()`: the `image` string, docker inspect for non-distrobox), `:157-170` (`engine_cmd(args, sel, cfg_env)`: the distrobox branch builds a `bash -c` line that sources setvars and exports env; the container branch runs docker/podman), `:173-177` (`kill_engine`), `:195-200` (`to_container`/`to_host`: `/work/` prefix), `:228` (`--tokens-file` written as a container path), `:232` (`sel`: `ONEAPI_DEVICE_SELECTOR` or `level_zero:gpu` for a layer split), `:350-351` (refuses when `REPO / BIN` is missing), `:358` (`disk_of(to_host(...))`).

#### Current behaviour

`RUNNER = os.environ.get("STRATA_SYCL_RUNNER", "docker")` (`:37`); `engine_cmd` has two shapes. The distrobox shape (`:161-164`) is already "run a shell that sources oneAPI, exports the env, cds into the repo and execs the binary", just wrapped in `distrobox enter`:

```python
    if BOX:   # the image's environment: its single-card selector, oneAPI's libraries, the OOM killer's first pick
        env = env0 + [f"ONEAPI_DEVICE_SELECTOR={sel or 'level_zero:0'}"]
        return box_cmd("echo 1000 > /proc/self/oom_score_adj; . /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; export "
                       + " ".join(shlex.quote(e) for e in env) + "; " + run)
```

`run` (`:159`) is `f"cd /work/{REPO.name} && exec {BIN} " + args`, so paths are container paths everywhere (`to_container`, `:195-196`), and `kill_engine` for distrobox uses `pkill -x` on the binary's name (`:175`). Benchy runs as root (page-cache drop and fdinfo VRAM reads, `:19-20`); results are chowned back (`benchy.sh:51`).

#### Change

1. `sycl/benchy.sh`: add a `native)` arm to the `case` at `:27-43`, before `*)`:

```bash
native)
    [ -x "${STRATA_NATIVE_BIN:-$here/../build-sycl/strata}" ] || [ -x "$here/../build-sycl-aot/strata" ] \
        || { echo "benchy: no native engine (build-sycl/strata or build-sycl-aot/strata; docs/INTEL_ARC.md)" >&2; exit 2; }
    [ -r "${ONEAPI_ROOT:-/opt/intel/oneapi}/setvars.sh" ] || { echo "benchy: oneAPI not found (ONEAPI_ROOT)" >&2; exit 2; } ;;
```

   and update the usage message at `:42` to "docker, podman, distrobox:<name> or native"; add `STRATA_NATIVE_BIN,ONEAPI_ROOT` to the `--preserve-env` list at `:46`; update the header comment `:14-16`.

2. `sycl/tools/perf_matrix.py`:
   - `:37-38`: add `NATIVE = RUNNER == "native"`. After `:33` (`BIN`), when `NATIVE`, default `BIN` to the first of `build-sycl-aot/strata`, `build-sycl/strata` that exists (relative to `REPO`), honouring `STRATA_NATIVE_BIN` when set (absolute).
   - `to_container` (`:195-196`): `if NATIVE: return str(p.resolve())`; `to_host` (`:199-200`) is already a no-op for non-`/work/` paths. The out dir restriction "must sit under the data root" (`:22`) does not apply natively: skip the check when `NATIVE` (there is no explicit check in the code; the restriction is only that `to_container` would raise `ValueError` from `relative_to`, which the native branch avoids).
   - `engine_cmd` (`:157-170`): a third branch before `if BOX:` that returns the distrobox shape without `distrobox enter`:

```python
    if NATIVE:   # the host's oneAPI; the env as strata-native.sh sets it, the config's "env" on top
        env = env0 + [f"ONEAPI_DEVICE_SELECTOR={sel or 'level_zero:gpu'}", "SYCL_CACHE_PERSISTENT=0", "ZES_ENABLE_SYSMAN=1"]
        exe = BIN if os.path.isabs(BIN) else str(REPO / BIN)
        return ["bash", "-c", "echo 1000 > /proc/self/oom_score_adj; . " + shlex.quote(os.environ.get("ONEAPI_ROOT", "/opt/intel/oneapi") + "/setvars.sh")
                + " >/dev/null 2>&1; export " + " ".join(shlex.quote(e) for e in env) + "; cd " + shlex.quote(str(REPO))
                + " && exec " + shlex.quote(exe) + " " + " ".join(shlex.quote(a) for a in args)]
```

     The `cd` into the repo keeps relative paths such as `data/expert-profile.bin` working as in the container (`:159`). Mirror the ENV default `STRATA_STAGER_THREADS=12` (`:42`) to 8 when `NATIVE` and `os.cpu_count() <= 8`, or better: leave `ENV` as is and let the config's `"env"` carry `STRATA_STAGER_THREADS` (I3 step 7 does).
   - `kill_engine` (`:173-177`): `if BOX or NATIVE:` use the `pkill` branch (natively the engine is our own child; `p.kill()` on the Popen would be simpler, but `run_one` has the Popen in scope only, so pass it or keep `pkill -x`).
   - `machine()` (`:115-133`): when `NATIVE`, build `image` from `icpx --version` run through the same `bash -c` sourcing (as the distrobox branch does, `:119-121`), labelled `native, <icpx line>`.
   - `:350-351`: the binary check must use the resolved `BIN` (absolute or `REPO / BIN`).
   - `:232`: `sel` already falls back to `None`; the native branch supplies `level_zero:gpu` (one Intel card on the VM: equivalent to `level_zero:0`).
   - The `default_configs()` filter (`:325-334`) wants `"backend": "sycl"` in the config; the hand-written config in I3 has it.
   - The `parse()` regexes (`:203-220`) are the engine's start and `--stats` lines; nothing to change.

3. Document the mode in `docs/INTEL_ARC.md` next to "Build without Docker" (one paragraph) and in the perf_matrix docstring (`:2-23`).

#### Tests

`STRATA_SYCL_RUNNER=native python3 sycl/tools/perf_matrix.py --configs /opt/llm/strata/strata-flash.json --sizes 20 --warm --out /opt/llm/strata/benchy/smoke` (no root, warm cache) produces a `matrix.md` with one row whose `exit` is 0. Add a unit test `sycl/tools/test_perf_matrix_native.py` that imports the module with `STRATA_SYCL_RUNNER=native`, `STRATA_NATIVE_BIN=/bin/true`, and asserts `engine_cmd(["--x", "/a b"], None, ["K=v"])` is a `bash -c` list containing `setvars.sh`, `export ... ONEAPI_DEVICE_SELECTOR=level_zero:gpu ... K=v`, `exec /bin/true --x '/a b'`, and that `to_container(Path("/opt/x"))` returns `/opt/x`. (The module imports `gpustat` at load, `:45`; our fork has `sycl/tools/gpustat.py` with `rd`, `clients`, `pcie_link`, `card_name`, `vram_total_mb`, `PCI`, so the import works; the branch's gpustat adds a `pdev` field to `clients()` entries that `vram_by_card_mb` (`:93-98`) reads: port that 16-line gpustat diff with perf_matrix, I5.)

#### Done when

`sudo STRATA_SYCL_RUNNER=native sycl/benchy.sh --configs /opt/llm/strata/strata-flash.json --sizes 20,2185,8000` runs cold in the bench window and writes `matrix.md`, `config-flash.txt` and `matrix.jsonl`.

#### Effort

0.5 day (the engine plan's estimate), including the gpustat port.

### (b) A 30-line HTTP driver for `tools/b70-tuning/prompts/suite.json`

#### Where

`tools/b70-tuning/prompts/suite.json` (keys `version` = 1, `sampling` = `{"temperature": 0, "top_k": 1, "seed": 1234}`, `prompts` = 12 entries with `id`, `category`, `max_tokens`, `messages`; ids `code-01..03`, `tool-01..04`, `prose-01..03`, `doc-01..02`). `tools/b70-tuning/scripts/bench.py:91-126` (its streaming reader takes `timings` from the last chunk), `:196-202` (reads `prompt_per_second` and `predicted_per_second`), `:208-226` (argparse: `--name`, `--model`, `--port` in 47690-47699, `--bin`, no URL option), `:33-35, :235` (refuses when any other `llama-server` is alive). `serve/server.py:3776-3792` (`request_timings`: the fields Strata returns), `:3758` (`"timings": timings` in the final event), `:4001-4002` and `:4045-4046` (the non-stream reply carries the last chunk's `timings`, "llama.cpp's field: the speed its clients show"); `sycl/serve/server_intel.py` does not touch `timings` (grep: no match).

#### Current behaviour

`bench.py` starts its own `llama-server` and cannot point at a URL, so it cannot drive Strata. Strata's non-stream `/v1/chat/completions` reply contains `timings` with exactly these keys (`server.py:3785-3792`): `cache_n`, `prompt_n`, `prompt_ms`, `prompt_per_token_ms`, `prompt_per_second`, `predicted_n`, `predicted_ms`, `predicted_per_token_ms`, `predicted_per_second`, plus `draft_n` and `draft_n_accepted` when the engine reported drafts. `prompt_per_second` and `predicted_per_second` are `None` when the count or the time is zero (`:3787, :3790`). This is llama.cpp's shape, so `bench.py`'s row format (`:196-202`) applies unchanged.

#### Change

Add `tools/b70-tuning/scripts/strata_timings.py` in homelab (beside `bench.py`; it reads the same suite):

```python
#!/usr/bin/env python3
"""POST each prompt of prompts/suite.json to an OpenAI chat endpoint and print the reply's `timings`
(prompt_per_second, predicted_per_second, drafts), one JSON line per prompt. For Strata on 127.0.0.1:8097
through an SSH port forward; works against any llama.cpp-shaped server. Key from $STRATA_API_KEY, never printed."""
import argparse, json, os, pathlib, sys, time, urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8097/v1/chat/completions")
    ap.add_argument("--model", default="flash")
    ap.add_argument("--prompts", default="", help="comma list of prompt ids (default: all)")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    suite = json.loads((ROOT / "prompts" / "suite.json").read_text())
    want = set(a.prompts.split(",")) if a.prompts else None
    headers = {"Content-Type": "application/json", "X-Client": "b70-bench"}
    if os.environ.get("STRATA_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["STRATA_API_KEY"]
    for p in suite["prompts"]:
        if want and p["id"] not in want:
            continue
        for run in range(1, a.runs + 1):
            body = {"model": a.model, "messages": p["messages"], "max_tokens": p["max_tokens"], "stream": False,
                    "chat_template_kwargs": {"enable_thinking": False}, **suite["sampling"]}
            req = urllib.request.Request(a.url, json.dumps(body).encode(), headers)
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=900) as r:
                d = json.load(r)
            t = d.get("timings") or {}
            print(json.dumps({"label": a.label, "id": p["id"], "category": p["category"], "run": run, "wall_s": round(time.time() - t0, 2),
                              "prefill_tok_s": t.get("prompt_per_second"), "decode_tok_s": t.get("predicted_per_second"),
                              "prompt_n": t.get("prompt_n"), "predicted_n": t.get("predicted_n"),
                              "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted"),
                              "finish": d["choices"][0].get("finish_reason")}), flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

   Notes: `suite["sampling"]` carries `seed` and `top_k`, which Strata's OpenAI endpoint accepts (`server.py:5508`, `SHARED_KEYS` includes `top_k` and `seed`). `enable_thinking: false` through `chat_template_kwargs` is handled at `serve/frontend.py:391-393`. `stream: False` so `timings` arrives in the final JSON (`server.py:4045-4046`). Compare with the qwen38 baseline rows (`REPORT.md` `s1007_*`) by the same `prefill_tok_s`/`decode_tok_s` names `bench.py:200` uses.

#### Tests

A `test_strata_timings.py` in `tools/b70-tuning/scripts/` that starts `http.server` on a free port in 12000-13000 (global rule), answers a canned reply with `timings`, runs `main()` with `--url` and one `--prompts code-01`, and asserts the printed line's `prefill_tok_s`/`decode_tok_s`. Also assert the request body has `max_tokens` from the suite and `chat_template_kwargs.enable_thinking == False`.

#### Done when

Twelve lines per run against Strata through `ssh -L 8097:127.0.0.1:8097 ubuntu@192.168.0.211`, and the same twelve against `http://192.168.0.211:8080/v1/chat/completions --model qwen38` for the cross-check (arc-queue forwards `timings` from llama-server; `bench.py:105-106` relies on the same field).

#### Effort

0.25 day.

### (c) `evals.py` target for Strata

#### Where

`tools/llm-evals/evals.py:34-40` (`TARGETS`: name to `(url, model)`), `:42` (`TIMEOUT_S = 900`), `:50-59` (`post`, `chat`: the request body), `:266` (`--target` choices are `TARGETS`). Strata: `serve/frontend.py:387-395` (`reasoning_effort`, `chat_template_kwargs`), `serve/server.py:2759-2769` (`reasoning_budget_tokens` read from the request), `:5508` (`SHARED_KEYS`).

#### Current behaviour

`evals.py:56-58`:

```python
    url, model = TARGETS[target]
    body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens + think, **extra}
    body.update({"reasoning_budget_tokens": think} if think else {"chat_template_kwargs": {"enable_thinking": False}})
```

`post` (`:50-54`) sends `Content-Type` and `X-Client` headers only: no `Authorization`. Strata's endpoint accepts both fields evals.py sends: `reasoning_budget_tokens` is a first-class request field (`server.py:2759-2769`, a non-integer is a 400) and `chat_template_kwargs.enable_thinking` is read at `frontend.py:391-393`. The `extra` the `extract` suite adds is `response_format: {type: json_schema, ...}` (`:71-72`); Strata has `serve/structured.py` and the frontend handles `response_format` (not line-verified here). Unknown top-level fields are read with `req.get(...)`; no whitelist that rejects unknown keys was found in `serve/frontend.py` or `serve/server.py` (searched for "unknown field", "unexpected", a key whitelist), so arc-queue-style extras such as `speculative.n_max` would be ignored, not rejected: not verified exhaustively.

#### Change

1. One line in `TARGETS` (`evals.py:40`, after `gateway-router`):

```python
    "strata": ("http://127.0.0.1:8097/v1/chat/completions", "flash"),  # Strata on the VM via ssh -L 8097:127.0.0.1:8097 ubuntu@192.168.0.211
```

2. The API key: `post` (`:50-54`) must add `Authorization: Bearer <key>` when `STRATA_API_KEY` is set in the environment (two lines: `if os.environ.get("STRATA_API_KEY"): headers["Authorization"] = "Bearer " + os.environ["STRATA_API_KEY"]`). The key is read from the 0600 file in a subshell (`set -a; . /path/strata-flash.env; set +a; python3 evals.py run --target strata ...`), never printed.

3. Usage: baseline before the window (research doc step 4): `evals.py run --target qwen38 --label pre-strata --suite all`; in the window `evals.py run --target strata --label strata-iq2xs --suite extract,tier,tool,reason` (the `--suite` syntax takes one value per `:4`; run the four separately if it does not take a list: not verified), then `evals.py compare results/<pre>.json results/<strata>.json`.

#### Tests

`python3 -c "import evals; assert 'strata' in evals.TARGETS"`; the existing suite has no unit tests for `TARGETS` (none found), so a smoke run against the forward is the test.

#### Done when

`evals.py compare` runs between the two labels with no exception; results are committed under `tools/llm-evals/results/` as the other labels are.

#### Effort

0.1 day.

---

## I5. Porting the author's finished items from `maxfridbe/intel-arc-0.1.40`

### Where (branch paths; the branch has unrelated history, `git merge-base` is empty)

Per switch, from `git grep` on the branch:

| Item | Files on the branch (line of the switch) | Our fork's state |
|---|---|---|
| `STRATA_LEND_MIRROR` | `sycl/src/program/generate.cpp:6222-6262` (the hunk; `:5202` makes `t_start` non-const), `sycl/include/strata/core/gguf_expert_source.hpp:45-52, 71-77` (`mirror(..., bool append=false)`, `blocks_`, `mirror_ptr_`, `layer_first_` as pointers), `sycl/src/core/gguf_expert_source.cpp` (47 lines changed: `close`, `blob`, `mirror` append path), `sycl/tools/perf_matrix.py:217` (parses the "lendable slots mirrored" line), `docs/INTEL.md:305-312` | `gguf_expert_source.hpp:49` has `mirror()` without `append`; `generate.cpp:4603` calls it once for the VRAM misses; the borrow machinery (`--prefill-borrow`, `plan_lend`) is present (`generate.cpp:1778-1779, 2151, 3348`) |
| `STRATA_SELECT_GEMM` | `sycl/src/prefill/prefill.cpp:41` (declaration of `qsa_block_scores_reduce`), `:106` (`kSelTile = 8192`), `:657, :829, :1176` (`sel_S` scratch), `:2996-3012` (the dispatch), `sycl/src/kernels/cuda/qsa_select.dp.cpp:1571` (`qsa_block_scores_reduce`, 36 lines changed in the file), `sycl/src/kernels/sel_scores_bench.cpp` (new, 160 lines), `sycl/CMakeLists.txt:314-315`, `docs/INTEL.md:326-333` | `prefill.cpp:2993-2994` has `old_sel` and the XMX `qsa_block_scores_tc` call; no `sel_S`, no `kSelTile`, no reduce kernel |
| `STRATA_ATTN_PERCELL` | `sycl/src/kernels/cuda/qsa_decode_attn.dp.cpp:601-700` (`attn_chunk_percell_kernel`, `launch_percell<KV_MODE>`), `:753-768` (dispatch; 184 lines changed in the file), `sycl/src/kernels/attn_bench.cpp` (new, 676 lines), `sycl/CMakeLists.txt:316-317`, `docs/INTEL.md:334-340` | `qsa_decode_attn.dp.cpp:118` (`attn_chunk_kernel`), `:554` and `:601` (`STRATA_ATTN_LANECELL` opt-in, the S25 variant the branch keeps below the percell branch) |
| `STRATA_MMVQ_PREUNPACK` (+ Q6_K/Q5_K parity and benches, `STRATA_MMVQ_LOOP`) | `sycl/include/strata/kernels/native_mmvq.hpp:150-176` (+40 lines: `native_mmvq_q6k_preunpack_bytes`, `native_q6k_preunpack`, `native_mmvq_set/register/unregister/clear_q6k_preunpack`, the Q5_K pair), `sycl/src/core/native_dense.cpp:110-115` (`q6k_preunpack_enabled`), `:363-399` (the one-time unpack at load; 83 lines changed), `sycl/src/kernels/cuda/native_mmvq.dp.cpp:67-68` (globals), `:1750-1751` (`STRATA_MMVQ_LOOP`), `:3160-3165` (the lookup in the Q6_K path), `:3554-3600` (the preunpack kernels; the file's diff is 642 lines and includes other work), `sycl/src/kernels/q6k_preunpack_parity.cpp`, `q5k_preunpack_parity.cpp`, `q6k_preunpack_bench.cpp`, `q5k_preunpack_bench.cpp`, `xmx_mmvq_bench.cpp` (new), `sycl/CMakeLists.txt:300-313`, `docs/INTEL.md:343-350` | none of the symbols exist (`git grep` for `preunpack` on `HEAD` is empty) |
| `sycl/benchy.sh` + `sycl/tools/perf_matrix.py` | new files (`benchy.sh` 52 lines, `perf_matrix.py` 377), `sycl/bench/v1/short.ids`, `long.ids` (new), `sycl/tools/gpustat.py` (+16: `pdev` per client, B65 name, `lspci -vmm`), `sycl/tools/rank_kernels.py` (+18) | none of the new files; `gpustat.py` is the older version |
| `sycl/TODO.md` | new (the author's log) | none |
| The five benches | `xmx_mmvq_bench.cpp`, `xmx_int8_bench.cpp` (208), `onednn_gemm_bench.cpp` (116), `sel_scores_bench.cpp`, `attn_bench.cpp`; CMake `:298-317` | none |

Files on the branch that are NOT part of these items and must not be dragged in: `sycl/src/core/verify.cpp` (1,156 lines changed), `sycl/src/core/mtp.cpp` (612), `sycl/src/kernels/cuda/fused_gr.dp.cpp` (1,899), `sycl/include/strata/sycl_queue.hpp` (190 removed), `sycl/src/prefill/moe_fused*.dp.cpp` (removed on the branch, present in ours), `sycl/src/core/vmm.cpp` (removed on the branch). Our fork is 0.1.41 and his is 0.1.40: those files differ because of upstream, not because of his items.

### Strategy

A git 3-way merge needs a common ancestor; `fb58e0d` and the branch share none, so `git merge`, `git cherry-pick` and `git merge-file` (which needs a base) are out. Two workable routes, by file:

1. New files (benches, parity tests, `benchy.sh`, `perf_matrix.py`, `bench/v1/*.ids`, `TODO.md`, `sel_scores_bench.cpp`, `attn_bench.cpp`): `git show maxfridbe/intel-arc-0.1.40:<path> > <path>` (`git checkout maxfridbe/intel-arc-0.1.40 -- <path>` does the same). Then adapt includes if a header moved between 0.1.40 and 0.1.41 (compile tells).
2. Modified files: extract the hunks, not the file. `git diff fb58e0d maxfridbe/intel-arc-0.1.40 -- <path> > /tmp/<file>.diff` is noisy (the whole-file diff mixes upstream's 0.1.40-to-0.1.41 drift with his change), so instead read the branch's version around the switch (`git show maxfridbe/intel-arc-0.1.40:<path> | sed -n A,Bp`) and apply the hunk by hand into our file at the matching function, keeping our 0.1.41 context. For `generate.cpp`, `prefill.cpp`, `qsa_decode_attn.dp.cpp`, `native_dense.cpp`, `native_mmvq.dp.cpp`, `gguf_expert_source.{hpp,cpp}`, `CMakeLists.txt`, `INTEL.md` that is the only safe way. Where a hunk is self-contained (the percell kernel and launcher, the preunpack kernels) a restricted patch works: `git diff fb58e0d maxfridbe/intel-arc-0.1.40 -- sycl/src/kernels/cuda/qsa_decode_attn.dp.cpp | filterdiff --hunks=N` or `git apply --3way` fails without a base, so use `patch -p1 --dry-run` per hunk and fix offsets.
3. Every item lands as its own commit on branch `b70` from `fb58e0d` (engine plan section 6), behind its env switch with the old path kept, default byte-identical where the author says so (LEND_MIRROR, PREUNPACK) and documented as not bitwise where he says so (SELECT_GEMM, ATTN_PERCELL: INTEL.md:321-324 on the branch).

### Order

1. `sycl/TODO.md`, `sycl/bench/v1/*`, `sycl/benchy.sh`, `sycl/tools/perf_matrix.py`, the `gpustat.py` hunk, `rank_kernels.py` (no engine change; gives the bench that gates everything else). Then I4(a)'s `native` mode on top.
2. `STRATA_LEND_MIRROR` (prompt path; default on, outputs identical per the author).
3. `STRATA_ATTN_PERCELL` (one file plus a bench; default on, not bitwise, same text).
4. `STRATA_SELECT_GEMM` (prefill.cpp scratch plumbing plus a kernel; default on, not bitwise; depends on oneMKL being linked, which `strata_prefill` already is: CMake `:291`).
5. `STRATA_MMVQ_PREUNPACK` + `STRATA_MMVQ_LOOP` + the Q6_K/Q5_K parity tests and benches (default off; the biggest hunk inside a 642-line file diff).
6. The remaining benches (`xmx_int8_bench`, `onednn_gemm_bench`; `onednn_gemm_bench` needs oneDNN headers, `TODO.md` 4: `intel-oneapi-dnnl-devel`, a package install, owner's yes; build it only `if(TARGET ...)`/optional).

### Gates (per item, in the bench window)

- Builds in both trees with `errors: 0`.
- `ctest --test-dir build-sycl`: the item's own parity tests pass (`q6k_preunpack_parity`, `q5k_preunpack_parity` for item 5; `qsa_parity`, `kv_hybrid_parity`, `qsa_prompt_attn_parity` unchanged for items 3 and 4) and no previously passing test regresses.
- Output identity: the three standard prompts (benchy v1's 20, 2,185 and 8,000 tokens, greedy, 256 new tokens; `perf_matrix.py:40-41`) produce byte-identical token streams with the switch at its default and at `0` for LEND_MIRROR and PREUNPACK (the author: "Outputs identical", `TODO.md:31-32, :110-111`); for SELECT_GEMM and ATTN_PERCELL the same text with a documented near-tie divergence after tens of tokens is accepted (`TODO.md:47-48`), recorded in the commit message with the first differing position.
- A benchy row before and after, from `native` mode, attached to the commit (engine plan section 6).
- The B70 is single-tenant for every gate (research doc step 4).

### Effort (engineer time, excluding card time)

| Item | Estimate |
|---|---|
| benchy + perf_matrix + gpustat + TODO.md (copy, compile nothing) | 0.25 day, plus I4(a) 0.5 |
| LEND_MIRROR (hpp/cpp append mode 60 lines, generate.cpp hunk 40 lines, log line) | 0.5 to 1 day |
| ATTN_PERCELL (kernel + launcher ~100 lines into one file, attn_bench copied) | 0.5 day |
| SELECT_GEMM (scratch allocation in three places, dispatch, reduce kernel in qsa_select.dp.cpp, sel_scores_bench) | 1 day |
| PREUNPACK + LOOP + parity + benches (header, native_dense load path, kernels and registry in native_mmvq.dp.cpp) | 1 to 1.5 days |
| other benches | 0.25 day |
| Total | 3.5 to 4.5 days, inside the engine plan's "2 to 4 days" once the bench tooling is counted separately |

### Done when

Branch `b70` carries one commit per item, each with its switch documented in `docs/INTEL.md`, the parity set recorded, and benchy rows; `fb58e0d`'s default outputs are reproduced with every new switch set to its "old path" value.

---

## I6. Operations integration in homelab

### I6.1 Steward `llm-mode strata`

#### Where

`apps/steward/src/llmMode.ts:1-6` (header), `:14-17` (constants), `:19-24` (`ModeFile`), `:56-92` (`parseArgs`: cmds at `:58-59`), `:96-103` (`vm()`: one argument per call), `:112-119` (`waitHealthy`), `:186-222` (`runCyber`), `:225-257` (`runNormal`), `:259-282` (`runLlmMode`: `status` at `:261-265`, `check` at `:269-277`). `infra/host/sbin/llm-mode-vm:1-14`. `infra/host/systemd/steward-llm-mode-check.service` and `.timer` (every 5 min; the service's `ExecStart` hard-codes `/home/sambou/.nvm/versions/node/v20.11.1/bin/node`, a Known conflict with the Node 24 rule; not changed here). Tests: `apps/steward/test/llmMode.test.ts` (9 tests; sandbox at `:43-75`, the fake `vm-cmd.sh` logs every argument and fails one chosen argument), `apps/steward/test/llmModeLease.test.ts` (2 tests). Test command: `npm test` (`tsc -p . && node --test dist/test/`, `apps/steward/AGENTS.md:52`).

#### Current behaviour

`parseArgs` accepts only `cyber|normal|status|check` (`:58`). `runCyber` (`:186-222`): refuses if cyber is already on (`:188-191`) or a foreign lease is active (`:192-196`); writes the `llm:*` lease (`:198-204`); `vm("start-cyber")`, `waitHealthy("health-cyber")`, on failure `vm("start-qwen38")` rollback; writes `{mode: "cyber", since, until, run_id}` (`:216`); always removes the lease (`:219-221`). `runNormal` (`:225-257`): lease, `start-qwen38`, `health-qwen38`, removes the mode file. `status` prints `cyber until N` or `normal` (`:263`). `check` reverts an expired cyber mode unless a foreign lease is active (`:269-277`). The VM helper (`llm-mode-vm:8-13`) is a `case` over four actions, each one `ssh` with a 120 s timeout:

```bash
  start-cyber)   ssh_vm 'sudo systemctl start cyber-llm' ;;          # Conflicts= stops qwen38
  start-qwen38)  ssh_vm 'sudo systemctl start qwen38' ;;             # Conflicts= stops cyber-llm
  health-cyber)  ssh_vm 'curl -s -m 5 http://127.0.0.1:8095/health | grep -q ok' ;;
  health-qwen38) ssh_vm 'curl -s -m 5 http://127.0.0.1:8090/health | grep -q ok' ;;
```

#### Change

1. `infra/host/sbin/llm-mode-vm`: two arms and the usage line:

```bash
  start-strata)  ssh_vm 'sudo systemctl start strata-flash' ;;       # Conflicts= stops qwen38, cyber-llm, shieldgemma
  health-strata) ssh_vm 'curl -s -m 5 http://127.0.0.1:8097/health | grep -q "\"status\": \"ok\""' ;;
```

   Strata's `/health` is `{"status": "ok", ...}` (server.py:4532) and needs no key; `grep -q ok` would also match `"model": "flash"`'s absence, so match the status field. `start-qwen38` already restores `shieldgemma`? No: `Conflicts=` only stops; after `normal`, `shieldgemma` must be started again. Add a third arm `start-shieldgemma) ssh_vm 'sudo systemctl start shieldgemma' ;;` and call it from `runNormal` after qwen38 is healthy when the previous mode was `strata` (see step 3). `install.sh` already maps `sbin/llm-mode-vm` (`infra/host/install.sh:36`).

2. `llmMode.ts`:
   - `:58-59`: accept `"strata"`; usage: `steward llm-mode cyber [--hours N] | strata [--hours N] | normal | status | check ...`.
   - Add `runStrata(o)` modelled on `runCyber` (`:186-222`): refuse if the mode file says `strata` (any `until`) or `cyber` still active (switching cyber to strata goes through `normal` first, so the mode file stays the single truth); foreign-lease check; lease reason `"strata mode: switching the B70 to the Strata engine"`; `vm("start-strata")`; `waitHealthy("health-strata")`; rollback `vm("start-qwen38")` then `vm("start-shieldgemma")` on failure; write `{mode: "strata", since, until, run_id}`. `until`: with `--hours N` it is `now + N*3600` (bench windows); without `--hours` it is `0`, meaning no expiry (production). Keep `ModeFile.until` a number so Sentinel's reader stays valid.
   - `runNormal` (`:225-257`): read the mode file before switching; after `health-qwen38` passes, if the previous mode was `strata`, `vm("start-shieldgemma")` (best effort, log on failure; shieldgemma has its own `Restart=on-failure`). Message on success: `normal mode (qwen38)`.
   - `status` (`:263`): print `strata until N` or `strata` when `until` is 0.
   - `check` (`:269-277`): also revert `strata` when `until > 0 && until <= now`; never when `until === 0`.
   - `HEALTH_ATTEMPTS` (`:15`, 24 x 13 s = 312 s) matches `TimeoutStartSec=300` of the new unit; a JIT build starts in about 90 s plus the mirror fill, so no change.

3. The mode file value is `strata`. Document in `docs/plans/cyber-mode.md` (one line under Pieces 3) and in `skills/proxmox-host-ops/SKILL.md:36` (the helper's action list).

#### Tests (`apps/steward/test/llmMode.test.ts`, same sandbox; add)

- `strata happy path`: calls are `["start-strata", "health-strata"]`, mode file `{mode: "strata", until: 0}`, lease removed, exit 0.
- `strata --hours 2` sets `until = now + 7200`.
- `strata health never passing` (fail arg `health-strata`): calls end with `start-qwen38`, `start-shieldgemma`; no mode file; exit 1.
- `normal after strata` starts qwen38, waits, then `start-shieldgemma`; mode file removed.
- `status` prints `strata` and `strata until N`.
- `check` reverts `strata` with `until <= now` and leaves `until: 0` alone.
- `llmModeLease.test.ts`: `strata` refuses while a foreign lease is active (mirror of F05 at `:46`).

#### Done when

`steward llm-mode strata` and `steward llm-mode normal` switch the card each way in about a minute with the lease held throughout, Sentinel quiet (I6.2), and `status` reflects the mode file.

#### Effort

1 day with tests.

### I6.2 Sentinel

#### Where

`apps/sentinel/sentinel/llm_mode.py:9-32` (`read_cyber`, `cyber_active`). `apps/sentinel/sentinel/collectors/llm.py:13-39` (`REMOTE_PROBE`: `systemctl is-active` for `qwen38`, `arc-gateway`, `arc-queue`, `fast-llm`, `embed`), `:64-67` (the `llmvm:<unit>:inactive` findings, `fix="restart_qwen38"` for qwen38), `:90` (`QWEN38_SIGS`), `:94-105` (`_cyber_findings`), `:108-110` (`collect` filters by `suppress`), `:116-119` (`llm:qwen38_unhealthy` from arc-queue's `/health`). `apps/sentinel/sentinel/qwen_heal.py:53` (`llm_mode_file` default), `:63-64` (`RESTART_CMD`), `:318-345` (`tick`: shed flag `:325`, cyber stand-down `:327-330`, lease hold `:331-338`). `apps/sentinel/sentinel/fixes.py:52-54` (`restart_qwen38` verify), `:87-95` (`restart_qwen38` run: `systemctl restart qwen38`). `apps/sentinel/config.toml:94-100` (`[fixes] enabled` does not include `restart_qwen38`), `:202-209` (`[qwen_heal]`). Tests: `apps/sentinel/tests/test_llm_cyber_aware.py` (4 tests; fakes `http_get` and `ssh`, injects `now`), `apps/sentinel/tests/test_qwen_heal_cyber.py` (7 tests; `_mode_file`, `_wire`, `_run` helpers at `:40-68`). Test command: `.venv/bin/python -m pytest -q tests` (`apps/sentinel/AGENTS.md:43`).

#### Current behaviour

`llm_mode.read_cyber` returns `until` only when `mode == "cyber"` (`:24-25`); `cyber_active` is `until > now` (`:29-32`). The collector suppresses `QWEN38_SIGS` while cyber is active and adds `llm:cyber_mode` (info) or `llm:cyber_mode_overdue` (warning) (`:94-110`). `qwen_heal.tick` returns `[]` while `cyber_active` (`:329-330`), so it never runs `RESTART_CMD["qwen38"]` (`sudo -n systemctl restart qwen38 && ...`, `:63`). The allowlisted fix `restart_qwen38` (`fixes.py:87`) is off in `config.toml:97`, but it exists and `collectors/llm.py:67, :119` attach it to findings.

#### Change

1. `llm_mode.py`: add a general reader beside the cyber one, keeping `read_cyber`/`cyber_active` unchanged (the 7 + 4 existing tests depend on them):

```python
def read_mode(path: str) -> tuple[str | None, float | None]:
    """(mode, until) from the file: (None, None) when missing or unreadable; until may be 0 = no expiry (strata)."""
    ...  # same parsing as read_cyber; returns (d.get("mode"), until) when mode is a str and until a number

def strata_active(path: str, now: float) -> bool:
    """True while the mode file says strata and it has not expired (until 0 = never expires)."""
    mode, until = read_mode(path)
    return mode == "strata" and until is not None and (until == 0 or until > now)

def b70_owned_elsewhere(path: str, now: float) -> bool:
    return cyber_active(path, now) or strata_active(path, now)
```

2. `collectors/llm.py`:
   - `REMOTE_PROBE` (`:36-38`): add `"strata": act("strata-flash")` and `"shieldgemma": act("shieldgemma")` to the JSON.
   - `parse_probe` (`:64-70`): the loop over `("qwen38", "gateway", "queue")` stays; add: if `d.get("strata")` is present and not `"active"` and the mode is strata, `Finding("llmvm:strata-flash:inactive", "critical", "strata-flash service is <state> while the mode file says strata")` (critical because no heal exists yet). `parse_probe` has no mode or time today (`:53`); pass `mode` in from `collect`, or compute the finding in `collect` from the parsed JSON: the simplest is to let `parse_probe` emit `llmvm:strata-flash:inactive` whenever `"strata" in d and d["strata"] != "active"`, and let `collect` drop it when the mode is not strata (mirror of the suppression at `:110`).
   - `_cyber_findings` (`:94-105`) becomes `_mode_findings(cfg, now) -> (suppress_qwen38, suppress_strata, findings)`: for `strata` active: suppress `QWEN38_SIGS`, add `Finding("llm:strata_mode", "info", "LLM in strata mode (qwen38 stopped on purpose; Strata on :8097)")` with `until` in the title when non-zero; for an expired strata (`0 < until <= now`): `llm:strata_mode_overdue` warning, qwen38 findings return. When not in strata mode, suppress `llmvm:strata-flash:inactive` (the unit is meant to be down).
   - Readiness probe, in `_collect` (`:113-133`) when the mode is strata: `http_get(f"http://{vm['host']}:8097/health")` would not work: Strata binds `127.0.0.1` (I1), so probe inside the SSH round trip. Add to `REMOTE_PROBE` a `curl -s -m 5 http://127.0.0.1:8097/health` captured as `"strata_health"` (the JSON string or `""`), and a `curl -s -m 5 -H "Authorization: Bearer $STRATA_API_KEY" http://127.0.0.1:8097/props`: the key is in a root 0600 file the probe cannot read as `ubuntu`, so the `/props` probe is only possible if the VM gets a `ubuntu`-readable copy or a `sudo -n cat` rule; until then probe `/health` only and record that `/props` needs the key (research doc step 5 asks for both; `/props` requires `_authorized()`, `server.py:4563-4564`). Finding `llm:strata_unhealthy` (warning) when `/health` is not 200 with `"status": "ok"`, mode strata only.
   - The arc-queue `/health` check at `:116-119` reports the default worker (`b70`) and says `error` in cyber or strata mode by design (`ops.py:54-56`): it is already suppressed through `QWEN38_SIGS`; nothing else to do.

3. `qwen_heal.py` `tick` (`:327-330`): replace `llm_mode.cyber_active(...)` with `llm_mode.b70_owned_elsewhere(...)` and extend the comment: "cyber or strata mode: another engine owns the B70 on purpose; a `systemctl restart qwen38` here would stop it through `Conflicts=`". `fixes.py` `restart_qwen38` (`:87`) gets the same guard at the top of its run branch (`if llm_mode.b70_owned_elsewhere(path, time.time()): return False, "B70 owned by another mode; refusing"`), reading the path from `cfg["qwen_heal"]["llm_mode_file"]` with the `DEFAULT_MODE_FILE` fallback; it is disabled in `config.toml:97` today, so this is belt and braces.

4. `strata_heal`: not in this slice. Until it exists Strata runs only in a lease or with the unit's own `Restart=on-failure` (research doc step 5: "Until these exist, Strata runs only inside a lease with Sentinel's heal held off"). File a Plane item (SENTINEL project) for a `strata_heal` in the `qwen_heal` pattern: two agreeing signals, `XE_EVENT_RE` (`qwen_heal.py:59-60`), backoff, `/health` + `/slots` with the key.

#### Tests

- `apps/sentinel/tests/test_llm_strata_aware.py` (copy `test_llm_cyber_aware.py:1-49` setup; `_probe_out` gains `"strata": "active"` and `"strata_health": '{"status": "ok"}'`): strata active suppresses `QWEN_SIGS` and adds `llm:strata_mode`; strata active with `"strata": "inactive"` adds `llmvm:strata-flash:inactive` critical; strata active with `strata_health` not ok adds `llm:strata_unhealthy`; mode normal with `"strata": "inactive"` adds nothing strata-related; `until: 0` never expires; `0 < until <= now` brings back qwen38 findings and adds `llm:strata_mode_overdue`.
- `apps/sentinel/tests/test_qwen_heal_strata.py` (copy `test_qwen_heal_cyber.py:15-68` helpers): active strata (until 0 and until future) stands down even when forced and down for many cycles (mirror of `:15`); expired strata heals; mode normal heals (existing).
- Existing cyber tests must stay green unchanged.

#### Done when

With the mode file saying `strata`, a full Sentinel cycle raises `llm:strata_mode` only, `qwen_heal` logs nothing and runs no restart; with `strata-flash` stopped under that mode, `llmvm:strata-flash:inactive` is critical.

#### Effort

1 day with tests.

### I6.3 arc-queue

#### Where

`infra/arc-llm-vm/etc/arc-queue/config.yaml:18-26` (worker `b70` with `params: {family: qwen38, dialect: llamacpp}`), `:73-80` (worker `cyber`, same params), `:81-82` (`metrics_extra`), `:83-92` (routing globs). `infra/arc-llm-vm/arc-queue/app/core/params.py:17-25` (`FAMILIES["qwen38"]`), `:40-49` (`_DRAFT_FIELDS`, `DIALECTS["llamacpp"]` with `budget_field: "reasoning_budget_tokens"` and `draft_n_max`), `:67` (`EFFORT_ALIASES = {"high": "xhigh"}`), `:89-129` (`apply_params`; `:116-117` writes the budget field, `:126-128` sets `speculative.n_max`). `app/config.py:25-42` (`ParamsConfig` validates `family` and `dialect` against the tables), `:80-95` (`ModelConfig`: `url`, `api_key_file`, `backend_model`, `params`, `readiness`, `metrics`, ...). `app/core/dispatcher.py:571-578` (`_backend_payload`: params applied only when `self._cfg.params`). `app/api/ops.py:80-85` (metrics relayed when `metrics` is true or dialect is `llamacpp`). `app/backend/openai_backend.py:144-152` (health: 200 ok, 503 loading, else down). Tests: `tests/unit/test_params.py` (pure table tests), `tests/unit/test_config_load.py` (`test_default_yaml_...`). Test command: `cd infra/arc-llm-vm/arc-queue && uv run --extra dev pytest -q` (`infra/arc-llm-vm/AGENTS.md:45`).

#### Current behaviour

`apply_params` for `llamacpp` injects `reasoning_effort: "low"` (`FAMILIES.qwen38.defaults`), maps `high` to `xhigh` (`:67`, "llama.cpp rejects high"), writes `reasoning_budget_tokens` (`:116-117`) and `speculative.n_max` (`:126-128`). A worker without `params` forwards the body untouched (`dispatcher.py:573`; `params.py:5-7`). Strata's endpoint: `reasoning_effort` takes `none|low|medium|high` and `xhigh` is a 400 (`serve/frontend.py:133-142`: "unknown values are a 400"; `budget_effort` at `:151` maps a budget to `low|medium|xhigh` internally, but the request-side parser `effort_kwargs` is what rejects); `reasoning_budget_tokens` is accepted (`server.py:2759-2769`); `speculative.n_max` is unknown and ignored (not verified, I4(c)). So the llamacpp dialect is wrong for Strata in one place (`xhigh`) and wasteful in another (`speculative.n_max`).

#### Change

1. `params.py`: add a dialect, no new family (the model card defaults for Qwen3.8-Flash-Next are not in the repo; sampling comes from Strata's config `"sampling"` block, I3 step 7, so the worker should inject none):

```python
    "strata": {  # Strata's OpenAI endpoint (serve/frontend.py): reasoning_effort none|low|medium|high, budget field accepted, no llama.cpp draft fields
        "budget_field": "reasoning_budget_tokens",
        "thinking_off_kwarg": True,
        "drop": _DRAFT_FIELDS,
        "penalties": True,
        "draft_n_max": None,
    },
```

   and make the `high -> xhigh` alias dialect-specific: move `EFFORT_ALIASES` into the dialect rows (`"effort_aliases": {"high": "xhigh"}` for `llamacpp`, `{}` for `strata`, `omlx` keeps today's behaviour) and read `dia.get("effort_aliases", {})` at `:109`. That is the "gating of the llama.cpp-only fills per worker" the research doc asks for: `speculative.n_max` is never set when `draft_n_max` is `None` (`:126`), and `reasoning_effort` is passed as given.

2. `params.py` family: a `flash` family row is needed because `ParamsConfig` requires a known family (`config.py:38-40`) and `apply_params` indexes `FAMILIES[family]`. Add `"flash": {"thinking": {}, "non_thinking": {}, "penalty": {}, "penalty_modes": (), "defaults": {"max_tokens": 8192}}` so nothing but `max_tokens` is injected (Strata's `"sampling"` config block supplies the rest). Check `apply_params:95-101` tolerates empty presets (it iterates `preset.items()`: yes).

3. `config.yaml`: a worker and a route:

```yaml
  strata:                                   # Strata SYCL engine on the B70, 127.0.0.1:8097 (strata-flash.service), only while qwen38 is unloaded (steward llm-mode strata)
    url: "http://127.0.0.1:8097"
    api_key_file: "/etc/arc-queue/strata-api-key"   # 0600 ubuntu; the same key as /etc/homelab/secrets/strata-flash.env, read once at startup
    backend_model: "flash"
    params: {family: flash, dialect: strata}
    metrics: true                           # Strata serves /metrics (serve/server.py:4487); not llamacpp, so opt in
    max_concurrency: 1
    max_queue_depth: 16
    sync_timeout_seconds: 3600
    read_timeout_seconds: 1800
    priority: 5
```

   under `models:`, and under `routing.models` before the catch-all comments: `"flash*": strata`. `api_key_file` is how the Mac workers carry a bearer key (`config.yaml:37`, `config.py:86`); arc-queue runs as `ubuntu`, so the key file is a second 0600 copy for `ubuntu` (placed by stdin like the unit's). Never in a kid alias (same note as `cyber*`, `:90`).

4. Readiness: do not add a `readiness:` block (that path is for remote backends and turns retries into 503s, `README.md:86`); the default behaviour re-queues while `/health` is 503 or down (`README.md:20`), which is right for a unit that is simply not started in normal mode: a `flash` request in normal mode fails after `max_attempts` (`policy.max_attempts: 3`, `config.yaml:134`) with a backend-down error, like `cyber` today.

5. Deploy: `infra/arc-llm-vm/deploy.sh --apply` copies `arc-queue/` and `etc/arc-queue/config.yaml` (`deploy.sh:22, 28`); then `sudo -n systemctl restart arc-queue` under an `llm:*` lease with empty queues (`skills/llm-vm-ops/SKILL.md:27`).

#### Tests

- `tests/unit/test_params.py`: `apply_params({"messages": MSGS, "reasoning_effort": "high"}, family="flash", dialect="strata")` has no `speculative.n_max`, keeps `reasoning_effort == "high"`, has `max_tokens == 8192` and no sampling keys; a no-think request gets `chat_template_kwargs.enable_thinking == False`; `reasoning_budget_tokens` passes through; `llamacpp` still maps `high` to `xhigh` (existing test `:14-29` stays).
- `tests/unit/test_config_load.py`: the default YAML has worker `strata` with `params.dialect == "strata"` and route `flash*` to `strata` (mirror of `test_default_yaml_has_single_b70_worker_on_moved_backend`, `:83`).

#### Done when

`curl -s 192.168.0.211:8080/v1/chat/completions -d '{"model":"flash",...}'` answers from Strata in strata mode and gets a backend-down error (not a 404 route error) in normal mode; `/metrics` on :8080 carries the `strata` labelled series.

#### Effort

0.5 day.

### I6.4 `inference-power-vm` UNITS

#### Where

`infra/arc-llm-vm/sbin/inference-power-vm:5` (`UNITS="qwen38 fast-llm cyber-llm"`), `:8-15` (shed: stops the running ones, remembers them), `:16-22` (restore: starts exactly those). Test: `infra/arc-llm-vm/tests/inference-power-vm.test.ts:1-42` (runs only in a container; fakes `systemctl is-active` from `/rec/active`, records the rest).

#### Change

`UNITS="qwen38 fast-llm cyber-llm strata-flash shieldgemma"`: `strata-flash` so a UPS shed stops the Strata engine and restores it (research doc step 5); `shieldgemma` because it is the other always-on B70 workload and restore must bring back exactly what ran. Order matters for `restore` only in that `systemctl start --no-block` of `strata-flash` and `shieldgemma` together would let `Conflicts=` stop one: shed records what was running, and in strata mode shieldgemma is not running, so the recorded set is consistent by construction. Note this in the comment on `:5`.

#### Tests

Add to the test file, following `:35-42`: `setup(["strata-flash", "fast-llm"]); run("shed")` records `systemctl stop fast-llm strata-flash` (order follows `UNITS`); `run("restore")` records `systemctl start --no-block fast-llm strata-flash`. A second case with `["qwen38", "shieldgemma", "fast-llm"]` restores all three. Run as the header says (`docker run --rm -v .../infra/arc-llm-vm:/v:ro -w /v/tests node:24-trixie-slim node --test inference-power-vm.test.ts`; the test refuses outside a container, `:10`).

#### Done when

The two new cases pass; `deploy.sh --apply` installs the script to `/usr/local/sbin/` (`deploy.sh:30`).

#### Effort

0.1 day.

### I6.5 Documents touched with the integration

`skills/qwen38-endpoint/SKILL.md` and `infra/arc-llm-vm/AGENTS.md` (the new unit, port 8097, the mode), `skills/llm-vm-ops/SKILL.md:35` table (a `strata-flash` row), `skills/proxmox-host-ops/SKILL.md:36` (the helper's actions), `docs/plans/cyber-mode.md` (the third mode), `infra/arc-llm-vm/VERSIONS.md` (Strata sha, llama.cpp sha, ocloc, model and MTP revisions). Every PR title carries its Plane item (`[LLM-n]` for the engine and bench work, `[SENTINEL-n]`, `[STEWARD-n]`, `[INFRA-n]` for the unit and power script).

---

## Not verified (collected)

- The sysfs attribute `b70-power-cap.sh` writes (`power1_cap` per `skills/llm-vm-ops/SKILL.md:46`); the script is not in the repo.
- Whether ggml-org llama.cpp `3cf03257`'s `gguf-py` knows `GGMLQuantizationType.Q2_0` (needed by `mtp_pack.py:36`); the fallback is `setup.py`'s own `LLAMA_CPP_COMMIT` (`setup.py:3138`).
- The Hugging Face API field for the LFS oid (`lfs.oid` in the tree listing, or `X-Linked-Etag` on a HEAD).
- Whether the engine needs a draft vocabulary file beside `--mtp` (setup copies one, `setup.py:4022, 4084`).
- How a config sets `aliases` for `/v1/models` (`server.py:4559`), if `flash-next` is wanted as a second name.
- Whether Strata rejects unknown top-level request fields (no whitelist found; `speculative.n_max` is dropped by the new dialect anyway).
- Whether `evals.py --suite` takes a comma list or one value.
- `requirements.txt` pins on Python 3.14 (use 3.12).
- `shieldgemma`'s restart after `normal`: `Conflicts=` only stops units; the helper arm `start-shieldgemma` covers it, but the owner may prefer `shieldgemma` to be started by `normal` unconditionally (today nothing starts it after a cyber window either).
