# B70 tuning: every switch, what it changes, and how to measure it

Fork `sam-fakhreddine/strata`, branches `b70*` (see `docs/B70_RUNBOOK.md` section 1). This file is the reference for the knobs: the engine flags that matter on one Arc Pro B70 with Qwen3.8-Flash-Next, the switches the port already had, the switches our branches add, and the measurement protocol that makes two numbers comparable. Expected effects are marked: (M) measured by someone on an Arc, (E) estimated from the code or a profile, (U) unknown. Nothing in this file was measured on the owner's card as of 2026-10-09.

## 1. Measurement protocol

Two numbers are comparable only when they come from the same procedure. The author's procedure (`docs/INTEL.md`, "Decode round 2" and the B70 IQ3_S section) is the one every row below assumes:

- One engine process per side of a comparison, started fresh, one warm-up request discarded.
- Interleaved pairs: off, on, off, on. Medians of at least 3 pairs; 5 when the difference is under 5%.
- Decode: the 20-token prompt (Fibonacci) and the 2,185-token prompt (code and prose), 256 greedy tokens (`--max-new 256 --greedy`), the `tok/s` of the generation phase. Prompt: the 2,185-token and the 8,000-token prompt, `prompt tok/s`. Context: the 40,000- and 128,000-token prompts when a change touches the prompt buffers or borrowing. Token id files: `bench/results/2026-10-07-community-arc-b65-v01402/benchy-short.ids` (20 ids) and `benchy-long.ids` (2,184 ids) in the fork; the longer ones are the author's `sycl/bench/v1` set.
- Output identity: `diff` of the printed token ids over the whole run. A switch marked bitwise must give identical ids. A switch marked rounding-level may show a near-tie flip after N tokens with the same text; it must pass the 10-prompt `mg_norepeat` gate (`docs/INTEL.md`) and a quality comparison before it is kept.
- Instrumentation, all per request, in the engine's stderr:
  - `STRATA_DECODE_TIMING=1`: `strata decode timing: N windows, avg T, tokens/window, ms/window = verify (...) + commit + draft`; with d6 also `graph N nodes/window, X us/node`. The verify / commit / draft split is host-side around the three calls.
  - `STRATA_PREFILL_TIMING=1`: per-chunk shares of the prompt path (GEMMs, dequant, host grouping, hyper-connection reads, attention).
  - `STRATA_VERIFY_NODES=1`: the node count of each captured window graph (kernels are counted, not named, on SYCL).
  - `STRATA_MIRROR_STATS=1` (d7): routed entries served from VRAM, the mirror, or neither, plus the worst layers; `=2` prints every layer.
  - `--stage-timing`: the stage profiler.
  - unitrace (`sycl/tools/Dockerfile.unitrace`, or a native install): `unitrace -d` around one request gives per-kernel device time; this is what ranks the decode items. The kernel names to look for are in the per-switch rows below.
- A D1 OFF binary reads 0 in every timing line by design. Use the API's `timings` (`sycl/tools/strata_timings.py`) or the wall-clock tok/s the engine prints for it.
- Record every number with: branch and commit, switch values, the exact command, quant, resident expert count, mirror size, driver (NEO) and oneAPI versions, power cap, and whether the card was single-tenant. The format is a `bench/results/<date>-b70-<topic>/README.md` as the community reports use.

Noise sources on this VM: the 5060 Ti's Gemma server shares the PCIe root complex and the CPU; the 180 W power cap (the card's own is higher; the cap is a house rule, keep it fixed across every comparison); JIT compile on the first request (the warm-up covers it); the page cache state after the mirror fill (d9's `STRATA_FILL_FADVISE=1` makes it reproducible).

## 2. Engine flags that matter on this card

| Flag | Recommended | Why | Trade-off |
|---|---|---|---|
| quant | IQ2_XS first | keeps all 24,576 experts; 18.3k resident in 32 GB, 8.4 GiB mirror (M, author); the only Flash-Next quant with a B70 number over 50 tok/s | 2-bit quality; run the homelab `llm-evals` suite before routing traffic |
| | Coder IQ1_M as the speed reference | all experts resident, no mirror; the author's 70 to 78 tok/s figure (M) | a different model; only a reference |
| | IQ3_XXS as the second experiment | about 15k resident, 18 GiB mirror; between IQ2_XS and IQ3_S (E) | needs about 25 GB RAM for the engine; swap must stay unused |
| | not IQ3_S at 48 GB RAM | 26 GiB mirror, 33 GB RAM (E) | the VM would need 64 GB (owner-gated) |
| `--stream-experts` | always | the port's expert path: cache in VRAM, mirror in RAM, in-kernel reads over PCIe | none |
| `--expert-cache auto` | always | fills VRAM down to the reserve | with `--vram-reserve-mib` too small the start says `LOW` |
| `--vram-reserve-mib` | 1024 at 32K context | setup's value for 32K; the KV and prompt buffers are allocated after the cache | 2048 or 3072 at 80K+ contexts |
| `--max-context` | 32768 | fits the 1 GB INT8 KV beside the cache | above 32K the prompt path borrows cache slots by default |
| `--kv int8` | yes | halves the KV; the port's int8 attention kernels | `k8v4` has no streaming yet |
| `--prefill` | 4096 | the author measured 980 to 1,002 prompt tok/s with 4096 against 730 with `auto` (2,048-token chunks) on IQ3_S (M); 8192 was 1,037 to 1,117 but costs cache slots (700 fewer at 4096 already, decode 31.4 to 30.5) | bigger chunks take VRAM from the expert cache |
| borrowing | own buffers at 32K | `--prefill-borrow` | the prompt path owns about 2.9 GiB of buffers at `--prefill 4096`; the engine borrows expert-cache slots for them by default only above 32K of context (`generate.cpp`: `no_prefill_borrow = max_context <= 32768` unless a flag was given). `--prefill-borrow` forces borrowing at any context: about 1.5k more IQ3_S experts (700 slots at IQ3_S were worth about 3 tok/s to the author, M), refilled from NVMe after each prompt (about 1.4 s) | the refill after every prompt; `--no-prefill-borrow` is the opposite |
| `--spec 4 --spec-min-p 0.5 --mtp <rt>` | yes | the MTP draft layer: 42.7 tok/s against 20.8 with the suffix drafter alone on the Coder (M); 66 to 90% of drafts accepted, 2 to 3.65 tokens per round | `--spec 6`: +6.5% on code, -1.5% on prose (M); `--spec 2` worse |
| `--ple-io direct` | yes | reads the 28.8 GB PLE table from NVMe with O_DIRECT; `ram` needs 28.8 GB of RAM and buys time-to-first-token, not throughput | `ram` only if RAM is free and TTFT matters |
| `--expert-profile data/expert-profile.bin` | yes | the resident set in profile order | the Coder has its own file |
| `--expert-profile-save <file>` | yes, from day one | the resident set follows our traffic after a week | the file replaces the shipped profile on the next start |
| `--adapt-every` | leave default (4); d8 turns it off automatically with a mirror | under `NO_HOST=1` the tier never swaps anyway (usage counters are only written by the host loop) | see D8 |
| `--kv-resident 32768` | only above 64K of context | KV streaming: whole KV in pinned RAM, attended window in VRAM; 256K measured at 56 tok/s on the Coder (M) | more RAM |

Environment the launcher sets (section 5.2 of the runbook): `STRATA_VERIFY_DEVICE_PLAN=1` (the device-built verify plan), `STRATA_VERIFY_NO_HOST=1` (required with a mirror on xe), `STRATA_STAGER_THREADS=8` (8 vCPUs), `SYCL_CACHE_PERSISTENT=0`, `ZES_ENABLE_SYSMAN=1`, `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`.

## 3. Switches the port already has (Tier 0: no code, one bench window)

| Switch | Default | Try | Expected | Measure with | Notes |
|---|---|---|---|---|---|
| `STRATA_SH_STREAM` | fork the shared expert onto a side queue per layer (3 barrier nodes a layer) | `=0` | +30% decode on HIP gfx1030 (M); unknown on Xe2 (U); tokens identical | decode tok/s, `STRATA_VERIFY_NODES` (144 fewer nodes a round) | `sycl/src/core/verify.cpp:98-110` |
| `STRATA_LFUSE` | off | `=1` | about 190 fewer nodes a round upstream; "all bitwise" upstream (U on SYCL) | decode tok/s, node count | inert for IQ2_XS with a mirror: the fusion applies on the all-resident path; measure on the Coder, expect nothing on IQ2_XS |
| `--spec 6` | 4 | 6 | +6.5% code, -1.5% prose (M) | decode tok/s on both prompts | acceptance rate in the timing line |
| `STRATA_MIRROR_MIB` | MemAvailable minus 4 GiB at start | an explicit value (9216 for IQ2_XS) | no change in speed; no squeeze of a later co-tenant | `free -m` after start | d9 changes the default's source, not the flag |
| `STRATA_STAGER_THREADS`, `STRATA_IO_THREADS` | 2 to 3 stager threads on 8 to 12 vCPUs; `IO_THREADS` 64 | 8, 64 | prompt path with streamed experts (E) | `STRATA_PREFILL_TIMING` stage share | `sycl/src/prefill/prefill.cpp:984-992`, `direct_file.cpp:61-67` |
| `SYCL_UR_USE_LEVEL_ZERO_V2` | adapter default | `=0` and `=1` | U | decode tok/s, one run each | p1's `=2` form hung once under v2 |
| `UR_L0_USE_IMMEDIATE_COMMANDLISTS` | adapter default | `=1` | U | decode tok/s | |
| `UR_L0_USE_COPY_ENGINE` | adapter default | `=0` | U | decode tok/s | the window's small copies may be faster on the compute engine |
| `UR_L0_IN_ORDER_BARRIER_BY_SIGNAL` | adapter default | `=1` | U | decode tok/s | |
| `STRATA_VERIFY_ALL_SLOTS` | off | `=1` once per driver or kernel change | catches the xe aliased-pages class of bug | start log | reads every filled slot back against the GGUF |
| `STRATA_MMVQ_A2` | aligned-load Q6_K kernel on | `=0` as a sanity check | decode drops by about 10 tok/s (M) | decode tok/s | proves the aligned kernels are active |
| `STRATA_PREFILL_FIRST` | 256-token first chunk only when the PLE rows are not yet read | leave | the short first chunk cost a whole extra expert stream per prompt (M, fixed) | | |

## 4. Switches our branches add

Each row: the branch, the switch, where it acts, what it changes, whether the bytes are the same, the expected effect, and the measurement. Full designs are in `docs/B70_WORKPLAN.md` under the item id.

### D1. Profiling queues (build-time)

- Branch `b70-d1-profiling`; CMake `-DSTRATA_SYCL_PROFILING_QUEUES=OFF` (default ON = today).
- What: the migrated code defines dpct's `DPCT_PROFILING_ENABLED` in 119 of 123 files, so every queue carries `enable_profiling` and every event record is a timestamp event. On Level Zero that can keep the adapter off its cheaper in-order path. With 2,000 to 2,600 launches per decode window the per-node cost (about 5 us of gap per node, 12 to 31% of a 40 ms round, E) is the first thing to measure.
- Bytes: identical. The OFF binary's timing lines read 0 and print "(profiling off)" once; `strata::prof_ns<Param>(event)` returns 0.
- Expected: U; the hypothesis is a few percent of decode. If `us/node` (d6's field, in the ON binary) is already under 3 us, this item is small.
- Measure: two binaries (`build-all`, `build-all-noprof`), decode tok/s via the API timings, tokens identical.

### D4. Decode-once lane path for IQ2_XS and IQ1_M

- Branch `b70-d4-iq2xs-multi`; `STRATA_IQ2XS_MULTI=1`, `STRATA_IQ1M_MULTI=1`.
- What: the expert gate/up and down dots for formats 17 (IQ2_XS) and 29 (IQ1_M) take the `Multi<>` path the other five expert formats already have: the weight part is decoded once per lane and applied to up to 4 routed entries, instead of once per entry. For mirrored experts this is also one PCIe fetch per 4 entries instead of per entry.
- Bytes: bitwise (same dp4a order, same integer expression, same reduction; `iq_multi_parity` check 3 requires memcmp equality against both the per-entry path and the old kernels).
- Expected: the gate/up call is 52 to 61 us per layer in a decode window today (M, Coder); the multi path saved a visible share on the formats that have it (E). Expect more on IQ2_XS with the mirror than on the all-resident Coder.
- Measure: `iq_multi_parity --bench` (off/on lines for groups of m = 1, 2, 4, 8), `NATIVE_BENCH=1 native_expert_parity` (the port's per-layer timing; there is no `native_expert_bench` target in this fork), unitrace share of `native_gu_port` kernels, decode tok/s, tokens identical.

### D6. Small decode items

- Branch `b70-d6-small`; `STRATA_INPUT_COPY_MULTI=1`, `STRATA_ARGMAX_MULTI=1`; `STRATA_DECODE_TIMING=1` now prints the node count.
- What: the window's three input copies become one `copy_from_mapped_multi`; the argmax over the vocabulary uses the multi-block kernel (`argmax_rows_wanted`) instead of one block; the timing line adds `graph N nodes/window, X us/node` from `dpct::experimental::get_nodes`.
- Bytes: bitwise.
- Expected: 2 nodes fewer per window from the copies; the argmax is one kernel of a 40 ms round (E: small, under 1%). The node count is the useful part: it tells whether D1 and the node-reduction switches are worth anything.
- Measure: decode tok/s; `us/node` before and after every other decode switch.

### D7. Mirror statistics (diagnostic)

- Branch `b70-d7-mirror-counter`; `STRATA_MIRROR_STATS=1` or `=2`.
- What: `resident_plan_kernel` gains a stats pointer; per layer it counts routed entries served from VRAM, from the mirror, and from neither (a miss the plan could not serve), plus the group count. One line per request: `strata serve: mirror stats: VRAM N, mirror N, neither N routed entries over N windows; worst layers ...`; `=2` adds every layer.
- Bytes: identical output; the counters are atomics on a device buffer, a small cost per window.
- Expected: for IQ2_XS with 18.3k of 24.6k resident and a profile-ordered cache, the mirror share should be well under the 25% the expert count suggests, because the profile puts the hot experts in VRAM. `neither` must be 0. A high mirror share on some layers is the input for `--expert-profile-save` and for the Tier 2 prefetch item.
- Measure: one request each on the three prompts; turn it off for the final numbers.

### D8. Adaptive tier guard

- Branch `b70-d8-adapt-guard`; on by default; `STRATA_ADAPT_WITH_MIRROR=1` keeps the tier.
- What: with a host mirror under the device plan, an expert the adaptive tier evicts from VRAM is never added to the mirror table, so the plan would route it to "neither". The guard prints a WARNING and sets `--adapt-every 0`. Under `STRATA_VERIFY_NO_HOST=1` the usage counters that drive the tier are never written (they come from the host service loop), so today the tier never swaps and the hazard is latent; the guard makes that explicit and protects a future run with the host loop on.
- Bytes: identical in practice.
- Expected: no speed change. A single A/B pair with `STRATA_ADAPT_WITH_MIRROR=1` confirms it.
- Measure: `STRATA_MIRROR_STATS=1` must show `neither 0` in both arms.

### D9. Mirror cap and page cache

- Branch `b70-d9-mirror-cap`; the cap change is always on; `STRATA_FILL_FADVISE=1` is opt-in.
- What: the default mirror cap now comes from `strata::core::detail::host_available_memory` (cgroup-aware) instead of `/proc/meminfo` `MemAvailable` minus 4 GiB; the log says `mirror cap N MiB from <source>`. With `STRATA_FILL_FADVISE=1` each fill read is followed by `posix_fadvise(DONTNEED)`, so the 35 GB model does not stay in the page cache after the fill (the mirror already holds the bytes the engine needs, pinned). The O(N squared) miss-list build is O(N).
- Bytes: identical.
- Expected: no speed change in steady state; a faster or equal fill (the page cache would otherwise evict the PLE table's pages during the fill); `free -m` shows about 35 GB more available after start with the switch on. Matters on the 48 GB VM where fast-llm's GGUF also lives in the page cache.
- Measure: fill time in the start log, `free -m` after start, decode tok/s unchanged.

### P1. The grouping bubble in the prompt path

- Branch `b70-p1-grouping-bubble`; `STRATA_GROUP_ASYNC=1` (polled sequence number) or `=2` (event wait).
- What: per MoE layer the prompt path drains the queue (`m.cs->wait()`) and then sorts the routed ids on the host; the GPU idles until the slot and source tables come back. With the switch, the ids copy is issued right after the router, the shared expert's projections are enqueued after it, and the host waits only for the ids copy, so the shared expert runs while the host sorts. `=2` waits on the copy's event (the form that hung the stager under the Level Zero v2 adapter once; kept for the comparison).
- Bytes: identical (ordering only).
- Expected: the host-grouping share of the prompt timeline, a few percent of a 4K chunk (E from the author's breakdown). More on long prompts.
- Measure: `STRATA_PREFILL_TIMING=1` host-grouping share; prompt tok/s on 2,185 and 8,000 tokens; tokens identical; `=2` last, with a hang watch.

### P2. Dequant work-group size

- Branch `b70-p2-dequant-occupancy`; `STRATA_DEQUANT_WG=2`, `4` or `8` (anything else = today's launch).
- What: the expert dequant that feeds the prompt GEMMs launches one 32-lane group per 256-value superblock today (12,800 groups per gate/up matrix). With the switch, a work-group of 32 x N lanes with `reqd_sub_group_size(32)` handles N superblocks, one sub-group each with the same lane mapping; a 32-lane group is one sub-group on Xe2, and the hardware wants 4 to 8 per thread group to hide store latency.
- Bytes: bitwise (`iq_multi_parity` checks wg 1 against 2, 4, 8 for every expert format at two shapes; `dequant_bench [type] [experts] [wg]` prints the FNV hashes).
- Expected: the dequant runs at about 160 GB/s today, 42 us per gate/up matrix (M, author's profile), a quarter of the card's bandwidth; if occupancy is the limit, 2x on that kernel is possible (E). The dequant is a modest share of a 4K prompt chunk, so a few percent of prompt tok/s.
- Measure: `dequant_bench 21 0 1`, `... 4`, `... 8` (GB/s and hashes), unitrace device time of `dequant_gu_wg_kernel_sg` against `dequant_gu_kernel_*`, prompt tok/s, tokens identical.

### P6. Fused residual read in the hyper-connection

- Branch `b70-p6-r-pass-fuse`; `STRATA_GR_FUSE_READ=1`; honours `STRATA_PF_SWITCH_MIN_T` (chunks of that many tokens or more; default 0 = always); not taken with `STRATA_PREFILL_BF16X2=1` or `STRATA_GR_UNFUSED=1`.
- What: the hyper-connection read of the prompt path makes two FP32 passes over the residual R (168 MB at a 4K chunk): the norm pass and the mix pass. The mix pass now reads x = R * rs * w from the BF16 image the norm already wrote for the GEMMs (`gr_mix_x16`), 84 MB instead of 168 MB.
- Bytes: rounding-level. x carries BF16's 8 mantissa bits, the same rounding the down and inject projections already see. Quality-gated like `STRATA_PF_HCDOWN`: same text, near-tie flips allowed, the 10-prompt `mg_norepeat` gate and an output comparison.
- Expected: the hyper-connection reads are 4.6% of the prompt timeline (M, author); the mix pass moves R + gated + mixed = 168 + 168 + 84 MB, so saving 84 MB is a quarter of the pass: roughly 1% of prompt time if bandwidth-bound (E). Small; it is in the set because it is cheap to flip.
- Measure: `gr_prompt_read_parity`; prompt tok/s on the 4K prompt; the quality gate.

### Infra (no switch)

- Branch `b70-infra`: the native launcher, the unit, the config example, the timings driver. See the runbook.

## 5. Build-time choices

| Option | Default | Effect |
|---|---|---|
| `-DSTRATA_SYCL_AOT=bmg-g31` | off (JIT) | compiles device code ahead of time; start 45 s instead of 90 s; needs `intel-ocloc` |
| `-DSTRATA_SYCL_PROFILING_QUEUES=OFF` | ON | D1; see above |
| `-DSTRATA_SYCL_PARITY=OFF` | ON | skips the parity tests and benches; keep ON for a tree you test with |
| `-DSTRATA_GGML_DIR=<llama.cpp>` | empty (fetch) | always set it to the pinned checkout |
| `-fsycl-default-sub-group-size=32` | set in `sycl/CMakeLists.txt` | 32 lanes everywhere; Xe2 native is 16. A SIMD16 build of the multi dots is Tier 2 (T2b), after D4 |

## 6. Order of experiments in the first window

Each step is a go/no-go for the next; a step that gives no change is recorded and the switch left off.

1. Baseline on `build-b70/strata`: decode on the 20-token and 2,185-token prompts, prompt tok/s on 2,185 and 8,000, `STRATA_DECODE_TIMING=1`, one unitrace run. Then `build-all/strata` with no switches: must match.
2. `STRATA_MIRROR_STATS=1` once per prompt (D7): read the mirror share and `neither 0`. This decides whether the mirror or the kernels bound decode.
3. Node cost: `STRATA_DECODE_TIMING=1` on `build-all` gives `us/node`. If above 4 us: run D1 (`build-all-noprof`) next, then `STRATA_SH_STREAM=0`. If below 3 us: skip to step 4.
4. `STRATA_IQ2XS_MULTI=1` (D4). The expected largest single decode item for IQ2_XS.
5. `STRATA_INPUT_COPY_MULTI=1 STRATA_ARGMAX_MULTI=1` (D6), together; then `--spec 6`; then `--prefill-borrow` (decode tok/s against the refill time per prompt).
6. Prompt path: `STRATA_DEQUANT_WG=4`, then `8` (P2); `STRATA_GROUP_ASYNC=1` (P1); `STRATA_GR_FUSE_READ=1` (P6) last, with its quality gate.
7. The Level Zero adapter knobs, one at a time, decode only.
8. The combination of every switch that won, as one run, against the baseline. Tokens identical except for P6.
9. `STRATA_FILL_FADVISE=1` (D9) for the production config regardless of speed, if the fill time is not worse.

What this does not cover, by design: a different quant per request, two cards, the Tier 2 items (prefetch of predicted misses, INT8 expert weights through oneDNN, the SIMD16 build), the Coder as a production model. Those are in `docs/B70_WORKPLAN.md` with their gates.

## 7. Known facts that change how to read a number

- `STRATA_LFUSE=1` is inert for IQ2_XS with a mirror: the fusion is on the all-resident shared-expert path. A "no change" on IQ2_XS says nothing; test it on the Coder.
- `--prefill-borrow` and `--no-prefill-borrow` both exist in the port. Without either, borrowing is on only above 32K of context, so the standard 32K config does not borrow. For a quant that does not fit, `--prefill-borrow` is the first flag to add (section 8).
- The adaptive tier's usage counters are written only by the host service loop, which `STRATA_VERIFY_NO_HOST=1` disables. Any "adaptive" effect seen under the recommended environment is noise.
- `native_expert_bench` does not exist as a target in this fork; `NATIVE_BENCH=1 native_expert_parity` is the per-layer timing.
- `sycl/tools/fixups.py` is not idempotent on a clean tree (two entries rewrite their target twice). The committed trees are correct; fix the two entries before re-running it after a re-migration from upstream.
- Every dpct queue carries `enable_profiling` in the default build (D1). Any timing read from an event is a real barrier; the timing lines themselves perturb what they measure by a small, constant amount. Compare like with like: both arms with the same timing switches.
- The power cap is 180 W by house rule (`b70-power-cap.service`), under the card's own limit. The author's numbers were taken without a cap. Expect a few percent less on the ALU-bound expert dots; never change the cap for a measurement.
- The 5060 Ti in the same VM is not part of any configuration here (the engine-plan document's verdict: no cross-process protocol for a layer or expert split; the vision encoder is the only candidate and is off).

## 8. Moving up a quant

The owner's direction (2026-10-09): take every token of speed and quality available, and move up from IQ2_XS if the card can carry it. What a bigger quant costs is mirror traffic: every expert that does not fit in VRAM is read over PCIe in the decode window. Everything below is sized for one B70 (32 GB) with 64 GB of VM RAM assumed.

| Quant | Bytes per expert | All experts | Resident at a 22 GiB cache | Resident at 26.5 GiB (`--prefill-borrow`, `--kv-resident`, reserve 512) | Mirror (RAM) at 26.5 GiB | Quality (ISTA's numbers) | Decode expectation |
|---|---|---|---|---|---|---|---|
| IQ2_XS | 1.44 MB | 35.5 GB | about 16.4k (67%) | about 19.7k (80%) | 7.0 GB | below the full model | 58 to 64 at Gen3 x8 (M), 70 to 78 at Gen5 (M, author) |
| IQ3_XXS | 1.75 MB | 42.9 GB | about 13.5k (55%) | about 16.2k (66%) | 14.6 GB | between | a 2x B70 split measured 66 (M); one card: between the two neighbours (E) |
| IQ3_S | 1.97 MB | 48.4 GB (experts) | 11.5k (47%, M) | about 14.4k (59%) | 20.0 GB | level with the full model | 30 to 41 on 200-token answers (M, maintainers, PCIe measured 13.3 GB/s), 48 to 57 on 256-token greedy code (M, community) |
| UD-IQ4_XS | about 2.4 MB | 59.5 GB | about 9.7k (40%) | about 11.8k (48%) | 30.6 GB | above | not measured by anyone; RAM-bound |

Resident counts are cache bytes divided by expert bytes; the author's measured 11.5k for IQ3_S at a 22 GiB cache anchors the column. RAM is the mirror plus about 7 GB for the engine, the PLE reads and the pack; IQ3_S at 26.5 GiB resident needs about 27 GB, IQ4_XS about 38 GB.

What makes the mirror cheap, in order:

1. **The link.** The author's B70 VM measured 13.3 GB/s host to device; this VM's root port is Gen5 x16. The engine prints a `PCIe probe` line at start: that number against 13.3 is the first-order correction to the author's IQ3_S figures (E: a doubled link halves the mirror share of a round, so the maintainers' 30 to 41 becomes 45 to 55 before any code change).
2. **The profile.** The resident set is chosen in profile order; `--expert-profile-save` makes it follow our traffic, so the mirror share by traffic is well under the share by count. D7 (`STRATA_MIRROR_STATS=1`) measures it: the `mirror` count over `VRAM + mirror` per request is the number to track across quants.
3. **VRAM for experts.** `--prefill-borrow` (+2.9 GiB at `--prefill 4096`), `--kv-resident 32768` (the KV in pinned RAM, +1 GiB), `--vram-reserve-mib 512` (+0.5 GiB, only if the start line never says LOW), a shorter `--max-context`. Together about +4.5 GiB, 2.3k more IQ3_S experts.
4. **Fewer fetches per entry.** The `Multi<>` lane path decodes a mirrored part once for up to 4 routed entries. IQ3_XXS, IQ3_S and IQ4_XS already have it; D4 adds it for IQ2_XS and IQ1_M.
5. **Prefetch (T2a, branch `b70-t2a-prefetch`).** Predict layer l+1's experts from layer l's input, copy the predicted mirrored ones into a per-layer VRAM slot ring with one coalesced copy kernel while layer l computes, and let the plan point at the slot. `STRATA_PF_SLOTS=N` (0 = off). The one item written for a quant that does not fit; its first measurement is the prediction hit rate (D7's `ring hits`) and the copy kernel's rate against the link.
6. **Prompt path.** A 4K prompt streams nearly every expert of every layer through the cache once; at IQ3_S that is 13k misses x 1.97 MB = 26 GB per chunk, 1 to 2 s of the link. `--prefill 4096` over `auto` (M, +34%), and the P-items, are what the prompt side has.

Quality gate before any routing decision: the homelab `llm-evals` suite (`extract`, `tier`, `tool`, `reason`) on IQ2_XS, IQ3_XXS and IQ3_S against the `pre-strata` qwen38 baseline, plus the 10-prompt `mg_norepeat` gate. ISTA's own figures put IQ3_S level with the full model and IQ2_XS below it; nothing has measured our tasks.

Order for the quant question in the windows: IQ2_XS baseline (window 1, with the `PCIe probe` and D7 numbers); IQ3_XXS in window 2 (fits 48 GB of RAM); IQ3_S once the VM has 64 GB (owner-gated: RAM from the host or docker-vm) with `--prefill-borrow` and `STRATA_PF_SLOTS` on, and the quality suite on each.

## 9. Making our own quant

Read from the file headers of ISTA's published files on 2026-10-09 (`tools/quantscope.py hf:ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF --all`, HTTP range reads, no download): the size names are recipe labels. Every GSQ-RCO file is already a per-layer mix of expert formats, allocated by their routing-consistent calibration:

| File | Routed experts, bits per weight | Expert formats by share of bytes | Dense (attention, head, shared experts) |
|---|---|---|---|
| IQ2_XS | 2.35 | IQ2_S 52%, Q2_0 32%, IQ2_XXS 13%, IQ1_M 3% | IQ4_XS mostly; head IQ4_XS |
| IQ3_XXS | 2.84 | IQ3_S 22%, IQ4_NL 20%, Q2_0 16%, IQ2_S 13%, IQ2_XS 11%, IQ2_XXS 9%, IQ3_XXS 9% | Q6_K / IQ4_XS / Q4_K; head Q5_K |
| IQ3_S | 3.33 | IQ4_NL 37%, IQ3_XXS 22%, IQ2_S 21%, IQ3_S 14%, Q2_0 4%, IQ4_XS 2% | Q6_K / Q5_K; head Q6_K |
| Q2_0 | 2.25 | Q2_0 100% | Q3_K / IQ4_XS; head Q5_K |

In every file the hyper-connections and the routers stay BF16 (the engine reads them natively) and the 28.8 GB PLE table is IQ4_NL (shard 2, identical across sizes). The engine's pack format is per layer (`native_experts.txt`: one line per layer with the gate/up type, the down type, the blob size and the shard), and the port's expert kernels take gate/up in IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S, IQ4_XS, IQ1_M and Q2_0 with down in IQ4_NL, IQ4_XS and Q2_0 (`iq_kernels.dp.cpp`, the `dq_dispatch` and `launch_gu_port` cases). The K-quant expert formats of Unsloth's files (Q4_K, Q5_K, Q5_1, Q8_0) have NVIDIA-only prompt kernels and are not an option on the B70.

Three ways to make a file of our own, in order of cost:

1. **Splice ISTA's own files per layer (recommended first).** Same model, same calibration, no quantization work. Build one shard 1 whose `blk.N.ffn_{gate,up,down}_exps` tensors for each layer N come from the IQ3_S, IQ3_XXS or IQ2_XS file (a layer's three tensors from the same file), the dense tensors from the IQ3_S file, shard 2 shared. The allocation is profile-guided by D7's per-layer counts: a layer whose routed entries are mostly served from VRAM costs VRAM bytes, so it takes the bigger version only while the cache budget holds; a layer mostly served from the mirror costs PCIe bytes per read, so it takes the smaller version. Any expert total between 35 and 50 GB is reachable, so the file can be sized to exactly what the card plus the RAM budget hold. Tooling: a splice script over gguf-py (read the three headers, copy the chosen tensors' byte ranges, write one GGUF with the IQ3_S file's metadata and the split keys), then `tools/iq_pack.py` as for any file; the pack already records a type per layer. Downloads: the three variants, about 80 GB beyond IQ2_XS (shard 2 is shared). Risk, not verified: whether GSQ-RCO calibrates layer l's experts on the quantized output of layers before it (sequential, as GPTQ does); if so, mixing files adds an error of the order of the quantization noise itself. The quality gate (section 8) decides; the splice is reversible by construction.
2. **Quantize from the BF16 checkpoint with llama.cpp.** `convert_hf_to_gguf.py` on `Qwen/Qwen3.8-Flash-Next` (about 250 GB of BF16 on disk), an importance matrix from calibration text (`llama-imatrix`; forward passes of a 125B MoE, hours on the VM's 8 vCPUs, faster with the B70 offloading under a lease), then `llama-quantize` with `--imatrix` and `--tensor-type blk.N.ffn_*_exps=<type>` per layer, keeping the hyper-connections and routers at BF16 and the PLE table at IQ4_NL (or `--compat-bf16` in `iq_pack.py` for whatever got quantized). What it buys over 1: a calibration set from our own traffic, and any bit allocation. What it costs: 2 to 3 days of hands-on and compute time, 400 GB of disk, and a quality gap: plain imatrix i-quants at equal bits are expected to sit below GSQ-RCO's routing-consistent allocation (ISTA's method is the research contribution behind these files). Only worth it if the splice fails its gate or a domain calibration is wanted.
3. **Expert pruning for our traffic, as the Coder does (256 of 512 experts per layer).** Research-grade (RCO calibration on a task mix); the Coder's own authors report 91% of SWE-bench Verified for code. Not planned.

Order: measure ISTA's IQ3_XXS and IQ3_S as published (section 8), read D7's per-layer map on each, then splice.
