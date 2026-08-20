# Performance evidence

This document records the measurements that justify `md5-many` backend and
scheduler choices. It is **not** a promise of absolute throughput: hosted and
virtualized CPUs change over time, frequency management adds noise, and the
same ISA can have very different instruction latencies on different
microarchitectures.

The durable rule is therefore:

> CPU capability tells us which backend *can* run; same-host measurements tell
> us which backend *should* run.

Keep user-facing dispatch descriptions in `README.md`, developer invariants in
`AGENTS.md`, and release changes in `CHANGELOG.md`. Put detailed benchmark
results and crossover evidence here.

## Measurement policy

Performance changes should be evaluated with the candidate and baseline on the
same CPU. Prefer a pinned schedulable CPU and alternate measurement order when
possible. The GitHub Actions performance guard uses ABBA ordering
(`base -> head -> head -> base`) and only blocks when both orderings independently
confirm a large regression.

For microarchitecture-specific work:

- compare the same public benchmark or forced backend before and after;
- inspect release machine code when the optimization depends on a particular
  instruction such as `VPTERNLOGD`, `VPROLD`, or `ANDN`;
- use forced backend A/B measurements to separate scheduler effects from the
  compressor itself;
- treat hosted-runner CPU names as observations, not as GitHub guarantees;
- do not generalize a crossover to another CPU family without measuring it.

Absolute MB/s and MiB/s numbers below were collected by different harnesses, so
only compare values within the same table or experiment. Relative percentages
are the primary evidence.

## Single-stream x86-64

### Intel Xeon Platinum 8573C (family 6, model `0xCF`)

The single-stream AVX-512VL backend uses XMM-width state with `VPTERNLOGD` for
MD5 Boolean functions and vector rotates. This is a latency optimization, not a
wide-lane throughput kernel.

Initial Rust measurements showed the AVX-512VL compressor beating the NoLEA
scalar compressor by roughly 7% per block, but trailing the reference
`md5-optimisation` implementation. Disassembly showed that the Rust path was
round-tripping A/B/C/D between scalar state and XMM registers at every block.
Keeping vector state live across consecutive blocks removed that overhead.

Measured public-path change relative to the original `0.1.0-alpha.2` AVX-512VL
bridge:

| Input | one-shot / streaming improvement |
| ---: | ---: |
| 512 B | about 3.7–3.9% |
| 1 KiB | about 4.2–4.3% |
| 4 KiB | about 4.6% |
| 64 KiB | about 4.6–4.7% |
| 1 MiB | about 4.7% |

For long single streams, pinned-cycle comparison against forced NoLEA was about
11% faster. A Criterion 1 MiB run after the vector-state change measured roughly
808 MiB/s for one-shot hashing and 816 MiB/s for RustCrypto-compatible
streaming. The upstream CC0 `md5-optimisation` AVX-512 implementation measured
about 816.7 MB/s on the same host, so the Rust path had effectively closed the
large implementation gap.

A later GitHub-hosted run independently reproduced the improvement on another
reported Xeon Platinum 8573C. The CI compared the published `0.1.0-alpha.2`
commit (`812a663`) with post-alpha.2 commit `1fd51b1` on the same runner using
the bidirectional ABBA guard. For `single-stream-1MiB/md5-many`, candidate
execution time was **5.58% lower** in the order-normalized ABBA result. Both
measurement orders agreed on the direction: the forward 95% confidence
interval was **-6.11% to -2.96%**, and the reverse order-normalized interval was
**-8.62% to -4.63%**. This is independent confirmation that keeping XMM state
live across blocks materially improves the Intel public path rather than being
a one-host artifact.

**Dispatch consequence:** supported Intel AVX-512F/VL CPUs may use the
single-stream AVX-512VL path. The vector-state core must remain inside an
AVX-512 target-feature context; outlining target-feature thunks or restoring
scalar state per block materially regresses throughput.

### AMD EPYC 9V74 / family 19h, model `0x11`

A virtualized EPYC 9V74 host exposed AVX-512F/DQ/BW/VL/VNNI/BF16. Capability
alone suggested that the XMM-width AVX-512VL single-stream backend was legal,
but forced A/B testing showed that it was a poor choice.

Pinned same-host measurements comparing the existing NoLEA scalar path against
a temporary build that selected AVX-512VL on any AVX-512F/VL CPU:

| Input | NoLEA scalar | AVX-512VL | AVX-512VL change |
| ---: | ---: | ---: | ---: |
| 64 B | ~350 MiB/s | ~279 MiB/s | **-20.2%** |
| 512 B | ~670 MiB/s | ~496 MiB/s | **-25.9%** |
| 1 KiB | ~708 MiB/s | ~528 MiB/s | **-25.4%** |
| 4 KiB | ~745 MiB/s | ~550 MiB/s | **-26.1%** |
| 64 KiB | ~761 MiB/s | ~552 MiB/s | **-27.5%** |
| 1 MiB | ~762 MiB/s | ~556 MiB/s | **-27.0%** |

The result was cross-checked by compiling the upstream CC0
`md5-optimisation` implementation on the same host:

| upstream backend | Throughput |
| --- | ---: |
| NoLEA + G shortcut (`NoL-G`) | ~760.4 MB/s |
| AVX-512 | ~562.6 MB/s |

That independent implementation showed approximately the same ~26% loss. Its
own design notes also call out Zen 4 vector bit-rotate latency as the reason
single-buffer AVX-512 is undesirable.

**Dispatch consequence:** AMD AVX-512 availability must **not** enable the
single-stream AVX-512VL path. Measured family 19h systems stay on the NoLEA
scalar compressor.

This is intentionally independent from the multi-buffer policy below: the same
CPU can be bad at a sequential AVX-512 dependency chain and excellent at many
independent AVX-512 chains.

## Multi-buffer x86-64

### AVX-512 on AMD family 19h

The EPYC 9V74 experiment above was repeated for `Md5Many`, comparing normal
AVX-512-capable dispatch against forced AVX2. Unlike the sequential compressor,
well-occupied multi-buffer AVX-512 was decisively faster because independent
message states expose lane-level and instruction-level parallelism.

Representative observations:

| Workload | AVX-512 advantage over forced AVX2 |
| --- | ---: |
| 32 messages × 1 KiB | roughly **+38–50%** |
| 64 messages × 1 KiB | roughly **+26–43%** |
| 32 messages × 64 KiB | roughly **+37–44%** |
| 64 messages × 64 KiB | roughly **+23–35%** |

Small/under-filled batches did not show the same advantage. The existing AMD
family 19h crossover therefore remains important: short 9–16-message batches
can stay on two AVX2 chains, while sufficiently occupied large batches retain
AVX-512.

**Scheduler consequence:** do not infer the many-buffer backend from the
single-stream choice. On measured family 19h hardware the intended policy is:

```text
single sequential stream       -> NoLEA scalar
small / under-filled batches   -> AVX2 where the measured crossover says so
well-occupied independent work -> AVX-512
```

### Intel family 6/model `0xCF` small batches

AVX-512 hosts normally keep 2–8-message batches on AVX2 because sparse ZMM
lanes do not automatically compensate for wider-vector costs. The measured
family 6/model `0xCF` host is a narrow exception: padded AVX-512 became useful
for equal batches at 512 B and above and for mixed batches whose shortest
message is at least 512 B.

**Scheduler consequence:** keep this exception model-specific until another
CPU is measured directly. The `x86-small-batch-*` Criterion groups exist to
protect this crossover.

### Incremental eight-stream AVX2 on Intel Xeon Platinum 8370C

Incremental multi-stream hashing was measured on an Intel Xeon Platinum 8370C
(family 6, model `0x6A`) with the normal benchmark profile (`thin` LTO, one
codegen unit). The initial stateful compressor gathered every message word
through the generic SIMD abstraction once per block. Reusing the one-shot AVX2
8x64-byte load/transpose and round schedule while loading and storing external
chaining states removed that bottleneck.

For eight 64 KiB streams, representative Criterion means were:

| Workload | Initial incremental | Stateful AVX2 | Same-host one-shot |
| --- | ---: | ---: | ---: |
| 32-byte updates | 234.6 MiB/s | **1.155 GiB/s** | 2.839 GiB/s |
| 4 KiB updates | 271.5 MiB/s | **2.569 GiB/s** | 2.839 GiB/s |

The 4 KiB streaming workload therefore reaches about **90%** of one-shot
throughput on this host. The remaining 32-byte gap is dominated by per-update
scheduling and partial-buffer handling rather than the MD5 compression loop.

**Implementation consequence:** keep equal eight-stream block-aligned updates
inside one AVX2 kernel boundary and use the same direct load/transpose schedule
as the one-shot path. Do not regress this path back to per-word generic lane
gathers.

### Incremental sixteen-stream AVX-512 on AMD EPYC 9V74

A later run on a benchmark host reporting AMD EPYC 9V74 (family 25/model 17) exposed the
same missing specialization at the 16-lane width. Before a stateful AVX-512
kernel, full 16-stream updates still used the generic per-word SIMD gather
path even though one-shot hashing already had a direct 16x64-byte transpose
kernel.

With 16 streams of 64 KiB each, the pre-specialization means were about
1.638 GiB/s for 32-byte updates and 3.311 GiB/s for 4 KiB updates. Adding the
stateful AVX-512 kernel and a lockstep equal-chunk fast path produced these
same-host Criterion means:

| Workload | Final incremental | Same-host one-shot | One-shot share |
| --- | ---: | ---: | ---: |
| 32-byte updates | **2.944 GiB/s** | 6.875 GiB/s | 42.8% |
| 64-byte updates | **3.959 GiB/s** | 6.875 GiB/s | 57.6% |
| 4 KiB updates | **6.753 GiB/s** | 6.875 GiB/s | 98.2% |

Relative to the generic 16-lane stateful path, 32-byte updates improved by
about **80%** and 4 KiB updates by about **104%**. The lockstep fast path alone
also moved the measured 64-byte case from roughly 3.10 GiB/s to roughly
4.0 GiB/s by avoiding repeated compaction and selected-lane bookkeeping.

**Implementation consequence:** a fully occupied AVX-512 incremental group
must use the direct 16x64-byte transpose/round kernel. Equal-length lockstep
updates should bypass the mixed-stream compaction scheduler. Small updates
still pay a kernel-boundary/state-materialization cost on every completed
block, so their remaining gap should not be treated as an MD5 round-kernel
problem. The reported CPU model is an observation of this host, not a runner or platform contract.

### Incremental x86 lockstep saturation on AMD EPYC 9V74

The incremental scheduler was subsequently extended to interleave two and
three stateful native groups, matching the dependency-hiding structure of the
one-shot AVX2 and AVX-512 kernels. The same pass removed a redundant
`buffer_len` field from `Md5State` (the buffered byte count is always
`bytes_hashed & 63`) and changed x86 state materialization from scalar
per-word gather/scatter to one 128-bit `[a, b, c, d]` load/store per stream
plus register transposes. `Md5State` is therefore 88 bytes while retaining one
64-byte partial block.

To separate MD5 compressor throughput from unavoidable public-API work, the
following measurements compare a steady block-aligned `update_many` call with
the same stateful compressor called directly. Values are medians from three
same-host runs; percentages are `update_many / direct compressor`. This is the
appropriate denominator for the remaining per-call scheduler and byte-count
bookkeeping overhead.

| ISA | Streams | 64 B/call | 256 B/call | 4 KiB/call |
| --- | ---: | ---: | ---: | ---: |
| SSE2 | 4 | 99.7% | 100.2% | 100.1% |
| SSE2 | 8 | 98.9% | 100.1% | 99.5% |
| SSE2 | 12 | 100.9% | 100.3% | 100.0% |
| AVX2 | 8 | 98.1% | 99.5% | 100.0% |
| AVX2 | 16 | 95.6% | 98.6% | 99.6% |
| AVX2 | 24 | 96.7% | 99.0% | 99.7% |
| AVX-512 | 16 | 94.0% | 99.4% | 100.0% |
| AVX-512 | 32 | 94.6% | 97.4% | 100.0% |
| AVX-512 | 48 | 95.0% | 99.3% | 99.6% |

Thus every measured 256-byte-and-larger lockstep path is at least 97.4% of
its direct stateful compressor, and 4 KiB calls are effectively saturated.
The worst 64-byte cases are the AVX-512 native/dual paths at roughly 94–95%,
within about one percentage point of the target. The 48-stream AVX-512 path
also improved from roughly 6.8 GiB/s to roughly 9.1 GiB/s for a representative
64-byte steady-call run when its older hardware gather/scatter state packing
was replaced by the 128-bit load/store transpose.

The remaining single-block gap is not another MD5-round scheduling problem:
every call must validate the safe slice shape, advance each stream's logical
byte count, enter the target-feature kernel, and make the externally visible
AoS state persistent again before returning. An AVX-512 hardware
gather/scatter experiment for the byte counters made the 64-byte native/dual
paths substantially worse (roughly 83–86% of the direct compressor) and was
rejected. A larger per-stream staging buffer was also re-measured and rejected
below. Materially reducing this last cost would require a different persistent
batch-state layout/trusted lockstep API, not another transparent optimization
of the current `Md5State` API.

**Implementation consequence:** preserve the stateful AVX2 8/16/24-way and
AVX-512 16/32/48-way kernels, the validated lockstep fast path, and the
128-bit AoS-to-SoA state transposes. The
`x86-incremental-lockstep-*` Criterion groups cover forced SSE2/AVX2/AVX-512
native, dual, and triple groups at 64 B and 4 KiB per call.

## Low-occupancy AMD family 19h scheduling

Sparse SIMD is not always the best way to hash two or three independent
messages. Family 19h measurements motivated a dual-GPR backend that interleaves
two scalar dependency chains and uses BMI1 `ANDN` in throughput-bound G/I
rounds.

### Two-message overlap

The dual-scalar path helps when both streams have useful work to overlap. With
a 64 KiB longer message, representative gains grew with the shorter message:

| Shorter message | approximate improvement |
| ---: | ---: |
| 4 KiB | ~5.7% |
| 8 KiB | ~10.1% |
| 16 KiB | ~18.3% |
| 32 KiB | ~29.9% |

Extremely skewed pairs with only one or two padded blocks on the short side
showed little or no benefit. The scheduler therefore always allows small pairs
(up to 32 padded blocks on the longer side) but requires at least a 1:16 overlap
ratio for larger pairs.

Adding BMI1 to the already-interleaved dual kernel produced another roughly
1.6–2.6% on medium/long equal pairs, with a larger ~6.9% result at 64 B in the
measured host. This differs from the single-stream BMI1 experiment, where fewer
instructions did not materially shorten the dependency-critical path.

### Three-message skew

For sorted padded workloads `a <= b <= c`, sparse AVX2 remained preferable for
equal and near-equal triples. Pairing the two longest messages in the dual-GPR
backend became consistently useful once:

```text
b * 4 <= c
```

Representative measured improvements were about 9–31%, including roughly
15.5% for `64 + 512 + 512`, 17.7% for `512 + 4 KiB + 4 KiB`, and 22.6% for
`4 KiB + 64 KiB + 64 KiB`. Around `b ~= c/2` the split began to lose, which is
why the quarter-gap condition is deliberately conservative.

### Four-to-eight-message skew

Blindly applying the general skew partitioner to every small batch regressed
one-short/many-long shapes by roughly 17–25%. The useful pattern was a split
whose long partition shrank to at most two messages; that lets recursion finish
in scalar/dual-scalar work instead of leaving a sparse vector tail.

Representative wins included roughly 25% for `64,64,4 KiB,4 KiB`, about 8% for
`1 KiB,1 KiB,4 KiB,4 KiB`, and about 24–28% for extreme 5–8-message clustered
short/long tails.

**Scheduler consequence:** preserve the current `long_count <= 2` restriction
for the small-tail partitioner. Do not generalize from "there is a large gap"
to "always split".

## AArch64 observations

### GitHub-hosted Neoverse-N2

GitHub's AArch64 performance jobs have repeatedly run on a reported
Neoverse-N2 host. In same-SHA / baseline-vs-head calibration runs the ABBA
paired measurements were unusually stable, commonly within a few tenths of a
percent. A representative run measured approximately:

| Benchmark | Throughput |
| --- | ---: |
| `md5-many` one-shot single stream | ~677.5 MiB/s |
| `md5-many` RustCrypto streaming | ~676.8 MiB/s |
| RustCrypto `md-5` reference | ~696.0 MiB/s |
| 4-way 64 KiB multi-buffer | ~963 MiB/s |

A same-binary backend probe on 2026-08-20 resolved the single-stream question.
On the same reported Neoverse-N2 host, the portable Rust compressor beat the
hand-scheduled AArch64 GPR compressor at every measured size: 173.52 vs
177.89 ns at 64 B, 1.4933 vs 1.5274 us at 1 KiB, 90.238 vs 92.154 us at
64 KiB, and about 693.2 vs 678.1 MiB/s at 1 MiB. The 0-55 B short probes
showed the same direction. Production AArch64 single-stream dispatch therefore
uses the portable compressor; the GPR path remains available only as a backend
comparison hook.

The same probe established a native equal-length NEON crossover. At 4 lanes,
the native kernel measured 730.9 MiB/s vs 346.6 MiB/s for Fearless SIMD at
55 B, 951.3 vs 833.4 MiB/s at 1 KiB, and 1018.6 vs 965.8 MiB/s at 64 KiB.
Round-interleaving independent four-lane chains was substantially stronger:
8-way reached 1.676 GiB/s at 1 KiB and 1.837 GiB/s at 64 KiB; 12-way reached
2.041 GiB/s and 2.206 GiB/s respectively. Serial native-NEON4 grouping stayed
near 952 MiB/s at 1 KiB and 1.016 GiB/s at 64 KiB, confirming that the extra
8/12-way gain is genuine inter-chain ILP rather than only a better transpose.

Production AArch64 batch dispatch consequently uses native NEON for measured
equal-length shapes. It consumes 12-way groups preferentially, except a
16-message region is scheduled as 8+8 to avoid a final 4-way group; remaining
one to three lanes use the existing generic path unless a measured padded
under-fill rule applies.

A follow-up same-binary N2 probe measured lane duplication for 5–15 equal
messages at 55, 56, 64, 119, 120, 128, 256, 1 KiB, and 64 KiB. Padding 6/7,
10/11, and 15 lanes to the next measured native width won at every sampled
point from 55 B upward, ranging from roughly 7% to 59% lower latency. Five
lanes lost about 12% at 55 B but won from the 56-byte padding boundary and by
about 21% on aligned/long inputs. Nine lanes lost about 8% at 55 B and about
1% at 119 B, while aligned inputs and long messages won by roughly 7–11%.
Padding 13/14 lanes to 16 was rejected: 13 lanes regressed about 26–32% and
14 lanes about 3.5–15% across the measured matrix. Production therefore keeps
explicit conservative crossover guards rather than padding every under-filled
group.

The same run supplied direct hardware evidence for native mixed scheduling.
For 4/8/12-message batches whose byte lengths differ but whose messages require
the same total number of padded MD5 blocks, the native path reduced latency by
about 38–55%, 63–67%, and 66–72% respectively versus the then-current
production scheduler across short, ~1 KiB, and ~64 KiB probes. The production
mixed scheduler now replaces consecutive 4/8/12-message generic chunks with
that native kernel when the padded-block-count condition holds, while preserving
the previous four-lane generic chunk boundaries for rejected heterogeneous
prefixes.

The preceding duplicate whole-batch/four-lane mixed-length scan was also
removed. The blocking `mixed-short` ABBA sentinel moved from a prior ~2.3%
slowdown to about **1.0% faster** than `0.1.0-alpha.4`, confirming that the
dispatch fixed-cost regression was eliminated before enabling the native mixed
path.

The hosted runner model is also not a platform contract; future GitHub runners
may expose different ARM CPUs.

## Short one-shot framing

The public one-shot path now specializes messages of at most 55 bytes as a
single padded block. This is deliberately a framing specialization, not a
second MD5 round implementation: it avoids constructing the generic two-block
finalization buffer and calls the existing selected compressor once. The
Intel-preferred AVX-512VL one-shot path has the same compact one-block framing
so its vector state still remains resident for the whole hash.

A same-binary `backend-short-framing` comparison on the AMD EPYC 9V74 runner
showed roughly 4-10% lower latency at representative 0-55-byte points. Forced
AVX-512VL measurements on the same host also improved by roughly 2-7% across
most of that range, despite AVX-512VL itself remaining the wrong single-stream
choice on AMD Family 19h. The benefit disappeared around the two-padding-block
boundary, so 56-byte and larger messages remain on the generic framing path. A
separate compact-final-block experiment for arbitrary 64-byte-aligned inputs
was inconsistent and is not part of production dispatch.

### x86 one-shot dispatch hoisting

The x86 one-shot entry point now resolves the AVX-512VL-vs-NoLEA choice once
per hash. Previously, a non-Intel/NoLEA hash resolved that choice at entry and
then re-read the cached preference from every `compress_block` call. The direct
NoLEA one-shot path keeps the same compressor and padding rules but removes
that per-block dispatch load.

On the AMD EPYC 9V74 host, Criterion's same-benchmark before/after comparison
reported approximately 3.3% lower latency at 64 B, 5.1% at 1 KiB, 3.7% at
64 KiB, and 3.0% at 1 MiB. The forced-NoLEA control in the same run showed no
significant movement, which isolates the gain to dispatch overhead rather than
a compressor/code-frequency change.

### AMD AVX-512 single-stream control

The current local VM exposes five vCPUs of an AMD EPYC 9V74 and includes
AVX-512F/DQ/BW/VL. A CPU-pinned, alternating same-process control reconfirmed
that AVX-512VL remains the wrong dependency-chain backend on this AMD host:
forced AVX-512 was about 29-30% slower than NoLEA at 1-64 B and about 35-36%
slower at 1 KiB through 1 MiB. This is stronger than comparing separate
Criterion runs because each pair alternated execution order inside one process.
It reinforces the existing AMD NoLEA dispatch, but it is not evidence for or
against the Intel family-6 preference.

### Backend candidates awaiting target hardware

Two candidates deliberately remain behind `bench-internals` rather than
production dispatch:

- AVX-512VL digest packing replaces four low-dword scalar extracts with two XMM
  unpacks and one 16-byte store. Rust 1.97.1 / LLVM 22 already auto-packs the
  generic (`>=56 B`) scalar-extract epilogue into a vector shuffle plus one
  16-byte store, so the handwritten candidate only changes the specialized
  `<=55 B` path in a material way. On the current AMD EPYC 9V74 VM, an
  alternating same-process A/B stayed within roughly -1% to +2% for the short
  path and within about +/-1% at 64 B through 1 MiB, so there is no AMD basis
  for productionizing it. Keep both epilogues available only for a future
  Intel same-binary A/B, where instruction latency may differ.
- AArch64 backend probes retain forced portable/GPR and forced Fearless/native
  NEON controls even though the measured Neoverse-N2 results have now moved
  portable single-stream, equal-length native NEON, selected padded under-fill
  shapes, and same-padded-block-count mixed chunks into production. These
  probes remain useful for detecting a future runner/microarchitecture where
  the crossover differs.

## Experiments rejected so far

Keeping rejected experiments documented avoids repeating attractive but
unproductive rewrites without new evidence.

- Single-stream `<56 B` specialized compressors removed known-zero message-word
  additions but produced only about 0.3–0.6% end-to-end improvement in tiny
  cases and essentially no broader gain.
- Replacing the short one-shot `copy_from_slice` with hand-written
  1/2/4/8/16-byte unaligned copy chunks did not improve 31/47-byte cases
  consistently and regressed the 55-byte boundary, so the standard slice copy
  remains.
- Scalar BMI1 G/I rewrites did not materially improve the single-stream
  dependency-critical path on the measured family 19h host. BMI1 is retained
  where it helped the throughput-bound dual-GPR kernel instead.
- H-round reuse conflicts with the current NoLEA schedule because the register
  that upstream reuses already contains a prefetched next message word; saving
  the old state would consume the instruction the reuse was meant to remove.
- A many-buffer `<=31 B` half-block transpose experiment showed about 2–6%
  gains for 15–31 B inputs but regressions for 1–14 B and code-layout-sensitive
  movement in unrelated cases. It was rejected rather than duplicating a large
  64-round kernel for a fragile small win.
- Incremental per-stream staging larger than one 64-byte block was rechecked
  after the AVX-512 stateful kernel and lockstep scheduler landed. A 256-byte
  buffer could improve the 16-byte-chunk case by roughly 10% on the measured
  EPYC 9V74, but 32/64-byte and 4 KiB updates showed no stable corresponding
  gain. The state used by that experiment was roughly 96 bytes; after removing
  the redundant `buffer_len`, the retained one-block state is 88 bytes, while
  256-byte staging would still add roughly another 192 bytes per stream.
  128-byte and 512-byte staging did not establish a better crossover either.
  The memory/API trade-off remains rejected; the public state keeps a single
  64-byte partial block.

## CI performance guard

The normal CI workflow runs architecture correctness, MSRV, and quality/package
checks before performance sentinels. Performance comparisons build baseline and
candidate separately on the same runner and pin both to the same schedulable
CPU. ABBA ordering reduces frequency/order drift; hardware-specific benchmark
filters may be explicitly optional when their ISA is unavailable, while normal
filters remain strict.

For implementation details and threshold invariants, see `AGENTS.md`. For the
commands and user-facing benchmark inventory, see `README.md`.
