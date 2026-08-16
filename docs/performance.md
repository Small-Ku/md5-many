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

The reference comparison is an **investigation signal, not a dispatch result**.
It compares different implementations and does not isolate the hand-scheduled
AArch64 compressor from `md5-many`'s portable scalar fallback. A direct forced
portable-vs-hand-tuned AArch64 A/B is still required before changing the
single-stream backend.

The hosted runner model is also not a platform contract; future GitHub runners
may expose different ARM CPUs.

## Experiments rejected so far

Keeping rejected experiments documented avoids repeating attractive but
unproductive rewrites without new evidence.

- Single-stream `<56 B` specialized compressors removed known-zero message-word
  additions but produced only about 0.3–0.6% end-to-end improvement in tiny
  cases and essentially no broader gain.
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

## CI performance guard

The normal CI workflow runs architecture correctness, MSRV, and quality/package
checks before performance sentinels. Performance comparisons build baseline and
candidate separately on the same runner and pin both to the same schedulable
CPU. ABBA ordering reduces frequency/order drift; hardware-specific benchmark
filters may be explicitly optional when their ISA is unavailable, while normal
filters remain strict.

For implementation details and threshold invariants, see `AGENTS.md`. For the
commands and user-facing benchmark inventory, see `README.md`.
