use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use md5_many::bench_internals::{md5_aligned, md5_generic, md5_short_one_block};
use std::hint::black_box;
use std::time::Duration;

fn bench_short_framing(c: &mut Criterion) {
    let data = [0x5au8; 119];
    let mut group = c.benchmark_group("backend-short-framing");
    group.sample_size(15);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_millis(250));

    for &len in &[0usize, 1, 7, 15, 31, 47, 55] {
        let input = &data[..len];
        group.throughput(Throughput::Bytes(len as u64));
        group.bench_with_input(BenchmarkId::new("generic", len), &len, |b, _| {
            b.iter(|| black_box(md5_generic(black_box(input))))
        });
        group.bench_with_input(BenchmarkId::new("specialized", len), &len, |b, _| {
            b.iter(|| black_box(md5_short_one_block(black_box(input))))
        });
    }
    group.finish();
}

fn bench_aligned_framing(c: &mut Criterion) {
    let data = vec![0x6bu8; 64 * 1024];
    let mut group = c.benchmark_group("backend-aligned-framing");
    group.sample_size(15);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_millis(250));

    for &len in &[64usize, 128, 256, 512, 1024, 4096, 64 * 1024] {
        let input = &data[..len];
        group.throughput(Throughput::Bytes(len as u64));
        group.bench_with_input(BenchmarkId::new("generic", len), &len, |b, _| {
            b.iter(|| black_box(md5_generic(black_box(input))))
        });
        group.bench_with_input(BenchmarkId::new("aligned", len), &len, |b, _| {
            b.iter(|| black_box(md5_aligned(black_box(input))))
        });
    }
    group.finish();
}

#[cfg(target_arch = "x86_64")]
fn bench_x86_avx512_short(c: &mut Criterion) {
    use md5_many::bench_internals::{md5_x86_avx512, md5_x86_avx512_generic, x86_avx512_supported};

    if !x86_avx512_supported() {
        return;
    }

    let data = [0x3cu8; 55];
    let mut group = c.benchmark_group("backend-x86-avx512-short");
    group.sample_size(15);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_millis(250));
    for &len in &[0usize, 1, 15, 31, 47, 55] {
        let input = &data[..len];
        group.bench_with_input(BenchmarkId::new("generic", len), &len, |b, _| {
            b.iter(|| black_box(md5_x86_avx512_generic(black_box(input))))
        });
        group.bench_with_input(BenchmarkId::new("specialized", len), &len, |b, _| {
            b.iter(|| black_box(md5_x86_avx512(black_box(input))))
        });
    }
    group.finish();
}

#[cfg(not(target_arch = "x86_64"))]
fn bench_x86_avx512_short(_c: &mut Criterion) {}

#[cfg(target_arch = "x86_64")]
fn bench_x86_single_stream(c: &mut Criterion) {
    use md5_many::bench_internals::{md5_x86_avx512, md5_x86_nolea, x86_avx512_supported};

    if !x86_avx512_supported() {
        return;
    }

    let data = vec![0xa5u8; 1024 * 1024];
    for &len in &[64usize, 1024, 64 * 1024, 1024 * 1024] {
        let input = &data[..len];
        let mut group = c.benchmark_group(format!("backend-x86-single-stream-{len}"));
        group.sample_size(20);
        group.warm_up_time(Duration::from_millis(250));
        group.measurement_time(Duration::from_millis(750));
        group.throughput(Throughput::Bytes(len as u64));
        group.bench_function("nolea", |b| {
            b.iter(|| black_box(md5_x86_nolea(black_box(input))))
        });
        group.bench_function("avx512vl", |b| {
            b.iter(|| black_box(md5_x86_avx512(black_box(input))))
        });
        group.finish();
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn bench_x86_single_stream(_c: &mut Criterion) {}

#[cfg(target_arch = "x86_64")]
fn bench_x86_avx512_digest_store(c: &mut Criterion) {
    use md5_many::bench_internals::{
        md5_x86_avx512, md5_x86_avx512_packed_digest, x86_avx512_supported,
    };

    if !x86_avx512_supported() {
        return;
    }

    let data = vec![0x42u8; 4096];
    let mut group = c.benchmark_group("backend-x86-avx512-digest-store");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(150));
    group.measurement_time(Duration::from_millis(400));
    for &len in &[1usize, 15, 31, 55, 64, 1024, 4096] {
        let input = &data[..len];
        group.throughput(Throughput::Bytes(len as u64));
        group.bench_with_input(BenchmarkId::new("scalar-extract", len), &len, |b, _| {
            b.iter(|| black_box(md5_x86_avx512(black_box(input))))
        });
        group.bench_with_input(BenchmarkId::new("packed-store", len), &len, |b, _| {
            b.iter(|| black_box(md5_x86_avx512_packed_digest(black_box(input))))
        });
    }
    group.finish();
}

#[cfg(not(target_arch = "x86_64"))]
fn bench_x86_avx512_digest_store(_c: &mut Criterion) {}

#[cfg(target_arch = "x86_64")]
fn bench_x86_dispatch_once(c: &mut Criterion) {
    use md5_many::bench_internals::md5_x86_nolea;

    let data = vec![0x73u8; 1024 * 1024];
    let mut group = c.benchmark_group("backend-x86-dispatch-once");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(150));
    group.measurement_time(Duration::from_millis(500));
    for &len in &[64usize, 1024, 64 * 1024, 1024 * 1024] {
        let input = &data[..len];
        group.throughput(Throughput::Bytes(len as u64));
        group.bench_with_input(BenchmarkId::new("public", len), &len, |b, _| {
            b.iter(|| black_box(md5_many::md5(black_box(input))))
        });
        group.bench_with_input(BenchmarkId::new("forced-nolea", len), &len, |b, _| {
            b.iter(|| black_box(md5_x86_nolea(black_box(input))))
        });
    }
    group.finish();
}

#[cfg(not(target_arch = "x86_64"))]
fn bench_x86_dispatch_once(_c: &mut Criterion) {}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
fn bench_aarch64_single_stream(c: &mut Criterion) {
    use md5_many::bench_internals::{md5_aarch64_gpr, md5_portable};

    let data = vec![0x96u8; 1024 * 1024];
    for &len in &[64usize, 1024, 64 * 1024, 1024 * 1024] {
        let input = &data[..len];
        let mut group = c.benchmark_group(format!("backend-aarch64-single-stream-{len}"));
        group.sample_size(20);
        group.warm_up_time(Duration::from_millis(250));
        group.measurement_time(Duration::from_millis(750));
        group.throughput(Throughput::Bytes(len as u64));
        group.bench_function("portable", |b| {
            b.iter(|| black_box(md5_portable(black_box(input))))
        });
        group.bench_function("gpr", |b| {
            b.iter(|| black_box(md5_aarch64_gpr(black_box(input))))
        });
        group.finish();
    }
}

#[cfg(not(all(target_arch = "aarch64", target_endian = "little")))]
fn bench_aarch64_single_stream(_c: &mut Criterion) {}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
fn bench_aarch64_short(c: &mut Criterion) {
    use md5_many::bench_internals::{md5_aarch64_gpr_short, md5_portable_short};

    let data = [0x5du8; 55];
    let mut group = c.benchmark_group("backend-aarch64-short");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(150));
    group.measurement_time(Duration::from_millis(500));
    for &len in &[0usize, 1, 15, 31, 47, 55] {
        let input = &data[..len];
        group.bench_with_input(BenchmarkId::new("portable", len), &len, |b, _| {
            b.iter(|| black_box(md5_portable_short(black_box(input))))
        });
        group.bench_with_input(BenchmarkId::new("gpr", len), &len, |b, _| {
            b.iter(|| black_box(md5_aarch64_gpr_short(black_box(input))))
        });
    }
    group.finish();
}

#[cfg(not(all(target_arch = "aarch64", target_endian = "little")))]
fn bench_aarch64_short(_c: &mut Criterion) {}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
fn bench_aarch64_neon4(c: &mut Criterion) {
    use md5_many::bench_internals::{md5_aarch64_neon4, md5_many_aarch64_fearless_neon};

    let engine = md5_many::Md5Many::new();
    if engine.lanes() != 4 {
        return;
    }

    for &len in &[0usize, 55, 64, 1024, 64 * 1024, 1024 * 1024] {
        let data: [Vec<u8>; 4] = core::array::from_fn(|lane| {
            (0..len)
                .map(|index| (index as u8).wrapping_add((lane as u8).wrapping_mul(41)))
                .collect()
        });
        let inputs = data.each_ref().map(|input| input.as_slice());
        let mut generic_outputs = [[0u8; 16]; 4];
        let mut production_outputs = [[0u8; 16]; 4];
        let mut group = c.benchmark_group(format!("backend-aarch64-neon4-{len}"));
        group.sample_size(20);
        group.warm_up_time(Duration::from_millis(250));
        group.measurement_time(Duration::from_millis(750));
        group.throughput(Throughput::Bytes((len * 4) as u64));
        group.bench_function("fearless-neon", |b| {
            b.iter(|| {
                md5_many_aarch64_fearless_neon(black_box(&inputs), black_box(&mut generic_outputs));
                black_box(generic_outputs)
            })
        });
        group.bench_function("production", |b| {
            b.iter(|| {
                engine.hash_many(black_box(&inputs), black_box(&mut production_outputs));
                black_box(production_outputs)
            })
        });
        group.bench_function("native-neon4", |b| {
            b.iter(|| black_box(md5_aarch64_neon4(black_box(inputs))))
        });
        group.finish();
    }
}

#[cfg(not(all(target_arch = "aarch64", target_endian = "little")))]
fn bench_aarch64_neon4(_c: &mut Criterion) {}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
fn bench_aarch64_neon_occupancy(c: &mut Criterion) {
    use md5_many::bench_internals::{
        md5_aarch64_neon4, md5_aarch64_neon8, md5_aarch64_neon12, md5_many_aarch64_fearless_neon,
    };

    let engine = md5_many::Md5Many::new();
    for &lanes in &[4usize, 8, 12] {
        for &len in &[55usize, 64, 1024, 64 * 1024] {
            let data: Vec<Vec<u8>> = (0..lanes)
                .map(|lane| {
                    (0..len)
                        .map(|index| (index as u8).wrapping_add((lane as u8).wrapping_mul(37)))
                        .collect()
                })
                .collect();
            let inputs: Vec<&[u8]> = data.iter().map(Vec::as_slice).collect();
            let mut generic_outputs = vec![[0u8; 16]; lanes];
            let mut production_outputs = vec![[0u8; 16]; lanes];
            let mut native_outputs = vec![[0u8; 16]; lanes];
            let mut group =
                c.benchmark_group(format!("backend-aarch64-neon-occupancy-{lanes}x{len}"));
            group.sample_size(20);
            group.warm_up_time(Duration::from_millis(200));
            group.measurement_time(Duration::from_millis(600));
            group.throughput(Throughput::Bytes((lanes * len) as u64));
            group.bench_function("fearless-neon", |b| {
                b.iter(|| {
                    md5_many_aarch64_fearless_neon(
                        black_box(&inputs),
                        black_box(&mut generic_outputs),
                    );
                    black_box(generic_outputs[0])
                })
            });
            group.bench_function("production", |b| {
                b.iter(|| {
                    engine.hash_many(black_box(&inputs), black_box(&mut production_outputs));
                    black_box(production_outputs[0])
                })
            });
            group.bench_function("native-neon4-groups", |b| {
                b.iter(|| {
                    for (group_index, chunk) in inputs.chunks_exact(4).enumerate() {
                        let four: [&[u8]; 4] = chunk.try_into().expect("four equal-length inputs");
                        let got = md5_aarch64_neon4(black_box(four));
                        native_outputs[group_index * 4..group_index * 4 + 4].copy_from_slice(&got);
                    }
                    black_box(native_outputs[0])
                })
            });
            match lanes {
                8 => {
                    let eight: [&[u8]; 8] = inputs.as_slice().try_into().expect("eight inputs");
                    group.bench_function("native-neon8-interleaved", |b| {
                        b.iter(|| black_box(md5_aarch64_neon8(black_box(eight))))
                    });
                }
                12 => {
                    let twelve: [&[u8]; 12] = inputs.as_slice().try_into().expect("twelve inputs");
                    group.bench_function("native-neon12-interleaved", |b| {
                        b.iter(|| black_box(md5_aarch64_neon12(black_box(twelve))))
                    });
                }
                _ => {}
            }
            group.finish();
        }
    }
}

#[cfg(not(all(target_arch = "aarch64", target_endian = "little")))]
fn bench_aarch64_neon_occupancy(_c: &mut Criterion) {}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
fn bench_aarch64_neon_scheduler(c: &mut Criterion) {
    use md5_many::bench_internals::md5_many_aarch64_fearless_neon;

    let engine = md5_many::Md5Many::new();
    for &lanes in &[5usize, 16, 20, 28] {
        for &len in &[55usize, 56, 64, 119, 120, 128, 256, 1024, 64 * 1024] {
            let data: Vec<Vec<u8>> = (0..lanes)
                .map(|lane| {
                    (0..len)
                        .map(|index| (index as u8).wrapping_add((lane as u8).wrapping_mul(29)))
                        .collect()
                })
                .collect();
            let inputs: Vec<&[u8]> = data.iter().map(Vec::as_slice).collect();
            let mut generic_outputs = vec![[0u8; 16]; lanes];
            let mut production_outputs = vec![[0u8; 16]; lanes];
            let mut group =
                c.benchmark_group(format!("backend-aarch64-neon-scheduler-{lanes}x{len}"));
            group.sample_size(20);
            group.warm_up_time(Duration::from_millis(200));
            group.measurement_time(Duration::from_millis(600));
            group.throughput(Throughput::Bytes((lanes * len) as u64));
            group.bench_function("fearless-neon", |b| {
                b.iter(|| {
                    md5_many_aarch64_fearless_neon(
                        black_box(&inputs),
                        black_box(&mut generic_outputs),
                    );
                    black_box(generic_outputs[0])
                })
            });
            group.bench_function("production", |b| {
                b.iter(|| {
                    engine.hash_many(black_box(&inputs), black_box(&mut production_outputs));
                    black_box(production_outputs[0])
                })
            });
            group.finish();
        }
    }
}

#[cfg(not(all(target_arch = "aarch64", target_endian = "little")))]
fn bench_aarch64_neon_scheduler(_c: &mut Criterion) {}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
fn bench_aarch64_neon_partial_candidates(c: &mut Criterion) {
    use md5_many::bench_internals::md5_aarch64_neon_padded_equal;

    let engine = md5_many::Md5Many::new();
    for &lanes in &[5usize, 6, 7, 9, 10, 11, 13, 14, 15] {
        for &len in &[55usize, 56, 64, 119, 120, 128, 256, 1024, 64 * 1024] {
            let data: Vec<Vec<u8>> = (0..lanes)
                .map(|lane| vec![(lane as u8).wrapping_mul(47); len])
                .collect();
            let inputs: Vec<&[u8]> = data.iter().map(Vec::as_slice).collect();
            let mut production_outputs = vec![[0u8; 16]; lanes];
            let mut padded_outputs = vec![[0u8; 16]; lanes];
            let mut group =
                c.benchmark_group(format!("backend-aarch64-neon-partial-{lanes}x{len}"));
            group.sample_size(20);
            group.warm_up_time(Duration::from_millis(200));
            group.measurement_time(Duration::from_millis(600));
            group.throughput(Throughput::Bytes((lanes * len) as u64));
            group.bench_function("production", |b| {
                b.iter(|| {
                    engine.hash_many(black_box(&inputs), black_box(&mut production_outputs));
                    black_box(production_outputs[0])
                })
            });
            group.bench_function("padded-native", |b| {
                b.iter(|| {
                    assert!(md5_aarch64_neon_padded_equal(
                        black_box(&inputs),
                        black_box(&mut padded_outputs),
                    ));
                    black_box(padded_outputs[0])
                })
            });
            group.finish();
        }
    }
}

#[cfg(not(all(target_arch = "aarch64", target_endian = "little")))]
fn bench_aarch64_neon_partial_candidates(_c: &mut Criterion) {}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
fn bench_aarch64_neon_mixed_candidates(c: &mut Criterion) {
    use md5_many::bench_internals::{
        md5_aarch64_neon_mixed_same_blocks, md5_many_aarch64_fearless_neon,
    };

    let engine = md5_many::Md5Many::new();
    let patterns: &[&[usize]] = &[
        &[0, 1, 7, 15],
        &[0, 1, 7, 15, 31, 47, 54, 55],
        &[0, 1, 2, 3, 7, 15, 23, 31, 39, 47, 54, 55],
        &[56, 57, 63, 64],
        &[56, 57, 63, 64, 65, 79, 96, 119],
        &[56, 57, 58, 63, 64, 65, 72, 80, 96, 104, 112, 119],
        &[1016, 1017, 1023, 1024],
        &[1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023],
        &[
            1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027,
        ],
        &[65_528, 65_529, 65_530, 65_531],
        &[
            65_528, 65_529, 65_530, 65_531, 65_532, 65_533, 65_534, 65_535,
        ],
        &[
            65_528, 65_529, 65_530, 65_531, 65_532, 65_533, 65_534, 65_535, 65_536, 65_537, 65_538,
            65_539,
        ],
    ];

    for &lengths in patterns {
        let lanes = lengths.len();
        let data: Vec<Vec<u8>> = lengths
            .iter()
            .enumerate()
            .map(|(lane, &len)| vec![(lane as u8).wrapping_mul(53); len])
            .collect();
        let inputs: Vec<&[u8]> = data.iter().map(Vec::as_slice).collect();
        let total: usize = lengths.iter().sum();
        let mut fearless_outputs = vec![[0u8; 16]; lanes];
        let mut production_outputs = vec![[0u8; 16]; lanes];
        let mut native_outputs = vec![[0u8; 16]; lanes];
        let mut group = c.benchmark_group(format!(
            "backend-aarch64-neon-mixed-same-blocks-{lanes}x{}",
            lengths.iter().copied().max().unwrap_or(0)
        ));
        group.sample_size(20);
        group.warm_up_time(Duration::from_millis(200));
        group.measurement_time(Duration::from_millis(600));
        group.throughput(Throughput::Bytes(total as u64));
        group.bench_function("fearless-neon", |b| {
            b.iter(|| {
                md5_many_aarch64_fearless_neon(
                    black_box(&inputs),
                    black_box(&mut fearless_outputs),
                );
                black_box(fearless_outputs[0])
            })
        });
        group.bench_function("production", |b| {
            b.iter(|| {
                engine.hash_many(black_box(&inputs), black_box(&mut production_outputs));
                black_box(production_outputs[0])
            })
        });
        group.bench_function("native-same-blocks", |b| {
            b.iter(|| {
                assert!(md5_aarch64_neon_mixed_same_blocks(
                    black_box(&inputs),
                    black_box(&mut native_outputs),
                ));
                black_box(native_outputs[0])
            })
        });
        group.finish();
    }
}

#[cfg(not(all(target_arch = "aarch64", target_endian = "little")))]
fn bench_aarch64_neon_mixed_candidates(_c: &mut Criterion) {}

criterion_group!(
    benches,
    bench_short_framing,
    bench_aligned_framing,
    bench_x86_avx512_short,
    bench_x86_single_stream,
    bench_x86_avx512_digest_store,
    bench_x86_dispatch_once,
    bench_aarch64_single_stream,
    bench_aarch64_short,
    bench_aarch64_neon4,
    bench_aarch64_neon_occupancy,
    bench_aarch64_neon_scheduler,
    bench_aarch64_neon_partial_candidates,
    bench_aarch64_neon_mixed_candidates
);
criterion_main!(benches);
