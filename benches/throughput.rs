use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fearless_md5::{Md5Many, md5};
use md5::{Digest as _, Md5 as RustCryptoMd5};
use std::hint::black_box;

fn bench_single(c: &mut Criterion) {
    let data = vec![0x5au8; 1024 * 1024];
    let mut group = c.benchmark_group("single-stream-1MiB");
    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_function("md5-many-md5", |b| {
        b.iter(|| black_box(md5(black_box(&data))))
    });
    group.bench_function("rustcrypto-baseline", |b| {
        b.iter(|| black_box(RustCryptoMd5::digest(black_box(&data))))
    });
    group.bench_function("rustcrypto-md5", |b| {
        b.iter(|| black_box(RustCryptoMd5::digest(black_box(&data))))
    });
    group.finish();
}

fn bench_many(c: &mut Criterion) {
    let engine = Md5Many::new();
    let lanes = engine.lanes();

    for &size in &[64usize, 1024, 64 * 1024, 1024 * 1024] {
        let storage: Vec<Vec<u8>> = (0..lanes)
            .map(|lane| vec![(lane as u8).wrapping_mul(17); size])
            .collect();
        let inputs: Vec<&[u8]> = storage.iter().map(Vec::as_slice).collect();
        let mut outputs = vec![[0u8; 16]; lanes];

        let mut group = c.benchmark_group(format!("many-{size}-bytes"));
        group.throughput(Throughput::Bytes((size * lanes) as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("md5-many-{lanes}-way"), size),
            &size,
            |b, _| {
                b.iter(|| {
                    engine.hash_many(black_box(&inputs), black_box(&mut outputs));
                    black_box(&outputs);
                })
            },
        );
        group.bench_with_input(BenchmarkId::new("rustcrypto-baseline-serial", size), &size, |b, _| {
            b.iter(|| {
                for input in &inputs {
                    black_box(RustCryptoMd5::digest(black_box(input)));
                }
            })
        });
        group.bench_with_input(
            BenchmarkId::new("rustcrypto-serial", size),
            &size,
            |b, _| {
                b.iter(|| {
                    for input in &inputs {
                        black_box(RustCryptoMd5::digest(black_box(input)));
                    }
                })
            },
        );
        group.finish();
    }
}

fn bench_lane_fill(c: &mut Criterion) {
    let engine = Md5Many::new();
    let max_lanes = engine.lanes();
    let size = 64 * 1024usize;
    let storage: Vec<Vec<u8>> = (0..max_lanes)
        .map(|lane| vec![(lane as u8).wrapping_mul(29); size])
        .collect();

    let mut group = c.benchmark_group("lane-fill-64KiB");
    for active in 1..=max_lanes {
        let inputs: Vec<&[u8]> = storage[..active].iter().map(Vec::as_slice).collect();
        let mut outputs = vec![[0u8; 16]; active];
        group.throughput(Throughput::Bytes((size * active) as u64));
        group.bench_with_input(BenchmarkId::new("fearless", active), &active, |b, _| {
            b.iter(|| {
                engine.hash_many(black_box(&inputs), black_box(&mut outputs));
                black_box(&outputs);
            })
        });
        group.bench_with_input(
            BenchmarkId::new("rustcrypto-baseline-serial", active),
            &active,
            |b, _| {
                b.iter(|| {
                    for input in &inputs {
                        black_box(RustCryptoMd5::digest(black_box(input)));
                    }
                })
            },
        );
    }
    group.finish();
}

fn bench_mixed_short(c: &mut Criterion) {
    let engine = Md5Many::new();
    let lanes = engine.lanes();
    let boundary_lengths = [
        0usize, 1, 7, 15, 31, 47, 55, 56, 57, 63, 64, 65, 119, 120, 127, 128,
    ];
    let storage: Vec<Vec<u8>> = (0..lanes)
        .map(|lane| vec![lane as u8; boundary_lengths[lane % boundary_lengths.len()]])
        .collect();
    let inputs: Vec<&[u8]> = storage.iter().map(Vec::as_slice).collect();
    let total: usize = inputs.iter().map(|input| input.len()).sum();
    let mut outputs = vec![[0u8; 16]; lanes];

    let mut group = c.benchmark_group("mixed-short");
    group.throughput(Throughput::Bytes(total as u64));
    group.bench_function(format!("md5-many-{lanes}-way"), |b| {
        b.iter(|| {
            engine.hash_many(black_box(&inputs), black_box(&mut outputs));
            black_box(&outputs);
        })
    });
    group.bench_function("rustcrypto-baseline-serial", |b| {
        b.iter(|| {
            for input in &inputs {
                black_box(RustCryptoMd5::digest(black_box(input)));
            }
        })
    });
    group.finish();
}

fn bench_mixed_lengths(c: &mut Criterion) {
    let engine = Md5Many::new();
    let lanes = engine.lanes();

    for batches in 1..=3 {
        let count = lanes * batches;
        let storage: Vec<Vec<u8>> = (0..count)
            .map(|lane| vec![lane as u8; 64 * 1024 - (lane % lanes) * 64])
            .collect();
        let inputs: Vec<&[u8]> = storage.iter().map(Vec::as_slice).collect();
        let total: usize = inputs.iter().map(|input| input.len()).sum();
        let mut outputs = vec![[0u8; 16]; count];
        let mut group = c.benchmark_group(format!("mixed-{count}x~64KiB"));
        group.throughput(Throughput::Bytes(total as u64));
        group.bench_function("fearless", |b| {
            b.iter(|| {
                engine.hash_many(black_box(&inputs), black_box(&mut outputs));
                black_box(&outputs);
            })
        });
        group.bench_function("rustcrypto-baseline-serial", |b| {
            b.iter(|| {
                for input in &inputs {
                    black_box(RustCryptoMd5::digest(black_box(input)));
                }
            })
        });
        group.finish();
    }
}

fn bench_batch_scaling(c: &mut Criterion) {
    let engine = Md5Many::new();
    let lanes = engine.lanes();
    let size = 64 * 1024usize;

    for batches in 1..=8 {
        let count = lanes * batches;
        let storage: Vec<Vec<u8>> = (0..count)
            .map(|lane| vec![(lane as u8).wrapping_mul(11); size])
            .collect();
        let inputs: Vec<&[u8]> = storage.iter().map(Vec::as_slice).collect();
        let mut outputs = vec![[0u8; 16]; count];
        let mut group = c.benchmark_group(format!("batch-scaling-{count}"));
        group.throughput(Throughput::Bytes((count * size) as u64));
        group.bench_function("fearless", |b| {
            b.iter(|| {
                engine.hash_many(black_box(&inputs), black_box(&mut outputs));
                black_box(&outputs);
            })
        });
        group.finish();
    }
}

criterion_group!(
    benches,
    bench_single,
    bench_many,
    bench_lane_fill,
    bench_mixed_short,
    bench_mixed_lengths,
    bench_batch_scaling,
);
criterion_main!(benches);
