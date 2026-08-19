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
fn bench_aarch64_neon4(c: &mut Criterion) {
    use md5_many::bench_internals::md5_aarch64_neon4;

    let engine = md5_many::Md5Many::new();
    if engine.lanes() != 4 {
        return;
    }

    for &len in &[0usize, 55, 64, 1024, 64 * 1024, 1024 * 1024] {
        let data: [Vec<u8>; 4] = core::array::from_fn(|lane| {
            (0..len)
                .map(|index| (index as u8).wrapping_add(lane as u8 * 41))
                .collect()
        });
        let inputs = data.each_ref().map(|input| input.as_slice());
        let mut generic_outputs = [[0u8; 16]; 4];
        let mut group = c.benchmark_group(format!("backend-aarch64-neon4-{len}"));
        group.sample_size(20);
        group.warm_up_time(Duration::from_millis(250));
        group.measurement_time(Duration::from_millis(750));
        group.throughput(Throughput::Bytes((len * 4) as u64));
        group.bench_function("fearless-neon", |b| {
            b.iter(|| {
                engine.hash_many(black_box(&inputs), black_box(&mut generic_outputs));
                black_box(generic_outputs)
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

criterion_group!(
    benches,
    bench_short_framing,
    bench_aligned_framing,
    bench_x86_avx512_short,
    bench_x86_single_stream,
    bench_x86_avx512_digest_store,
    bench_aarch64_single_stream,
    bench_aarch64_neon4
);
criterion_main!(benches);
