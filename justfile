# disables panic handling
minsize command *args:
    RUSTFLAGS="-Ctarget-cpu=x86-64-v3" cargo {{command}} --target x86_64-unknown-linux-gnu -Z build-std=std,panic_abort -Z build-std-features=panic_immediate_abort --no-default-features --profile minsize {{args}}

# unwinding and backtraces enabled
releasedebug command *args:
    RUSTFLAGS="-Ctarget-cpu=x86-64-v3" cargo {{command}} --features backtrace --profile releasedebug {{args}}

devfast command *args:
    RUSTFLAGS="-Zub-checks -Ctarget-cpu=x86-64-v3" cargo {{command}} --features backtrace --profile devfast {{args}}

dev command *args:
    RUSTFLAGS="-Zub-checks -Ctarget-cpu=x86-64-v3" cargo {{command}} --features backtrace --profile dev {{args}}

# simple panic handling enabled
release-build-all-targets *args:
    RUSTFLAGS="-Ctarget-cpu=x86-64-v3" cargo build --target x86_64-unknown-linux-gnu -Z build-std=std,panic_abort -Z build-std-features= --release {{args}}
    mkdir natives/linux-x64-avx2+fma --parents
    -mv target/x86_64-unknown-linux-gnu/release/libnative_cull.so natives/linux-x64-avx2+fma/ || true

    RUSTFLAGS="-Ctarget-cpu=x86-64-v3" cargo zigbuild --target x86_64-apple-darwin -Z build-std=std,panic_abort -Z build-std-features= --release {{args}}
    mkdir natives/macos-x64-avx2+fma --parents
    -mv target/x86_64-apple-darwin/release/libnative_cull.dylib natives/macos-x64-avx2+fma/ || true

    RUSTFLAGS="-Ctarget-cpu=x86-64-v3" cargo xwin build --target x86_64-pc-windows-msvc -Z build-std=std,panic_abort -Z build-std-features= --release {{args}}
    mkdir natives/windows-x64-avx2+fma --parents
    -mv target/x86_64-pc-windows-msvc/release/native_cull.dll natives/windows-x64-avx2+fma/ || true

    RUSTFLAGS="-Ctarget-feature=+ssse3,+sse4.1" cargo build --target x86_64-unknown-linux-gnu -Z build-std=std,panic_abort -Z build-std-features= --release {{args}}
    mkdir natives/linux-x64-sse4_1+ssse3 --parents
    -mv target/x86_64-unknown-linux-gnu/release/libnative_cull.so natives/linux-x64-sse4_1+ssse3/

    RUSTFLAGS="-Ctarget-feature=+ssse3,+sse4.1" cargo zigbuild --target x86_64-apple-darwin -Z build-std=std,panic_abort -Z build-std-features= --release {{args}}
    mkdir natives/macos-x64-sse4_1+ssse3 --parents
    -mv target/x86_64-apple-darwin/release/libnative_cull.dylib natives/macos-x64-sse4_1+ssse3/ || true

    RUSTFLAGS="-Ctarget-feature=+ssse3,+sse4.1" cargo xwin build --target x86_64-pc-windows-msvc -Z build-std=std,panic_abort -Z build-std-features= --release {{args}}
    mkdir natives/windows-x64-sse4_1+ssse3 --parents
    -mv target/x86_64-pc-windows-msvc/release/native_cull.dll natives/windows-x64-sse4_1+ssse3/

    cargo zigbuild --target aarch64-unknown-linux-gnu -Z build-std=std,panic_abort -Z build-std-features= --release {{args}}
    mkdir natives/linux-arm64 --parents
    -mv target/aarch64-unknown-linux-gnu/release/libnative_cull.so natives/linux-arm64/

    cargo zigbuild --target aarch64-apple-darwin -Z build-std=std,panic_abort -Z build-std-features= --release {{args}}
    mkdir natives/macos-arm64 --parents
    -mv target/aarch64-apple-darwin/release/libnative_cull.dylib natives/macos-arm64/ || true

    cargo xwin build --target aarch64-pc-windows-msvc -Z build-std=std,panic_abort -Z build-std-features= --release {{args}}
    mkdir natives/windows-arm64 --parents
    -mv target/aarch64-pc-windows-msvc/release/native_cull.dll natives/windows-arm64/