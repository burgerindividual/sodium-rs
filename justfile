# disables panic handling
minsize:
    RUSTFLAGS="-Ctarget-cpu=x86-64-v3" cargo build --target x86_64-unknown-linux-gnu -Z build-std=std,panic_abort -Z build-std-features=panic_immediate_abort --no-default-features --profile production

production:
    RUSTFLAGS="-Ctarget-cpu=x86-64-v3" cargo build --target x86_64-unknown-linux-gnu -Z build-std=std,panic_abort --profile production

fastdebug:
    RUSTFLAGS="-Zub-checks -Ctarget-cpu=x86-64-v3" cargo build --release

slowdebug:
    RUSTFLAGS="-Zub-checks -Ctarget-cpu=x86-64-v3" cargo build --profile dev