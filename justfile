# disables panic handling
minsize command *args:
    RUSTFLAGS="-Ctarget-cpu=x86-64-v3" cargo {{command}} --target x86_64-unknown-linux-gnu -Z build-std=std,panic_abort -Z build-std-features=panic_immediate_abort --no-default-features --profile minsize {{args}}

# simple panic handling enabled
release command *args:
    RUSTFLAGS="-Ctarget-cpu=x86-64-v3" cargo {{command}} --target x86_64-unknown-linux-gnu -Z build-std=std,panic_abort -Z build-std-features= --release {{args}}

# unwinding and backtraces enabled
releasedebug command *args:
    RUSTFLAGS="-Ctarget-cpu=x86-64-v3" cargo {{command}} --features backtrace --profile releasedebug {{args}}

devfast command *args:
    RUSTFLAGS="-Zub-checks -Ctarget-cpu=x86-64-v3" cargo {{command}} --features backtrace --profile devfast {{args}}

dev command *args:
    RUSTFLAGS="-Zub-checks -Ctarget-cpu=x86-64-v3" cargo {{command}} --features backtrace --profile dev {{args}}

# asm profile="asm" target="":
#     RUSTFLAGS="-Ctarget-cpu=x86-64-v3" cargo asm --target x86_64-unknown-linux-gnu -Z build-std=std,panic_abort --profile {{profile}} --include-constants {{target}}
