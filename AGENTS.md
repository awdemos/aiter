# AITER

AI Tensor Engine for ROCm — a Python/C++ library of optimized AMD GPU kernels for inference and training.

## Project layout

```
aiter/              Python operator bindings and JIT helpers
aiter/aot/flydsl    FlyDSL-generated kernel sources
aiter/jit/          JIT build artifacts (excluded from lint)
aiter/ops/          Operator entry points
aiter/configs/      Tuned GEMM / MoE configs
csrc/               C++/HIP/ASM kernel sources
op_tests/           Per-operator correctness tests (Python)
3rdparty/           Vendored submodules (excluded from lint)
gradlib/            Gradient helper library
scripts/            Build/test helper scripts
```

## Setup commands

```bash
# Clone with submodules (required)
git clone --recursive https://github.com/awdemos/aiter.git
# or, if already cloned:
git submodule sync && git submodule update --init --recursive

# Install in develop mode (builds extensions)
python3 setup.py develop

# Install runtime deps only
pip install -r requirements.txt
```

## Build/test/lint commands

```bash
# Run ruff (CI pins ruff==0.16.0)
ruff check .

# Run operator tests (require ROCm GPU)
python3 op_tests/test_mha.py
python3 op_tests/test_mla.py
python3 op_tests/test_moe.py
python3 op_tests/test_gemm_a8w8.py
python3 op_tests/test_rmsnorm2d.py
ls op_tests/test_*.py   # full list
```

## Key conventions

- Python target is `py310`. Ruff config in `pyproject.toml` pins `target-version` and explicitly excludes `3rdparty/`, `aiter/jit/build`, and `aiter_logs/`.
- C++ / HIP code uses `.clang-format` and `.clang-tidy` settings at the repo root.
- FlyDSL is a build-time dependency (`flydsl==0.3.1` in `pyproject.toml`).
- New operators need a matching `op_tests/test_<name>.py` correctness test.
- Versioning uses `setuptools_scm` from git tags; fallback version is `0.1.0`.

## Gotchas

- The `aiter/dist/` package is real source code, not setuptools output. The ruff `exclude` list uses `dist/` (with trailing slash) so the output dir is skipped but `aiter/dist/` is still linted.
- CI runs `ruff check .` on PRs; reviewdog reports findings and fails on error.
- `setup.py` uses `ninja` and builds HIP/ASM kernels; a ROCm toolchain is required for the C++ build.
- The repository contains large generated files; run `python3 setup.py develop` before importing `aiter`.
