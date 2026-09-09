# syntax=docker/dockerfile:1
# GPU EP combos (pyke prebuilt binaries, CUDA/TensorRT exist only for linux/amd64):
#   ep-cuda | ep-cuda,ep-tensorrt | ep-nvrtx,lax-ep-matching
# To build the GPU image from Apple Silicon, run: docker build --platform linux/amd64 .
# Leave FEATURES unset for target-aware auto-selection (amd64 -> GPU EPs, other -> CPU-only).
# ONNX Runtime prebuilts need libstdc++ >= GCC 13 (trixie), not bookworm's GCC 12.
FROM rust:1.96-trixie AS build
ARG TARGETPLATFORM
ARG FEATURES
WORKDIR /src
COPY . .
RUN set -eux; \
    if [ -z "${FEATURES:-}" ]; then \
        case "${TARGETPLATFORM:-linux/amd64}" in \
            linux/amd64) FEATURES="ep-cuda,ep-tensorrt" ;; \
            *) FEATURES="" ;; \
        esac; \
    fi; \
    if [ -n "$FEATURES" ]; then \
        cargo build --release -p rsinfer-server --no-default-features --features "$FEATURES"; \
    else \
        cargo build --release -p rsinfer-server --no-default-features; \
    fi

FROM debian:trixie-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libgomp1 libstdc++6 curl && rm -rf /var/lib/apt/lists/*
# Add CUDA/cuDNN/TensorRT runtime libraries to this image (or mount them) when using GPU EPs.
COPY --from=build /src/target/release/rsinfer-server /usr/local/bin/rsinfer-server
EXPOSE 8080
USER nobody
ENTRYPOINT ["/usr/local/bin/rsinfer-server"]
CMD ["--config", "/etc/rsinfer/config.yaml"]
