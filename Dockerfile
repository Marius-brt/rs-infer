# syntax=docker/dockerfile:1
# GPU EP combos (pyke prebuilt binaries): ep-cuda | ep-cuda,ep-tensorrt | ep-nvrtx,lax-ep-matching
ARG FEATURES=ep-cuda,ep-tensorrt
FROM rust:1.96-bookworm AS build
ARG FEATURES
WORKDIR /src
COPY . .
RUN cargo build --release -p ortinfer-server --no-default-features --features ${FEATURES}

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libgomp1 curl && rm -rf /var/lib/apt/lists/*
# Add CUDA/cuDNN/TensorRT runtime libraries to this image (or mount them) when using GPU EPs.
COPY --from=build /src/target/release/ortinfer-server /usr/local/bin/ortinfer-server
EXPOSE 8080
USER nobody
ENTRYPOINT ["/usr/local/bin/ortinfer-server"]
CMD ["--config", "/etc/ortinfer/config.yaml"]
