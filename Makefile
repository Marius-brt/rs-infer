# Build profiles (ONNX Runtime prebuilt binaries do not combine all EPs).
BIN = ortinfer-server

.PHONY: mac mac-coreml gpu-cuda gpu-trt gpu-rtx cpu test fmt lint docs docs-build docs-clean

cpu:            ## CPU-only (works everywhere)
	cargo build --release -p ortinfer-server

mac-coreml:     ## macOS + CoreML/ANE
	cargo build --release -p $(BIN) --features ep-coreml

gpu-cuda:       ## Linux + CUDA
	cargo build --release -p $(BIN) --features ep-cuda

gpu-trt:        ## Linux + TensorRT (datacenter GPUs, includes CUDA)
	cargo build --release -p $(BIN) --features ep-cuda,ep-tensorrt

gpu-rtx:        ## Linux + TensorRT for RTX (consumer GeForce/RTX cards)
	cargo build --release -p $(BIN) --features ep-nvrtx,lax-ep-matching

mac: mac-coreml
test:
	cargo test --workspace
fmt:
	cargo fmt --all
lint:
	cargo clippy --workspace --all-targets

docs:           ## Serve the docs site locally (http://localhost:3000)
	cd docs-site && npm install && npm run dev

docs-build:     ## Static-export the docs site to docs-site/out (GitHub Pages-ready)
	cd docs-site && npm install && npm run build

docs-clean:
	rm -rf docs-site/out docs-site/.next docs-site/.source
