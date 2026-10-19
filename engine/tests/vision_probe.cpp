// Runs the engine's vision tower + aligner on prepared images, for scripts/vision_check.py.
//   vision_probe <weights dir> <manifest.json>
// manifest: [{"pixels": "<RGB file>", "n_vit_h": H, "n_vit_w": W, "out": "<BF16 rows file>"}, ...]
// Each output file holds n_llm_h * n_llm_w rows of 5120 BF16 values.
#include "ds41/vision.hpp"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <iterator>
int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: vision_probe <weights dir> <manifest.json>\n";
    return 2;
  }
  try {
    ds41e::Store source(argv[1]);
    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess) throw std::runtime_error("stream");
    ds41e::VisionEncoder encoder(source, stream);
    for (const auto &item : ds41e::read_json(argv[2])) {
      ds41e::ImageSpan img;
      img.n_vit_h = item.at("n_vit_h").get<int>();
      img.n_vit_w = item.at("n_vit_w").get<int>();
      img.n_llm_h = (img.n_vit_h + 2) / 3;
      img.n_llm_w = (img.n_vit_w + 2) / 3;
      std::ifstream in(item.at("pixels").get<std::string>(), std::ios::binary);
      img.pixels.assign(std::istreambuf_iterator<char>(in), {});
      void *rows = encoder.encode(img);
      const size_t bytes = size_t(img.n_llm_h) * img.n_llm_w * 5120 * 2;
      std::vector<char> host(bytes);
      if (cudaMemcpy(host.data(), rows, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) throw std::runtime_error("copy");
      cudaFree(rows);
      std::ofstream(item.at("out").get<std::string>(), std::ios::binary).write(host.data(), std::streamsize(bytes));
      std::cout << item.at("out").get<std::string>() << " patches=" << img.n_vit_h * img.n_vit_w
                << " ms=" << encoder.last_ms() << std::endl;
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "vision_probe: " << e.what() << "\n";
    return 1;
  }
}
