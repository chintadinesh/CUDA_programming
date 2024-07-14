#include <vector>
#include <cassert>
#include <iostream>

__global__ void square(float *d_out, float *d_in) {
  d_out[threadIdx.x] = d_in[threadIdx.x] * d_in[threadIdx.x];
}

int main() {
  constexpr int ARRAY_SIZE = 64;
  constexpr int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

  // host side memory allocation
  std::vector<float> h_in(ARRAY_SIZE, 0);
  for(int i = 0; i < ARRAY_SIZE; ++i) h_in[i] = static_cast<float>(i);
  std::vector<float> h_out(ARRAY_SIZE, 0);

  // GPU memory
  float *d_in, *d_out;
  cudaMalloc((void **)&d_in, ARRAY_BYTES);
  cudaMalloc((void **)&d_out, ARRAY_BYTES);

  cudaMemcpy(d_in, &h_in[0], ARRAY_BYTES, cudaMemcpyHostToDevice);

  square<<<1, ARRAY_SIZE>>>(d_out, d_in);

  cudaMemcpy(&h_out[0], d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

  // clean up
  cudaFree(d_in);
  cudaFree(d_out);


  // verify the results
  for(int i = 0; i < ARRAY_SIZE; i++) assert(h_out[i] == h_in[i] * h_in[i]);
  std::cout << "TEST PASSED!\n";

  return 0; }