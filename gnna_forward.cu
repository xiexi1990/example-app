#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <stdio.h>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#define WARP_SIZE 32

__device__ inline 
void atomicAdd_F(float* address, float value)
{
  float old = value;  
  while ((old = atomicExch(address, atomicExch(address, 0.0f)+old))!=0.0f);
}

template <typename scalar_t>
__global__ void spmm_forward_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> row_pointers, 
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> column_index,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> degrees,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> part_pointers,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> part2Node,
    const int num_nodes, 
    const int dim,
    const int num_parts,
    const int partSize,
    const int dimWorker,
    const int warpPerBlock
) {

    int tid =  blockIdx.x * blockDim.x + threadIdx.x;  // global thread-id
    int warpId = tid / WARP_SIZE;                             // global warp-id
    int block_warpId = threadIdx.x / WARP_SIZE;               // block warp-id
    int laneid = threadIdx.x % WARP_SIZE;                     // warp thread-id -- laneid

    extern __shared__ int part_meta[];                                      // part information.
    int *partial_ids = part_meta;                                           // caching ids
    float *partial_results = (float*)&part_meta[partSize*warpPerBlock];     // caching partial results.

    if (warpId < num_parts){

        int srcId = part2Node[warpId];              // aggregated source node
        int partBeg = part_pointers[warpId];        // partitioning pointer start
        int partEnd = part_pointers[warpId + 1];    // part pointer end
        float src_norm = degrees[srcId];            // norm of the source node

        // Cache the part neighbors by all threads from a warp.
        const int pindex_base = block_warpId * partSize;
        #pragma unroll
        for (int nidx = partBeg + laneid; nidx < partEnd; nidx += WARP_SIZE){
            // if(column_index[nidx] >= num_nodes || column_index[nidx] < 0) printf("column_index: %d\n", column_index[nidx]);
            partial_ids[pindex_base + nidx - partBeg] = column_index[nidx];
        }
        
        __syncwarp();

        // Neighbor aggregation within each part
        const int presult_base = block_warpId * dim;
        for (int nIdx = 0; nIdx < partEnd - partBeg; nIdx++)
        {
            int nid = partial_ids[pindex_base + nIdx];
            // int nid = partial_ids[nIdx];
            // if (laneid == 0)
            //     printf("verify nid - 222222: %d\n", nid);
            float degree_norm_inv = __fmaf_rn(src_norm, degrees[nid], 0);

            // Initialize shared memory for partial results
            if (nIdx == 0)
                if (laneid < dimWorker)
                #pragma unroll
                for (int d = laneid; d < dim; d += dimWorker){
                    partial_results[presult_base + d] = 0.0f;
                }
            
            if (laneid < dimWorker)
            #pragma unroll
            for (int d = laneid; d < dim; d += dimWorker){
                // if(nid >= num_nodes || nid < 0) printf("aggregation: %d\n", nid);
                partial_results[presult_base + d] += __fmaf_rn(degree_norm_inv, input[nid][d], 0);
                // partial_results[presult_base + d] += input[nid][d];
            }
        }

        // output the result to global memory from the shared memory
        if (laneid < dimWorker)
        #pragma unroll
        for (int d = laneid; d < dim; d += dimWorker){
            atomicAdd_F((float*)&output[srcId][d], partial_results[presult_base + d]);
        }
    }
}


std::vector<torch::Tensor> spmm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int partSize, 
    int dimWorker, 
    int warpPerBlock
) 
{
    //auto tmp = torch::mm(input, weight);
    // auto output = torch::zeros_like(tmp);
    
    auto tmp = input;
    printf("change tmp to input!!!!!");
    auto output = torch::zeros({input.size(0), weight.size(1)}, torch::kCUDA);
    const int dim = output.size(1);
    const int num_nodes = output.size(0);
    const int num_parts = part2Node.size(0);

    const int block = warpPerBlock * WARP_SIZE;
    const int grid = (num_parts * WARP_SIZE + block  - 1) / block; 
    int shared_memory = partSize*warpPerBlock*sizeof(int)+warpPerBlock*dim*sizeof(float);

    // printf("grid: %d, block: %d\n", grid, block);
    // printf("dim: %d, num_nodes: %d, num_parts: %d\n", dim, num_nodes, num_parts);
    // printf("input: (%d, %d)\n", tmp.size(0), tmp.size(1));
    // printf("dimWorker: %d\n", dimWorker);
    // printf("shared_memory: %d\n", tmp.size(0), tmp.size(1));

    AT_DISPATCH_FLOATING_TYPES(input.type(), "spmm_cuda_forward", ([&] {
                                spmm_forward_cuda_kernel<scalar_t><<<grid, block, shared_memory>>>(
                                    output.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                    tmp.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                    row_pointers.packed_accessor32<int,1,torch::RestrictPtrTraits>(), 
                                    column_index.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                    degrees.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                                    part_pointers.packed_accessor32<int,1,torch::RestrictPtrTraits>(), 
                                    part2Node.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                    num_nodes, 
                                    dim,
                                    num_parts,
                                    partSize,
                                    dimWorker,
                                    warpPerBlock
                                );
                            }));
                                 
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    
    return {output};
}


int main() {
    torch::jit::script::Module container = torch::jit::load("gnna_forward_vs");

    torch::Tensor input = container.attr("tmp").toTensor().cuda();
    torch::Tensor weight = container.attr("weight").toTensor().cuda();
    torch::Tensor row_pointers = container.attr("inputInfo.row_pointers").toTensor().cuda();
    torch::Tensor column_index = container.attr("inputInfo.column_index").toTensor().cuda();
    torch::Tensor degrees = container.attr("inputInfo.degrees").toTensor().cuda();
    torch::Tensor part_pointers = container.attr("inputInfo.partPtr").toTensor().cuda();
    torch::Tensor part2Node = container.attr("inputInfo.part2Node").toTensor().cuda();
    int partSize = container.attr("inputInfo.partSize").toInt();
    int dimWorker = container.attr("inputInfo.dimWorker").toInt();
    int warpPerBlock = container.attr("inputInfo.warpPerBlock").toInt();
    container.attr("").toVec

    // std::cout << weight << std::endl;

    auto output = spmm_forward_cuda(input, weight, row_pointers, column_index, degrees, part_pointers, part2Node, partSize, dimWorker, warpPerBlock); 
    std::cout << "finish" << std::endl;
 
    return 0;
}