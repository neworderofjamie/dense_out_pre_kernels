// Standard C++ includes
#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

// Standard C includes
#include <cassert>
#include <cmath>

// CUDA includes
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

//------------------------------------------------------------------------
// Macros
//------------------------------------------------------------------------
#define SEED 124
#define BLOCK_SIZE 32

#define CHECK_CUDA_ERRORS(call) {                                                                   \
    cudaError_t error = call;                                                                       \
    if (error != cudaSuccess) {                                                                     \
            std::ostringstream errorMessageStream;                                                  \
            errorMessageStream << "cuda error:" __FILE__ << ": " << __LINE__ << " ";                \
            errorMessageStream << cudaGetErrorString(error) << "(" << error << ")" << std::endl;    \
            throw std::runtime_error(errorMessageStream.str());                                     \
        }                                                                                           \
    }

template<typename T>
using HostDeviceArray = std::pair < T*, T* > ;

enum Mode
{
    ModeGlobalAtomic,
    ModeGlobalAtomicLifted,
    ModeSharedAtomic,
    ModeWarpShuffle,
    ModeWarpShuffleLifted,
    ModeMax,
};

const char *const s_ModeNames[] = {
    "Global atomic",
    "Global atomic lifted",
    "Shared atomic",
    "Warp shuffle",
    "Warp shuffle lifted"};

//------------------------------------------------------------------------
// Timer
//------------------------------------------------------------------------
template<typename A = std::milli>
class Timer
{
public:
    Timer(const std::string &title) : m_Start(std::chrono::high_resolution_clock::now()), m_Title(title)
    {
    }

    ~Timer()
    {
        std::cout << m_Title << get() << std::endl;
    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    double get() const
    {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = now - m_Start;
        return duration.count();
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::chrono::time_point<std::chrono::high_resolution_clock> m_Start;
    std::string m_Title;
};

//-----------------------------------------------------------------------------
// Device functions
//-----------------------------------------------------------------------------
//  Von Neumann'synBlk exponential distribution generator from Ripley p.230
//  Mean number of U(0,1) per call = 5.2
__device__ float exponentialDist(curandState &rng) {
    float a = 0.0f;

    while (true) {
        float u = curand_uniform(&rng);
        const float u0 = u;

        while (true) {
            float uStar = curand_uniform(&rng);
            if (u < uStar) {
                return  a + u0;
            }

            u = curand_uniform(&rng);

            if (u >= uStar) {
                break;
            }
        }

        a += 1.0f;
    }
}
//-----------------------------------------------------------------------------
// Kernel to initialise device RNG seed
template<typename RNGStateType>
__global__ void initRandomSeed(unsigned int sequenceStart, unsigned int numSeed, RNGStateType *d_rowState)
{
    const int i = threadIdx.x + (blockIdx.x * BLOCK_SIZE);
    if (i < numSeed) {
        curand_init(SEED, sequenceStart + i, 0, &d_rowState[i]);
    }

}
//-----------------------------------------------------------------------------
// Kernel to initialise initial Poisson time-to-spike
__global__ void initPoissonTimeToSpike(unsigned int numPoisson, float meanISI, curandState *d_poissonState,
                                       float *d_timeToSpike)
{
    // Get index of neuron in population
    const int i = threadIdx.x + (blockIdx.x * BLOCK_SIZE);
    if (i < numPoisson) {
        d_timeToSpike[i] = meanISI * exponentialDist(d_poissonState[i]);
    }
}
//-----------------------------------------------------------------------------
// Kernel to simulate population of poisson neurons
__global__ void poisson(unsigned int numPoisson, float meanISI, curandState *d_poissonState,
                        float *d_timeToSpike, unsigned int *d_numOutSpikes, unsigned int *d_outSpikes)
{
    // Count and buffer to hold spikes output by this block
    __shared__ unsigned int blockSpikeCount;
    __shared__ unsigned int blockOutSpikes[BLOCK_SIZE];

    // Offset into global spike output buffer
    __shared__ unsigned int blockSpikeOffset;

    // Get index of neuron in population
    const unsigned int batch = blockIdx.y;
    const int i = threadIdx.x + (blockIdx.x * BLOCK_SIZE);

    const unsigned int batchOffset = numPoisson * batch;

    // Use first thread in each block to zero spike counts
    if (threadIdx.x == 0) {
        blockSpikeCount = 0;
    }
    __syncthreads();

    // If there is a neuron for this thread to simulate
    if (i < numPoisson) {
        float tts = d_timeToSpike[i];

        if (tts <= 0.0f) {
            tts += (meanISI * exponentialDist(d_poissonState[batchOffset + i]));

            // Add spike to output
            unsigned int blockSpikeIndex = atomicAdd(&blockSpikeCount, 1);
            blockOutSpikes[blockSpikeIndex] = i;
        }

        d_timeToSpike[batchOffset + i] = (tts - 1.0f);
    }

    // If block has emitted any spikes, use the first thread to  
    // determine where in global spike output buffer to copy them
    __syncthreads();
    if (threadIdx.x == 0 && blockSpikeCount > 0) {
        blockSpikeOffset = atomicAdd(&d_numOutSpikes[batch], blockSpikeCount);
    }

    // Copy spikes from block output buffer into correct offset in global buffer
    __syncthreads();
    if (threadIdx.x < blockSpikeCount) {
        d_outSpikes[batchOffset + blockSpikeOffset + threadIdx.x] = blockOutSpikes[threadIdx.x];
    }
}
//-----------------------------------------------------------------------------
__global__ void globalAtomic(unsigned int numPre, unsigned int numPost, const unsigned int *d_numInSpikes, 
                             const unsigned int *d_inSpikes, const float *d_weights, const float *d_lambdaV,
                             const float *d_lambdaI, float *d_outCurrents, float *d_gradient)
{
    __shared__ unsigned int s_spike[BLOCK_SIZE];

    const unsigned int batch = blockIdx.y;
    const unsigned int id = threadIdx.x + (blockIdx.x * BLOCK_SIZE);

    const unsigned int preBatchOffset = numPre * batch;
    const unsigned int postBatchOffset = numPost * batch;
    const unsigned int synBatchOffset = preBatchOffset * numPost;

    // Calculate number of blocks (dictated by shared memory) spikes need to be processed in
    const unsigned int numSpikes = d_numInSpikes[batch];
    const unsigned int numSpikeBlocks = (numSpikes + BLOCK_SIZE - 1) / BLOCK_SIZE;
    

    // Loop through spikes blocks
    for (unsigned int b = 0; b < numSpikeBlocks; b++) {
        // Determine how many spikes are in this block
        const unsigned int numSpikesInBlock = (b == (numSpikeBlocks - 1))
            ? ((numSpikes - 1) % BLOCK_SIZE) + 1 : BLOCK_SIZE;
        __syncthreads();
       // Use first row of threads in block to read spikes and row lengths into shared memory
        if (threadIdx.x < numSpikesInBlock) {
            const unsigned int i = d_inSpikes[preBatchOffset + (b * BLOCK_SIZE) + threadIdx.x];
            s_spike[threadIdx.x] = i;
        }

        __syncthreads();

        // If there is a synapse for this thread to process
        if(id < numPost) {
            // Loop through spikes in block
            for(unsigned int i = 0; i < numSpikesInBlock; i++) {
                // Get postsynaptic index
                const unsigned int synAddress = (s_spike[i] * numPost) + id;

                // Update gradient and back-propagate
                d_gradient[synBatchOffset + synAddress] -= (d_lambdaI[postBatchOffset + id] * 5.000000000e+00f);
                atomicAdd(&d_outCurrents[preBatchOffset + s_spike[i]], d_weights[synAddress] * (d_lambdaV[postBatchOffset + id] - d_lambdaI[postBatchOffset + id]));
            }
        }

    }
}
//-----------------------------------------------------------------------------
__global__ void globalAtomicLifted(unsigned int numPre, unsigned int numPost, const unsigned int *d_numInSpikes, 
                                   const unsigned int *d_inSpikes, const float *d_weights, const float *d_lambdaV,
                                   const float *d_lambdaI, float *d_outCurrents, float *d_gradient)
{
    __shared__ unsigned int s_spike[BLOCK_SIZE];

    const unsigned int batch = blockIdx.y;
    const unsigned int id = threadIdx.x + (blockIdx.x * BLOCK_SIZE);

    const unsigned int preBatchOffset = numPre * batch;
    const unsigned int postBatchOffset = numPost * batch;
    const unsigned int synBatchOffset = preBatchOffset * numPost;

    // Calculate number of blocks (dictated by shared memory) spikes need to be processed in
    const unsigned int numSpikes = d_numInSpikes[batch];
    const unsigned int numSpikeBlocks = (numSpikes + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Lift reads of lambda
    const float lambdaV = (id < numPost) ? d_lambdaV[postBatchOffset + id] : 0.0f;
    const float lambdaI = (id < numPost) ? d_lambdaI[postBatchOffset + id] : 0.0f;
    const float lambdaVI = lambdaV - lambdaI;
    
    // Loop through spikes blocks
    for (unsigned int b = 0; b < numSpikeBlocks; b++) {
        // Determine how many spikes are in this block
        const unsigned int numSpikesInBlock = (b == (numSpikeBlocks - 1))
            ? ((numSpikes - 1) % BLOCK_SIZE) + 1 : BLOCK_SIZE;
        __syncthreads();
       // Use first row of threads in block to read spikes and row lengths into shared memory
        if (threadIdx.x < numSpikesInBlock) {
            const unsigned int i = d_inSpikes[preBatchOffset + (b * BLOCK_SIZE) + threadIdx.x];
            s_spike[threadIdx.x] = i;
        }

        __syncthreads();

        // If there is a synapse for this thread to process
        if(id < numPost) {
            // Loop through spikes in block
            for(unsigned int i = 0; i < numSpikesInBlock; i++) {
                // Get postsynaptic index
                const unsigned int synAddress = (s_spike[i] * numPost) + id;

                // Update gradient and back-propagate
                d_gradient[synBatchOffset + synAddress] -= (lambdaI * 5.000000000e+00f);
                atomicAdd(&d_outCurrents[preBatchOffset + s_spike[i]], d_weights[synAddress] * lambdaVI);
            }
        }

    }

}
//-----------------------------------------------------------------------------
__global__ void sharedAtomic(unsigned int numPre, unsigned int numPost, const unsigned int* d_numInSpikes,
    const unsigned int* d_inSpikes, const float* d_weights, const float* d_lambdaV,
    const float* d_lambdaI, float* d_outCurrents, float* d_gradient)
{
    __shared__ unsigned int s_spike[BLOCK_SIZE];
    __shared__ float s_outCurrents[BLOCK_SIZE];

    const unsigned int batch = blockIdx.y;
    const unsigned int id = threadIdx.x + (blockIdx.x * BLOCK_SIZE);

    const unsigned int preBatchOffset = numPre * batch;
    const unsigned int postBatchOffset = numPost * batch;
    const unsigned int synBatchOffset = preBatchOffset * numPost;

    // Calculate number of blocks (dictated by shared memory) spikes need to be processed in
    const unsigned int numSpikes = d_numInSpikes[batch];
    const unsigned int numSpikeBlocks = (numSpikes + BLOCK_SIZE - 1) / BLOCK_SIZE;


    // Loop through spikes blocks
    for (unsigned int b = 0; b < numSpikeBlocks; b++) {
        // Determine how many spikes are in this block
        const unsigned int numSpikesInBlock = (b == (numSpikeBlocks - 1))
            ? ((numSpikes - 1) % BLOCK_SIZE) + 1 : BLOCK_SIZE;
        __syncthreads();
        // Use first row of threads in block to read spikes and row lengths into shared memory
        if (threadIdx.x < numSpikesInBlock) {
            const unsigned int i = d_inSpikes[preBatchOffset + (b * BLOCK_SIZE) + threadIdx.x];
            s_spike[threadIdx.x] = i;
            s_outCurrents[threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // If there is a synapse for this thread to process
        if (id < numPost) {
            // Loop through spikes in block
            for (unsigned int i = 0; i < numSpikesInBlock; i++) {
                // Get postsynaptic index
                const unsigned int synAddress = (s_spike[i] * numPost) + id;

                // Update gradient and back-propagate
                d_gradient[synBatchOffset + synAddress] -= (d_lambdaI[postBatchOffset + id] * 5.000000000e+00f);
                atomicAdd(&s_outCurrents[i], d_weights[synAddress] * (d_lambdaV[postBatchOffset + id] - d_lambdaI[postBatchOffset + id]));
            }
        }

        __syncthreads();

        if (threadIdx.x < numSpikesInBlock) {
            atomicAdd(&d_outCurrents[preBatchOffset + s_spike[threadIdx.x]], s_outCurrents[threadIdx.x]);
        }

        __syncthreads();
    }
}
//-----------------------------------------------------------------------------
__global__ void warpReduction(unsigned int numPre, unsigned int numPost, const unsigned int* d_numInSpikes,
    const unsigned int* d_inSpikes, const float* d_weights, const float* d_lambdaV,
    const float* d_lambdaI, float* d_outCurrents, float* d_gradient)
{
    __shared__ unsigned int s_spike[BLOCK_SIZE];

    const unsigned int batch = blockIdx.y;
    const unsigned int id = threadIdx.x + (blockIdx.x * BLOCK_SIZE);
    const unsigned int lane = threadIdx.x % 32;

    const unsigned int preBatchOffset = numPre * batch;
    const unsigned int postBatchOffset = numPost * batch;
    const unsigned int synBatchOffset = preBatchOffset * numPost;

    // Calculate number of blocks (dictated by shared memory) spikes need to be processed in
    const unsigned int numSpikes = d_numInSpikes[batch];
    const unsigned int numSpikeBlocks = (numSpikes + BLOCK_SIZE - 1) / BLOCK_SIZE;


    // Loop through spikes blocks
    for (unsigned int b = 0; b < numSpikeBlocks; b++) {
        // Determine how many spikes are in this block
        const unsigned int numSpikesInBlock = (b == (numSpikeBlocks - 1))
            ? ((numSpikes - 1) % BLOCK_SIZE) + 1 : BLOCK_SIZE;
        __syncthreads();
        // Use first row of threads in block to read spikes and row lengths into shared memory
        if (threadIdx.x < numSpikesInBlock) {
            const unsigned int i = d_inSpikes[preBatchOffset + (b * BLOCK_SIZE) + threadIdx.x];
            s_spike[threadIdx.x] = i;
        }

        __syncthreads();

        
        // Loop through spikes in block
        for (unsigned int i = 0; i < numSpikesInBlock; i++) {
            // If there is a synapse for this thread to process
            float outCurrent = 0.0f;
            if (id < numPost) {
                // Get postsynaptic index
                const unsigned int synAddress = (s_spike[i] * numPost) + id;

                // Update gradient and back-propagate
                d_gradient[synBatchOffset + synAddress] -= (d_lambdaI[postBatchOffset + id] * 5.000000000e+00f);

                // Calculate output current
                outCurrent += d_weights[synAddress] * (d_lambdaV[postBatchOffset + id] - d_lambdaI[postBatchOffset + id]);
            }
                
            // Perform warp-level tree reduction into first lane
            outCurrent += __shfl_down_sync(0xFFFFFFFF, outCurrent, 16);
            outCurrent += __shfl_down_sync(0xFFFFFFFF, outCurrent, 8);
            outCurrent += __shfl_down_sync(0xFFFFFFFF, outCurrent, 4);
            outCurrent += __shfl_down_sync(0xFFFFFFFF, outCurrent, 2);
            outCurrent += __shfl_down_sync(0xFFFFFFFF, outCurrent, 1);

            // Issue atomic add on first lane of warp
            if (lane == 0) {
                atomicAdd(&d_outCurrents[preBatchOffset + s_spike[i]], outCurrent);
            }
        }
    }
}
//-----------------------------------------------------------------------------
__global__ void warpReductionLifted(unsigned int numPre, unsigned int numPost, const unsigned int* d_numInSpikes,
    const unsigned int* d_inSpikes, const float* d_weights, const float* d_lambdaV,
    const float* d_lambdaI, float* d_outCurrents, float* d_gradient)
{
    __shared__ unsigned int s_spike[BLOCK_SIZE];

    const unsigned int batch = blockIdx.y;
    const unsigned int id = threadIdx.x + (blockIdx.x * BLOCK_SIZE);
    const unsigned int lane = threadIdx.x % 32;

    const unsigned int preBatchOffset = numPre * batch;
    const unsigned int postBatchOffset = numPost * batch;
    const unsigned int synBatchOffset = preBatchOffset * numPost;

    // Calculate number of blocks (dictated by shared memory) spikes need to be processed in
    const unsigned int numSpikes = d_numInSpikes[batch];
    const unsigned int numSpikeBlocks = (numSpikes + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Lift reads of lambda
    const float lambdaV = (id < numPost) ? d_lambdaV[postBatchOffset + id] : 0.0f;
    const float lambdaI = (id < numPost) ? d_lambdaI[postBatchOffset + id] : 0.0f;
    const float lambdaVI = lambdaV - lambdaI;

    // Loop through spikes blocks
    for (unsigned int b = 0; b < numSpikeBlocks; b++) {
        // Determine how many spikes are in this block
        const unsigned int numSpikesInBlock = (b == (numSpikeBlocks - 1))
            ? ((numSpikes - 1) % BLOCK_SIZE) + 1 : BLOCK_SIZE;
        __syncthreads();
        // Use first row of threads in block to read spikes and row lengths into shared memory
        if (threadIdx.x < numSpikesInBlock) {
            const unsigned int i = d_inSpikes[preBatchOffset + (b * BLOCK_SIZE) + threadIdx.x];
            s_spike[threadIdx.x] = i;
        }

        __syncthreads();

        // Loop through spikes in block
        for (unsigned int i = 0; i < numSpikesInBlock; i++) {
            // If there is a synapse for this thread to process
            float outCurrent = 0.0f;
            if (id < numPost) {
                // Get postsynaptic index
                const unsigned int synAddress = (s_spike[i] * numPost) + id;

                // Update gradient and back-propagate
                d_gradient[synBatchOffset + synAddress] -= (lambdaI * 5.000000000e+00f);

                // Calculate output current
                outCurrent += d_weights[synAddress] * lambdaVI;
            }

            // Perform warp-level tree reduction into first lane
            outCurrent += __shfl_down_sync(0xFFFFFFFF, outCurrent, 16);
            outCurrent += __shfl_down_sync(0xFFFFFFFF, outCurrent, 8);
            outCurrent += __shfl_down_sync(0xFFFFFFFF, outCurrent, 4);
            outCurrent += __shfl_down_sync(0xFFFFFFFF, outCurrent, 2);
            outCurrent += __shfl_down_sync(0xFFFFFFFF, outCurrent, 1);

            // Issue atomic add on first lane of warp
            if (lane == 0) {
                atomicAdd(&d_outCurrents[preBatchOffset + s_spike[i]], outCurrent);
            }
        }
    }
}

//-----------------------------------------------------------------------------
// Host functions
//-----------------------------------------------------------------------------
template<typename T>
HostDeviceArray<T> allocateHostDevice(unsigned int count)
{
    T *array = nullptr;
    T *d_array = nullptr;
    CHECK_CUDA_ERRORS(cudaMallocHost(&array, count * sizeof(T)));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_array, count * sizeof(T)));

    return std::make_pair(array, d_array);
}
//-----------------------------------------------------------------------------
template<typename T>
void hostToDeviceCopy(HostDeviceArray<T> &array, unsigned int count, bool deleteHost=false)
{
    CHECK_CUDA_ERRORS(cudaMemcpy(array.second, array.first, sizeof(T) * count, cudaMemcpyHostToDevice));
    if (deleteHost) {
        CHECK_CUDA_ERRORS(cudaFreeHost(array.first));
        array.first = nullptr;
    }
}
//-----------------------------------------------------------------------------
template<typename T>
void deviceToHostCopy(HostDeviceArray<T> &array, unsigned int count)
{
    CHECK_CUDA_ERRORS(cudaMemcpy(array.first, array.second, count * sizeof(T), cudaMemcpyDeviceToHost));
}
//-----------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    try
    {
        unsigned int numPre = 256;
        unsigned int numPost = 256;
        unsigned int numBatch = 32;
        const float dt = 1.0f;
        const float poissonRate = 10.0f;
        const float poissonMeanISI = 1000.0f / (poissonRate * dt);

        // Read mode from command line
        Mode mode;
        if(argc < 2) {
            std::cerr << "Expected parameters specifying:" << std::endl;
            std::cerr << "\t Mode (";
            for(int m = 0; m < ModeMax; m++) {
                std::cerr << m << " = " << s_ModeNames[m];
                if(m != (ModeMax - 1)) {
                    std::cerr << ", ";
                }
            }
            std::cerr << ")" << std::endl;
            return EXIT_FAILURE;
        }
        else {
            mode = (Mode)std::stoul(argv[1]);
        }
    
        // If additional parameters are specified, read N
        if(argc > 2) {
            numPre = numPost = std::stoul(argv[2]);
        }

        // If additional parameters are specified, read B
        if(argc > 3) {
            numBatch = std::stoul(argv[3]);
        }

        const unsigned int preBlocks = (unsigned int)std::ceil((float)numPre / (float)BLOCK_SIZE);
        const unsigned int preBatchBlocks = (unsigned int)std::ceil((float)(numPre * numBatch)/ (float)BLOCK_SIZE);
        std::cout << "Mode:" << s_ModeNames[mode] << " pre:" << numPre << ", num post:" << numPost << ", num batch:" << numBatch << std::endl;
    
        CHECK_CUDA_ERRORS(cudaSetDevice(0));

        //------------------------------------------------------------------------
        // Configure fixed-probability connector
        //------------------------------------------------------------------------
        // Create arrays to hold pre-synaptic currents
        const unsigned int numOutCurrents = numPre * numBatch;
        auto outCurrents = allocateHostDevice<float>(numOutCurrents);
        std::fill_n(&outCurrents.first[0], numOutCurrents, 0.0f);
        hostToDeviceCopy(outCurrents, numOutCurrents);

        // Allocate, fill and upload weight array
        const unsigned int numSynapses = numPre * numPost;
        HostDeviceArray<float> weights = allocateHostDevice<float>(numSynapses);
        std::fill_n(&weights.first[0], numSynapses, 1.0f);
        hostToDeviceCopy(weights, numSynapses, true);

        // Allocate, fill and upload gradient array
        const unsigned int numGradients = numSynapses * numBatch;
        std::cout << "NUM GRADIENTS:" << numGradients << std::endl;
        HostDeviceArray<float> gradients = allocateHostDevice<float>(numGradients);
        std::fill_n(&gradients.first[0], numGradients, 0.0f);
        hostToDeviceCopy(gradients, numGradients, true);

        // Allocate, fill and upload lambda arrays
        const unsigned int numLambda = numPost * numBatch;
        HostDeviceArray<float> lambdaV = allocateHostDevice<float>(numLambda);
        HostDeviceArray<float> lambdaI = allocateHostDevice<float>(numLambda);
        std::fill_n(&lambdaV.first[0], numLambda, 1.0f);
        std::fill_n(&lambdaI.first[0], numLambda, 0.0f);
        hostToDeviceCopy(lambdaV, numLambda, true);
        hostToDeviceCopy(lambdaI, numLambda, true);

        //------------------------------------------------------------------------
        // Configure poisson population
        //------------------------------------------------------------------------
        // Create arrays to hold poisson spike count
        auto poissonNumSpikes = allocateHostDevice<unsigned int>(numBatch);

        // Create arrays to hold poisson spikes
        auto poissonSpikes = allocateHostDevice<unsigned int>(numPre * numBatch);

        // Create device random number generator states for poisson generators
        curandState *d_poissonState = nullptr;
        CHECK_CUDA_ERRORS(cudaMalloc(&d_poissonState, numPre * numBatch * sizeof(curandState)));
        {
            Timer<std::milli> t("Seed poisson:");
            // Initialise these seeds using kernel
            // **NOTE** first numPre sequences used by Poisson spike sources
            initRandomSeed <<<preBatchBlocks, BLOCK_SIZE>>>(0, numPre, d_poissonState);
            cudaDeviceSynchronize();
        }

        // Create device array for poisson generator time to spike
        float *d_poissonTimeToSpike = nullptr;
        CHECK_CUDA_ERRORS(cudaMalloc(&d_poissonTimeToSpike, numPre * numBatch * sizeof(float)));

        // Initialise time to spike using kernel
        {
            Timer<std::milli> t("Init poisson TTS:");
            initPoissonTimeToSpike <<<preBatchBlocks, BLOCK_SIZE>>>(numPre, poissonMeanISI, d_poissonState, d_poissonTimeToSpike);
            cudaDeviceSynchronize();
        }

        // Create timing events
        cudaEvent_t kernelStartEvent;
        cudaEvent_t kernelEndEvent;
        double kernelTime = 0.0;
        CHECK_CUDA_ERRORS(cudaEventCreate(&kernelStartEvent));
        CHECK_CUDA_ERRORS(cudaEventCreate(&kernelEndEvent));

        {
            // Loop through time
            for (unsigned int t = 0; t < 1000; t++) {
                // Zero spike counters
                std::fill_n(&poissonNumSpikes.first[0], numBatch, 0);
                hostToDeviceCopy(poissonNumSpikes, numBatch);

                // Simulate poisson population
                {
                    dim3 threads(BLOCK_SIZE, 1);
                    dim3 grid(preBlocks, numBatch);
                    poisson <<<grid, threads>>>(numPre, poissonMeanISI, d_poissonState, d_poissonTimeToSpike,
                                                poissonNumSpikes.second, poissonSpikes.second);
                }
            
                CHECK_CUDA_ERRORS(cudaEventRecord(kernelStartEvent));
                
                {
                    const unsigned int numPostSynapseBlocks = (unsigned int)std::ceil((float)numPost / (float)BLOCK_SIZE);

                    dim3 threads(BLOCK_SIZE, 1);
                    dim3 grid(numPostSynapseBlocks, numBatch);

                    if (mode == ModeGlobalAtomic) {
                        globalAtomic<<<grid, threads>>>(numPre, numPost, poissonNumSpikes.second, poissonSpikes.second,
                                                        weights.second, lambdaV.second, lambdaI.second,
                                                        outCurrents.second, gradients.second);
                    }
                    else if (mode == ModeGlobalAtomicLifted) {
                        globalAtomicLifted<<<grid, threads>>>(numPre, numPost, poissonNumSpikes.second, poissonSpikes.second,
                                                              weights.second, lambdaV.second, lambdaI.second,
                                                              outCurrents.second, gradients.second);
                    }
                    else if (mode == ModeSharedAtomic) {
                        sharedAtomic<<<grid, threads>>>(numPre, numPost, poissonNumSpikes.second, poissonSpikes.second,
                                                        weights.second, lambdaV.second, lambdaI.second,
                                                        outCurrents.second, gradients.second);
                    }
                    else if (mode == ModeWarpShuffle) {
                        warpReduction<<<grid, threads >>>(numPre, numPost, poissonNumSpikes.second, poissonSpikes.second,
                            weights.second, lambdaV.second, lambdaI.second,
                            outCurrents.second, gradients.second);
                    }
                    else if (mode == ModeWarpShuffleLifted) {
                        warpReductionLifted<< <grid, threads >> > (numPre, numPost, poissonNumSpikes.second, poissonSpikes.second,
                            weights.second, lambdaV.second, lambdaI.second,
                            outCurrents.second, gradients.second);
                    }
                }
                

                CHECK_CUDA_ERRORS(cudaEventRecord(kernelEndEvent));
                CHECK_CUDA_ERRORS(cudaEventSynchronize(kernelEndEvent));

                float tmp;
                CHECK_CUDA_ERRORS(cudaEventElapsedTime(&tmp, kernelStartEvent, kernelEndEvent));
                kernelTime += tmp;
            }
        }

        std::cout << "Kernel time:" << kernelTime << " ms" << std::endl;

        deviceToHostCopy(outCurrents, numOutCurrents);
        float meanCurrent = std::accumulate(&outCurrents.first[0], &outCurrents.first[numOutCurrents], 0.0f) / (float)numOutCurrents;;
        std::cout << "Mean current:" << meanCurrent << ", estimated mean current:" << numPost * poissonRate << std::endl;
    }
    catch(std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

