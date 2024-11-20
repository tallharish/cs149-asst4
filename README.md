# Assignment 4: Programming a Machine Learning Accelerator #

**Due Thurs Dec 5, 11:59pm**

**100 points total**

## Overview ##

In this assignment, you will learn how to implement and optimize kernels for the [AWS Trainium](https://aws.amazon.com/ai/machine-learning/trainium/) architecture, which features multiple tensor-oriented accelerated processing engines as well as software-managed on-chip storage that provides these engines high-bandwidth access to data. 

The assignment is organized into two parts.  In part 1 you will be given several implementations of a kernel that performs element-wise vector addition on Trainium. This is a simple application used to teach you about key concepts of using the Trainium architecture.  In part 2 you will implement a convolution layer for Trainium.

Overall, this assignment will:

1) Give you experience with the low-level details of tensor processing and managing on-chip SRAM on the accelerator.

2) Show the value of key locality-preserving optimizations like loop blocking and loop fusion.

## Environment Setup ##

You will be programming and testing your code on an AWS VM featuring Trainium accelerators. Please follow the instructions in [cloud_readme.md](cloud_readme.md) for setting up a machine to run the assignment.

Once you have logged in to your AWS machine, you should download the assignment starter code from the course Github using:

`git clone https://github.com/stanford-cs149/asst4-trainium`

After downloading the Assignment 4 repository, move to the `asst4-trainium` directory and **run the install script we have provided**:
```
cd asst4-trainium
source install.sh
```
The install script will activate a Python [virtual environment](https://builtin.com/data-science/python-virtual-environment) with all the needed assignment dependencies. It will also modify your `~/.bashrc` file so the virtual environment is activated upon future logins to your machine. Finally, the script sets up your InfluxDB credentials so that you may use `neuron-profile`.

## Part 0: Getting familiar with Trainium and Neuron Core Architecture

### Trainium Architecture Overview

First, let's get you acquainted with Trainium.

The `Trn1.2xlarge` instance used in this assignment features a single Trainium device, which comprises of two NeuronCores, as shown in image below. Each core is equipped with its own dedicated HBM (High-bandwidth memory). Each NeuronCore can be considered a standalone processing unit, which contains its own on-chip storage as well as a collection of specialized compute engines for performing 128x128 matrix operations (tensor engine), 128-wide vector operations (vector engine), etc. While each Trainium device has two NeuronCores, in this assignment we will be writing kernels that execute on a single NeuronCore.

<p align="center">
  <img src="https://github.com/stanford-cs149/asst4-trainium/blob/main/handout/trainium_chip.png" width=40% height=40%>
</p>

More details on the four distinct compute engines that exist in a NeuronCore can be found [here](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/neuron-core-v2.html#neuroncores-v2-arch).

### Trainium Memory Hierarchy

In Assignment 3, one of the key concepts was learning about the GPU memory hierarchy presented by CUDA, where there was main host memory, GPU device global memory, per-thread block shared memory, and private per-CUDA-thread memory.  On Trainium, the memory hierarchy consists of four levels: **host memory (DRAM)**, **device memory (HBM)**, and two fast on-chip memory types, **SBUF (State Buffer)** and **PSUM (Partial Sum Buffer)**. These levels are shown in the figure below.

<p align="center">
  <img src="https://github.com/stanford-cs149/asst4-trainium/blob/main/handout/memory_hierarchy.png" width=80% height=80%>
</p>

* __Host memory__, is the memory address space of the host machine, and is external to the Trainium device. You can think of host memory on Trainium as similar to host memory in a CUDA programming environment.
* __HBM__ is high-bandwidth memory located on the Trainium device. Moving data from host memory to HBM requires data transfer over the machine's PCIe interconnect. HBM serves as the device's primary memory, offering large storage (32 GiB). 
* __SBUF__ is on-chip storage on the NeuronCore. In comparison, SBUF is significantly smaller than HBM (24 MiB) but offers much higher bandwidth (~20x than HBM). 
* __PSUM__ is a small, specialized memory (2 MiB) dedicated to holding matrix multiplication results produced by the tensor engine.

In Trainium, all computations require loading data from HBM into SBUF, which is accessible by all engine types. Intermediate data generated during kernel execution by the compute engines is also stored in SBUF. Once the computation is complete, the results are written back to HBM. 

<p align="center">
  <img src="https://github.com/stanford-cs149/asst4-trainium/blob/main/handout/neuron_core.png" width=40% height=40%>
</p>

Recall that in a system that features a traditional data cache, decisions about what data from off-chip memories is replicated and stored in on-chip storage are made by the cache (based on cache organization and eviction policies). Software loads data at a given memory address, and the hardware is responsible for fetching that data from memory and managing what data is stored in the cache for efficient future access. In other words, from the perspective of software correctness, the cache does not exist--it is a hardware implementation detail. 

In contrast, the memories available to a NeuronCore are *software managed*. This means that a software must explicitly move data to and from these memories using data movement commands. Either the programmer must explicitly describe data movement in their program, or the NKI compiler must analyze the application and generate the appropriate data movement operations.  Some of the biggest challenges of efficiently using the NeuronCore architecture involve efficiently orchestrating the movement of data through the machine.

## Part 1: Learning the Neuron Kernel Interface with Vector Add (30 points)

In this section, we introduce the basics of the Trainium programming model by providing several different implementations of an application that adds the elements of two vectors.

The corresponding code is organized within the `/part1` directory. Specifically, the vector addition kernels discussed here can be found in `kernels.py`. Additionally, we provide a script, `vec_add.py`, which offers a convenient command-line interface for executing these kernels with different vector sizes. The script also includes an optional flag for collecting profiling metrics.

```
usage: python vec_add.py [-h] --kernel {naive,tiled,stream,alloc,alloc_bad} --size SIZE [--profile PROFILE_NAME]

options:
  -h, --help            show this help message and exit
  --kernel {naive,tiled,stream,alloc,alloc_bad}
  --size SIZE
  --profile PROFILE_NAME
                        Save profiler metrics into .neff & .ntff files with specified name.
                        Will overwrite a .neff / .ntff file with same name if it exists.
```

### NKI Programming Model:

The Neuron Kernel Interface (NKI) is a language and compiler for developing kernels that run on Trainium devices. NKI kernels are written in Python, and make use of three types of NKI operations:
1. **Loading data** from HBM to the on-chip SBUF.
2. **Computation** executed on the NeuronCore compute engines.
3. **Storing outputs** from SBUF back to HBM.

As an example, the following kernel defines how to perform vector addition using NKI. Note that the `@nki.jit` is a Python decorator that indicates that a function should be compiled to run on NeuronDevices, much like how the `__global__` function decorator in CUDA C++ designates that a function is to be compiled as a device-side function and run on the GPU.

Similar to how arguments to CUDA kernels are arrays in CUDA device global memory, the arguments to Python functions decorated with `@nki.jit`
 are tensors that reside in HBM accessible to the NeuronCore. In the following code, `a_vec` and `b_vec` are assumed to be length 128 vectors in HBM. (The code will not work for vectors that are larger than 128. We'll explain why shortly.)
```
@nki.jit
def vector_add_naive(a_vec, b_vec):
    
    # Allocate space for the output vector in HBM
    # it is the same dimensions as a_vec, with the same element type as a_vec
    out = nl.ndarray(shape=a_vec.shape, dtype=a_vec.dtype, buffer=nl.hbm)

    # Load the input vectors from HBM into variables stored in SBUF 
    a = nl.load(a_vec)
    b = nl.load(b_vec)

    # Add the input vectors
    res = nl.add(a, b)

    # Store the result into HBM
    nl.store(out, value=res)

    return out
```

In the code above...

- `nl.load` loads the two input vectors `a_vec` and `b_vec` from HBM to SBUF. Although it's not explicit in the code that `a` and `b` are variables backed by allocations in SBUF, you know this because they are variables returned by the `nl.load` function, which returns SBUF-based values.
- `nl.add` performs the vector addition of the two vectors.
- `nl.store` stores the results back to HBM.

<p align="center">
  <img src="https://github.com/stanford-cs149/asst4-trainium/blob/main/handout/sbuf_layout.png" width=60% height=60%>
</p>

**When looking at the code above, notice that NKI operations operate on tensors, not scalar values.** Specifically, the on-chip memories, SBUF and PSUM, store data that is arranged as 2D memory arrays. The first dimension of the 2D array is called the "partition dimension" `P`. The second dimension is referred to as the "free dimension" `F`.  NeuronCores are able load and process data along the partition dimension in parallel, *but the architecture also places a restriction that the size of the partition dimension is 128 or smaller.*  In other words, 
when loading a tensor from HBM to SBUF, the partition dimension of the tensor can be at most 128.  We will talk about the restrictions of the free dimension later.

As a result, in the code above, since `a_vec` and `b_vec` are 1D vectors, their only dimension is the partition dimension, and thus their size is limited to 128 elements.  In other words the code only works for vector sizes 128 or less.

### Step 1: Chunking Vectors to Parallelize Across 128 Compute Lanes

To fix the code to work for vectors with a size greater than 128, we need to load the vectors in chunks (subsets of the original tensor). 

```
@nki.jit
def vector_add_tiled(a_vec, b_vec):
    
    # Allocate space for the output vector in HBM
    out = nl.ndarray(shape=a_vec.shape, dtype=a_vec.dtype, buffer=nl.hbm)

    # Get the total number of vector rows
    M = a_vec.shape[0]
    
    # TODO: You should modify this variable for Step 1
    ROW_CHUNK = 1

    # Loop over the total number of chunks, we can use affine_range
    # because there are no loop-carried dependencies
    for m in nl.affine_range((M // ROW_CHUNK)):

        # Allocate row-chunk sized tiles for the input vectors
        a_tile = nl.ndarray((ROW_CHUNK, 1), dtype=a_vec.dtype, buffer=nl.sbuf)
        b_tile = nl.ndarray((ROW_CHUNK, 1), dtype=b_vec.dtype, buffer=nl.sbuf)
        
        # Load a chunk of rows
        a_tile[...] = nl.load(a_vec[m * ROW_CHUNK : (m + 1) * ROW_CHUNK])
        b_tile[...] = nl.load(b_vec[m * ROW_CHUNK : (m + 1) * ROW_CHUNK])

        # Add the row chunks together
        res = nl.add(a_tile, b_tile)

        # Store the result chunk into HBM
        nl.store(out[m * ROW_CHUNK : (m + 1) * ROW_CHUNK], value=res)
    
    return out
```

The above example breaks the vector rows into single-element chunks (the chunk size is 1 element of the vector---yes, this is inefficient, we'll come back to this in a second). This is achieved by indexing the vector using the standard Python slicing syntax `Tensor[Index:Index:...]`. More details regarding Tensor indexing in NKI can be found [here](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/programming_model.html#nki-tensor-indexing). 

In the code above [affine_range](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.language.affine_range.html) used here generates a sequence of numbers for loop iterators, similar to Python’s `range` function, but it requires that there are no loop-carried dependencies across iterations. Since there are no loop-carried dependencies across iterations, the NKI compiler can more aggressively optimize loop iterations to allow for increased pipelining across compute engines. In cases where loop iterations have dependencies, [sequential_range](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.language.sequential_range.html) should be used instead.

**What you need to do:**
1. Run the above `vector_add_tiled` implementation where *row_chunk = 1* with a vector size of 25600 (*this may take a couple minutes*). You may do so using the following command:

   ```
   python vec_add.py --kernel tiled --size 25600
   ```

   What was the execution time in microseconds (μs)?

3. Remember that the maximum partition size (number of rows) that can be loaded at once on a NeuronDevice is 128. Inside `kernels.py`, change `vector_add_tiled` so that it uses *row_chunk = 128*. Record the execution time in microseconds (μs) for `vector_add_tiled` with the *row_chunk = 128* operating on a vector size of 25600. How much faster is `vector_add_tiled` on a vector size of 25600 when *row_chunk = 128* compared to when *row_chunk = 1*? Why do you think it is faster?  (Hint: you should think of the execution as loading `ROW_CHUNK` elements in parallel from HBM and then performing a `ROW_CHUNK` wide vector add on the vectors in SBUF.)
	
4. Try running `vector_add_tiled` on vector sizes of 25600 when *row_chunk = 256*. You should see an error. In one sentence, explain why you get an error when trying to run *row_chunk = 256*.

### Step 2a: Improved Data Streaming

So far, we have been able to exploit the fact that the Vector Engine can perform computation with all 128 vector lanes in parallel, with each lane streaming a single element from/to a single SBUF/PSUM partition.

However, we can improve performance further by streaming more elements across the free dimension. The Neuron Compiler is responsible for converting each NKI load/store call to Direct Memory Access (DMA) transfer. You should think of a DMA transfer as a single asynchronous operation that moves a block of data from one storage location to another -- in this case between HBM and SBUF.  

The NeuronCore has 16 DMA engines that can all work on different data transfer operations in parallel. The caveat is that there is an overhead cost when setting up a DMA transfer and assigning DMA engines to work on them. In order to reduce this setup overhead, efficient implementations should aim to move a large amount of data in each DMA transfer to amortize DMA transfer overhead.

Although the first dimension (partition dimension) of a SBUF tensor can be no greater than 128, the second dimension for a single SBUF vector instruction can be up to 64K elements. This means that it is possible to use a single instruction to load 128 * 64k = 8192k elements from HBM to SBUF. Furthermore, we can perform vector addition on two 8192k element SBUF tiles in a single arithmetic instruction. Therefore, rather than performing a `nl.load` operations (and equivalently a hardware DMA transfer request) for each 128 element chunk of a vector, we should should instead move many 128-row chunks with each DMA transfer request. This streamlined approach allows us to amortize the setup time required for transferring data.

In order to improve DMA transfer overhead, we will need to reshape our vectors so they are two-dimensional tiles, rather than linearized arrays. In Assignment 3, we worked with CUDA thread blocks partitioned across an entire image, and in order to map CUDA threads to image pixels we flattened our grid by calculating a thread’s global linear index. You can think about the reshaping process for the NeuronCore as the inverse: the goal is to turn a single-dimension vector into a dense 2D matrix. NumPy comes with a built-in [reshape function](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html) allowing you to reshape arrays into the shape of your choosing. 

<p align="center">
  <img src="https://github.com/stanford-cs149/asst4-trainium/blob/main/handout/non_reshaped_DMA.png" width=48% height=48%>
  <img src="https://github.com/stanford-cs149/asst4-trainium/blob/main/handout/reshaped_DMA.png" width=48% height=48%>
</p>


Take a look at `vector_add_stream`, which extends `vector_add_tiled` so that there are less DMA transfers:
```
@nki.jit
def vector_add_stream(a_vec, b_vec):

    # Get the total number of vector rows
    M = a_vec.shape[0]

    # TODO: You should modify this variable for Step 1
    FREE_DIM = 2

    # The maximum size of our Partition Dimension
    PARTITION_DIM = 128

    # The total size of each tile
    TILE_M = PARTITION_DIM * FREE_DIM

    # Reshape the the input vectors
    a_vec_re = a_vec.reshape((M // TILE_M, PARTITION_DIM, FREE_DIM))
    b_vec_re = b_vec.reshape((M // TILE_M, PARTITION_DIM, FREE_DIM))

    # Allocate space for the reshaped output vector in HBM
    out = nl.ndarray(shape=a_vec_re.shape, dtype=a_vec_re.dtype, buffer=nl.hbm)

    # Loop over the total number of tiles
    for m in nl.affine_range((M // TILE_M)):

        # Allocate space for a reshaped tile
        a_tile = nl.ndarray((PARTITION_DIM, FREE_DIM), dtype=a_vec.dtype, buffer=nl.sbuf)
        b_tile = nl.ndarray((PARTITION_DIM, FREE_DIM), dtype=a_vec.dtype, buffer=nl.sbuf)

        # Load the input tiles
        a_tile = nl.load(a_vec_re[m])
        b_tile = nl.load(b_vec_re[m])

        # Add the tiles together
        res = nl.add(a_tile, b_tile)

        # Store the result tile into HBM
        nl.store(out[m], value=res)

    # Reshape the output vector into its original shape
    out = out.reshape((M,))

    return out
```

**What you need to do:**
1. Run the above `vector_add_stream` implementation where *FREE_DIM = 2*. How many microseconds (μs) did it take to run for a vector size of 25600? How much faster is this compared to `vector_add_tiled` with *row_chunk = 128* from Step 1?
2. The current `vector_add_stream` implementation reduces the number of DMA transfers slightly, but the number of DMA transfers can be reduced further. Inside `kernels.py`, change the value of *FREE_DIM* for `vector_add_stream` to reduce the number of DMA transfers as much as possible on a vector of size 25600.

   What value of *FREE_DIM* did you choose? What was the execution time in microseconds (μs) on a vector size of 25600 for this value of *FREE_DIM*. How much faster is `vector_add_stream` with the *FREE_DIM* number you chose than `vector_add_stream` with *FREE_DIM = 2*? How much faster is `vector_add_stream` with the *FREE_DIM* number you chose than `vector_add_tiled` with *row_chunk = 128*?

### Step 2b: Learning to Use Neuron-Profile

There is a trade-off in choosing a tile's free dimension size:
1. Too small of a tile size exposes significant instruction overhead leading to inefficient engine execution.
2. Too large of a tile size often leads to inefficient pipelining between engines and high memory pressure in SBUF in cases of data reuse ("memory pressure" means that SBUF may fill up).

Currently, we have explored the benefits of increasing tile sizes to their maximum amount in order for us to amortize instruction overhead and DMA transfer setup / teardown. Now, we will explore why making the free dimension as large as possible is not always the best solution.

For this task, you will need to use the profiling tool for NeuronDevices: `neuron-profile`, which can provide detailed analysis of the performance of an application running on a NeuronCore. In order to run the profiling tool, you must make sure that you ran the install script as detailed in [Environment Setup](https://github.com/stanford-cs149/asst4-trainium/tree/main?tab=readme-ov-file#environment-setup) and that you forwarded ports 3001 and 8086 when you ssh'd into your machine. To reiterate on the latter, the command you should have ran is:

 `ssh -i path/to/key_name.pem ubuntu@<public_dns_name> -L 3001:localhost:3001 -L 8086:localhost:8086`
 
 More details about why this is needed can be found in the [cloud_readme.md](https://github.com/stanford-cs149/asst4-trainium/blob/main/cloud_readme.md).

**What you need to do:**
1.  This time, we are going to increase the vector sizes by a factor of 10 so that instead of adding 25600 elements we will be adding 256000 elements. This will allow us to see trade offs that comes from dealing with tile sizes that are too large.  

    First, set *FREE_DIM = 2000* in `vector_add_stream`. Now, just like the prior steps we are going to execute our kernel, but this time we are going to save the compiled kernel 
    into a **.neff** file and the kernel execution trace into a **.ntff** trace file. Let's run `vector_add_stream` on a vector_size of 256000 and save the compiled kernel and 
    trace into files prefixed with `stream_256k_fd2k` with the following command:

    ```
    python vec_add.py --kernel stream --size 256000 --profile stream_256k_fd2k
    ```

    You should have generated two files: ***stream_256k_fd2k.neff*** and ***stream_256k_fd2k.ntff***.
    
    Now, using a similar workflow run `vector_add_stream` with *FREE_DIM = 1000* on a vector_size of 256000 and save the compiled kernel and trace into files prefixed with 
    `stream_256k_fd1k`.
2.  These generated files will allow us to collect kernel execution metrics using the `neuron-profile` tool. These profiling metrics will be very useful for analyzing the 
    performance of your kernels. Let's look at a brief summary of execution metrics for `vector_add_stream` with *FREE_DIM = 2000* by running the following command:

    ```
    neuron-profile view --output-format summary-text -n stream_256k_fd2k.neff -s stream_256k_fd2k.ntff
    ```

    You will see a summarized output consisting of various execution metrics in alphabetical order. Let's look at two specific metrics: 
    
     * **dma_transfer_count**: The number of DMA transfers
     * **total_time**: Kernel execution time in seconds

    What was kernel execution time in seconds when *FREE_DIM = 2000*? How many DMA transfers were made when *FREE_DIM = 2000*?
    
    Using the same workflow as before, look at the summary of execution metrics when *FREE_DIM = 1000*.
    
    What was kernel execution time in seconds when *FREE_DIM = 1000*? How many DMA transfers were made when *FREE_DIM = 1000*?

3. Although the kernel with *FREE_DIM = 1000* had more DMA transfers, it was faster! Let's analyze why.
  
   We can dive deeper into the kernel execution metrics using the GUI functionality of `neuron-profile`. Let's launch the GUI for `vector_add_stream` with *FREE_DIM = 2000* by 
   running the following command:

   ```
   neuron-profile view -n stream_256k_fd2k.neff -s stream_256k_fd2k.ntff
   ```

   After running the command, you will see an output like the following:

   `View profile at http://localhost:3001/profile/...`

   Paste this *http* link into a browser of your choice to view more in-depth profiler analytics. **Note:** You will only be able to view this if you have correctly forwarded 
   ports 3001 and 8086 when you ssh'd into your machine.

   ![Profiler GUI Example](https://github.com/stanford-cs149/asst4-trainium/blob/main/handout/profiler_GUI.gif)

   Hover over various events in the graph generated from the profiler. See the above GIF as an example for `vector_add_stream` with *FREE_DIM = 2000* on a vector size of 256000. 
   In the example we see the following:
   
   * **qGpSimdIO0**: Shows DMA transfer for storing the output vector
   * **qSyncIO0**: Shows DMA transfers for loading the two input vectors
   * **VectorE**: Shows the Vector Engine instruction for add the two input vectors
   * **Pending DMA Count**: Shows the number of pending DMA transfers over time
   * **DMA Throughput**: Shows the device bandwidth over time

   Now, in your terminal press `ctrl-c` to exit the current `neuron-profile view`. Note that you can still view the GUI analytics for `vector_add_stream` with *FREE_DIM = 2000* in 
   your browser as they have been temporally stored in a database. Following the same workflow, launch the GUI analytics for `vector_add_stream` with *FREE_DIM = 1000*.

4. After analyzing the GUI analytics graph for `vector_add_stream` with both *FREE_DIM = 2000* and *FREE_DIM = 1000*, briefly explain why FREE_DIM = 1000 has a faster execution 
   time than FREE_DIM = 2000 even though it required more DMA transfers (*Hint:* pipelining).

   Feel free to also play around with various functionalities in the `neuron-profile` GUI. Some particularly useful things are in the `View Settings` tab located at the bottom 
   toolbar. In the `View Settings` tab you can toggle `Group DMA by engine` and `Show expanded DMA` to have the graph show the DMA transfers from each of the NeuronCore's 16 DMA 
   engines. You may also want to look at the `Summary` tab located at the bottom toolbar. This tab displays the same brief summary of execution metrics we saw when running  
   `neuron-profile view --output-format summary-text ...` in Question 2. Feel free to learn more about `neuron-profile` functionality from the [user guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-profile-user-guide.html) and interesting performance metrics for NKI kernels from the [NKI performance guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/nki_perf_guide.html).

### Step 3: Direct Allocation

As discussed above, a multi-dimensional NKI tensor requires one of its dimensions to be mapped to the partition dimension (`P`)  of physical memory. We also define an NKI Tile as any NKI Tensor with the first dimension being the partition dimension. 

To introduce NKI allocation API, let’s define the block (`B`) dimension as any dimension before the partition dimension in a NKI Tensor. Therefore, a NKI Tensor has three types of dimensions: `(B, P, F)`. Note, a NKI Tensor can have one or more dimensions in both `B` and `F`, but there can only be one dimension in `P` due to Neuron ISA requirements. The block dimension effectively describes how many `(P, F)` NKI Tiles the tensor has, which commonly corresponds to how many NKI API compute API invocations we need to process the entire tensor.

In fact, the block dimensions are considered **logical** in NKI - the number of `(P, F)` NKI Tiles suggested by dimension (`B`) may not match the number of physical tiles allocated. The Neuron Compiler analyzes the code and uses a heuristic-driven allocator to determine the number of physical tiles allocated to each tensor. Given the amount of parallelism available in hardware and the complex parallel programs seen in common machine learning workloads, the heuristic-based memory allocator in Neuron Compiler may not yield the optimal allocation decisions. Bad allocation decisions typically lead to sub-optimal engine parallelism and/or on-chip memory over-subscription causing excessive spills of intermediate data to device memory. 

For example, consider a scenario where the block dimension size of `nki_tensor` is 2, yet only a single physical tile, `T`, is allocated in `sbuf`.
```
   i_block = 0
1. nl.load(nki_tensor[0, :, :]) => write T
2. nl.exp(nki_tensor[0, :, :])  => read T

   i_block = 1
3. nl.load(nki_tensor[1, :, :]) => write T
4. nl.exp(nki_tensor[1, :, :])  => read T
```

In this example, there is only one logical tile in `nki_tensor` that is alive at one time. However, in NeuronCore, `nl.load` and `nl.exp` are executed using two independent resources: DMA Engine and Scalar Engine. In this serialized execution, there is no instruction parallelism achieved between these engines. There is potential for performance optimization by allocating more physical tiles.

With [NKI direct allocation API](https://awsdocs-neuron-staging.readthedocs-hosted.com/_/sharing/PsLqSocIDcWDsjcHd3ql7JCqOhou3AKO?next=/en/nki_docs_2.21_beta_class/general/nki/nki_direct_allocation_guide.html#nki-direct-allocation-guide), programmers can bypass the compiler allocator and take full control of memory allocation in SBUF/PSUM for NKI Tensors. NKI provides two kinds of direct allocation API, `nisa.sbuf.alloc()` and `nisa.sbuf_mod_alloc()`. The first one requires programmers to define an allocation algorithm from scratch, while the second one invokes a pre-defined modulo allocation scheme in Neuron Compiler. In this tutorial, we will focus on the `nisa.sbuf.mod_alloc()` API, which is already powerful enough.
Modulo allocation works as follows. Suppose that we allocate two physical tiles for a tensor with a logical shape of `(8, par_dim(128), 512)`. The eight logical tiles are assigned to the two physical tiles by taking a modulo of two on the logical tile index. Therefore, logical tiles with index (0, ), (2, ), (4, ), (6, )share the same physical tile, while logical tiles (1, ), (3, ), (5, ), (7, ) share the other physical tile.  

The `nisa.sbuf.mod_alloc` API takes four input parameters:
1. `base_addr` indicates the starting byte offset within each SBUF partition of the physical tiles.
2. `base_partition` indicates the starting SBUF partition of the physical tiles.
3. `num_par_tiles` indicates the number of physical tiles to be allocated along the partition dimension of SBUF. This is only applicable for tiles that use fewer than 64 partitions per ISA constraints.
4. `num_free_tiles` indicates the number of physical tiles to be allocated along the free dimension of SBUF.

Take a look at `vector_add_direct_allocation`, which uses `nisa.sbuf.mod_alloc` to allocate physical tiles for input and output tensors.

```
@nki.jit
def vector_add_direct_allocation(a_vec, b_vec):
    # Create output tensor in HBM 
    out = nl.ndarray(a_vec.shape, dtype=a_vec.dtype, buffer=nl.shared_hbm)
    
    # Define constants for free dimension, physical tile count, and partition dimension
    FREE_DIM = 1000
    FREE_DIM_TILES = 4
    PARTITION_DIM = 128

    # Get the total number of vector rows
    M = a_vec.shape[0]

    # Define the size of each tile
    TILE_M = PARTITION_DIM * FREE_DIM

    # Reshape the the input vectors
    a_vec = a_vec.reshape((M // TILE_M, PARTITION_DIM, FREE_DIM))
    b_vec = b_vec.reshape((M // TILE_M, PARTITION_DIM, FREE_DIM))
    out = out.reshape((M // TILE_M, PARTITION_DIM, FREE_DIM))

    # Get the total number of tiles
    N_TILES = M // TILE_M

    # Initialize the starting byte offset for a_tensor
    current_offset = 0
    
    # Allocate space for the entire reshaped a_tensor with 4 physical tiles
    a_tile = nl.ndarray((N_TILES, nl.par_dim(PARTITION_DIM), FREE_DIM), dtype=a_vec.dtype,
            buffer=ncc.sbuf.mod_alloc(base_addr=current_offset, num_free_tiles=(FREE_DIM_TILES,)))

    # Increment the starting byte offset for b_tensor based on tile and feature size
    current_offset += FREE_DIM_TILES * FREE_DIM * 4
    
    # Allocate space for the entire reshaped b_tensor with 4 physical tiles
    b_tile = nl.ndarray((N_TILES, nl.par_dim(PARTITION_DIM), FREE_DIM), dtype=b_vec.dtype,
            buffer=ncc.sbuf.mod_alloc(base_addr=current_offset, num_free_tiles=(FREE_DIM_TILES,)))

    # Increment the starting byte offset, and allocate space for the entire reshaped out_tensor
    current_offset += FREE_DIM_TILES * FREE_DIM * 4
    res_tile = nl.ndarray((N_TILES, nl.par_dim(PARTITION_DIM), FREE_DIM), dtype=out.dtype,
            buffer=ncc.sbuf.mod_alloc(base_addr=current_offset, num_free_tiles=(FREE_DIM_TILES,)))

    # Loop over the total number of tiles
    for m in nl.affine_range(N_TILES):
        
        # Load the input tiles
        a_tile[m] = nl.load(a_vec[m])
        b_tile[m] = nl.load(b_vec[m])

        # Add the tiles together
        res_tile[m] = nisa.tensor_tensor(a_tile[m], b_tile[m], op=nl.add)
        
        # Store the result tile into HBM
        nl.store(out[m], value = res_tile[m])

    # Reshape the output vector into its original shape
    out = out.reshape((M, 1))

    return out
```

**What you need to do:**
1. Run the above `vector_add_direct_allocation` implementation on a vector size of 256000 and record the execution time in microseconds (μs). Now run `vector_add_stream` from Step 2 with *FREE_DIM=1000* on a vector size of 256000 and record the execution time in microseconds (μs).
   
    You should notice that the execution times are the same. This is because `vector_add_direct_allocation` manually performs the allocations that the Neuron compiler 
    automatically performs for `vector_add_stream`. Having the ability to perform manual allocations does not yield much benefit when performing vector addition as there is no 
    data reuse and no need for engine paralelism. However, in other scenarios you will find that having more fine-grained control of memory allocations allows you to keep data on- 
    chip for as long as possible, as well as giving you the ability to perform different computations in parallel across multiple engines.
2. How many physical tiles are allocated for each tensor? What is the problem if we set the number of physical tiles too large?
3. Why should we have different offsets for each tensor? Try to run `vector_add_direct_allocation_bad` on a vector size of 256000. What is the result? Please provide a possible explanation.

## Part 2: Implementing a Convolution Layer (70 points)

In the second part of the assignment you will implement a convolution layer on the NeuronCore. Part two of the assignment will be released after the midterm.

## Hand-in Instructions

Please submit your work using Gradescope. If you are working with a partner please remember to tag your partner on gradescope.

Please submit your writeup as the file `writeup.pdf`.

Details on final code submission to follow.

