import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A fused convolution - maxpool kernel that you need to implement for Part 2.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.
    pool_size: the size of the pool filter and pool stride.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: pool_size == 1 || pool_size == 2
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height // pool_size
out_pool_width = out_width // pool_size

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""


@nki.jit
def fused_conv2d_maxpool(X, W, bias, pool_size=1):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]
    bias = bias.reshape([bias.shape[0] // nl.tile_size.pmax, nl.tile_size.pmax, 1])
    print(f"bias.shape {bias.shape}")
    print(f"X.dtype {X.dtype}")
    print(f"W.dtype {W.dtype}")
    print(f"bias.dtype {bias.dtype}")

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size

    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.zeros(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax = nl.tile_size.pmax  # 128
    c_out_pmax = nl.tile_size.pmax  # 128
    n_tiles_c_in = in_channels // c_in_pmax
    n_tiles_c_out = out_channels // c_out_pmax

    out_chunk_size = 2
    in_chunk_size = out_chunk_size + filter_height - 1
    n_out_chunks = out_height // 2  # assume out_height even

    # allocate and load in weights tensor to sbuf
    W_sbuf = nl.ndarray(
        shape=(
            n_tiles_c_out,
            nl.par_dim(c_out_pmax),
            n_tiles_c_in,
            c_in_pmax,
            filter_height,
            filter_width,
        ),
        dtype=W.dtype,
        buffer=nl.sbuf,
    )

    W = W.reshape(
        [
            n_tiles_c_out,
            c_out_pmax,
            n_tiles_c_in,
            c_in_pmax,
            filter_height,
            filter_width,
        ]
    )
    for tile_c_out in nl.affine_range(n_tiles_c_out):
        W_sbuf[tile_c_out] = nl.load(W[tile_c_out])

    w = nl.ndarray(
        shape=(
            filter_height,
            filter_width,
            n_tiles_c_out,
            n_tiles_c_in,
            nl.par_dim(c_in_pmax),
            c_out_pmax,
        ),
        dtype=W.dtype,
        buffer=nl.sbuf,
    )
    for h in nl.affine_range(filter_height):
        for wi in nl.affine_range(filter_width):
            for tile_c_out in nl.affine_range(n_tiles_c_out):
                for tile_c_in in nl.affine_range(n_tiles_c_in):
                    w[h, wi, tile_c_out, tile_c_in, :, :] = nl.copy(
                        W_sbuf[tile_c_out, :, tile_c_in, :, h, wi]
                    )
                    w[h, wi, tile_c_out, tile_c_in, :, :] = nl.transpose(
                        w[h, wi, tile_c_out, tile_c_in, :, :]
                    )

    # Pool index patterns
    i_0 = nl.arange(c_out_pmax)[:, None, None, None, None]
    i_1 = nl.arange(out_chunk_size // pool_size)[None, :, None, None, None]  # y_outer
    i_2 = nl.arange(pool_size)[None, None, :, None, None]  # y_inner
    i_3 = nl.arange(out_pool_width)[None, None, None, :, None]  # x_outer
    i_4 = nl.arange(pool_size)[None, None, None, None, :]  # x_inner

    # Process the images in batches
    for b in nl.affine_range(batch_size):
        # TODO: Perform the convolution of X[b] with the weights W and bias b, followed by a maxpool
        # and store the result in X_out[b]

        # allocate space for one chunk of an image in sbuf
        for chunk in nl.affine_range(n_out_chunks):
            # correct for divisibility issues for the last chunk
            if chunk == n_out_chunks - 1:
                cur_out_chunk_size = out_height % n_out_chunks
                cur_in_chunk_size = out_chunk_size + filter_height - 1
            else:
                cur_out_chunk_size = out_chunk_size
                cur_in_chunk_size = in_chunk_size
            # for output rows [chunk * out_chunk_size, (chunk + 1) * out_chunk_size)
            # so, need input rows [chunk * out_chunk_size, chunk * out_chunk_size + in_chunk_size]
            x = nl.ndarray(
                # shape=(n_tiles_c_in, nl.par_dim(c_in_pmax), input_height, input_width),
                shape=(
                    n_tiles_c_in,
                    nl.par_dim(c_in_pmax),
                    cur_in_chunk_size,
                    input_width,
                ),
                dtype=X.dtype,
                buffer=nl.sbuf,
            )

            # load the chunk of the image one tile at a time

            for tile in nl.affine_range(n_tiles_c_in):

                t_start = tile * c_in_pmax
                t_end = (tile + 1) * c_in_pmax

                # x[tile] = nl.load(X[b, t_start: t_end])
                x[tile] = nl.load(
                    X[
                        b,
                        t_start:t_end,
                        chunk
                        * out_chunk_size : min(
                            input_height, chunk * out_chunk_size + in_chunk_size
                        ),
                    ]
                )

            for n_tile_out in nl.affine_range(n_tiles_c_out):
                per_tile_out = nl.ndarray(
                    # shape=(nl.par_dim(c_out_pmax), out_height, out_width),
                    shape=(nl.par_dim(c_out_pmax), cur_out_chunk_size, out_width),
                    dtype=X.dtype,
                    buffer=nl.sbuf,
                )
                print(f"per_tile_out.shape: {per_tile_out.shape}")

                bias_tile = nl.ndarray(
                    shape=(n_tiles_c_out, nl.par_dim(c_out_pmax), 1),
                    dtype=bias.dtype,
                    buffer=nl.sbuf,
                )

                bias_tile[n_tile_out] = nl.load(bias[n_tile_out])
                print(f"bias_tile.shape: {bias_tile.shape}")

                # for row in nl.affine_range(out_height):
                for row in nl.affine_range(cur_out_chunk_size):
                    row_out = nl.zeros(
                        shape=(nl.par_dim(c_out_pmax), out_width),
                        dtype=nl.float32,
                        buffer=nl.psum,
                    )

                    for h in nl.affine_range(filter_height):
                        for wi in nl.affine_range(filter_width):
                            for n_tile_in in nl.affine_range(n_tiles_c_in):
                                row_out += nl.matmul(
                                    w[h, wi, n_tile_out, n_tile_in, :, :],
                                    x[n_tile_in, :, row + h, wi : wi + out_width],
                                    transpose_x=True,
                                )
                                # row_out = nl.add(row_out, row_out_cur)
                    # copy each row's output back to tile in sbuf
                    print(f"row_out.shape: {row_out.shape}")

                    # per_tile_out[:, row] = nl.copy(row_out_tmp, dtype=X_out.dtype) # Old Style
                    per_tile_out[:, row] = nisa.tensor_scalar(
                        row_out,
                        op0=nl.add,
                        operand0=bias_tile[n_tile_out],
                    )

                print(f"per_tile_out.shape: {per_tile_out.shape}")
                print(
                    f"X_out.shape: {X_out[b,n_tile_out * c_out_pmax : (n_tile_out + 1) * c_out_pmax, chunk * out_chunk_size : min(out_height, (chunk + 1) * out_chunk_size),].shape}"
                )

                # per_tile_out -> shape (128, 2, 222)
                out_tile = nl.max(
                    per_tile_out[i_0, pool_size * i_1 + i_2, pool_size * i_3 + i_4], # shape -> (128, 2, 2, 111, 2)
                    axis=[2, 4],
                )
                print(f"out_tile.shape: {out_tile.shape}")

                # MaxPool on per_tile_out to generate per_tile_out_maxPool
                # copy each tile back to hbm
                nl.store(
                    X_out[
                        b,
                        n_tile_out * c_out_pmax : (n_tile_out + 1) * c_out_pmax,
                        chunk
                        * (out_chunk_size // pool_size): min(
                            out_pool_height, (chunk + 1) * (out_chunk_size // pool_size)
                        ),
                    ],
                    out_tile,
                )
                # X_out[b, n_tile_out * c_out_pmax: (n_tile_out + 1) * c_out_pmax] = nl.copy(per_tile_out, dtype=X_out.dtype)

    return X_out


input_channels = 128
output_channels = 128
kernel_size = 3
batch_size = 4
image_dims = (32, 16)

X = np.random.rand(batch_size, input_channels, image_dims[0], image_dims[1]).astype(
    np.float32
)
W = np.random.rand(output_channels, input_channels, kernel_size, kernel_size).astype(
    np.float32
)
bias = np.random.rand(output_channels).astype(np.float32)

fused_conv2d_maxpool(X, W, bias)
