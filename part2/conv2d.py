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
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax = nl.tile_size.pmax  # 128
    c_out_pmax = nl.tile_size.pmax  # 128
    n_tiles_c_in = in_channels // c_in_pmax
    n_tiles_c_out = out_channels // c_out_pmax

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

    # Process the images in batches
    for b in nl.affine_range(batch_size):
        # TODO: Perform the convolution of X[b] with the weights W and bias b, followed by a maxpool
        # and store the result in X_out[b]

        # allocate space for one image in sbuf
        x = nl.ndarray(
            shape=(n_tiles_c_in, nl.par_dim(c_in_pmax), input_height, input_width),
            dtype=X.dtype,
            buffer=nl.sbuf,
        )

        # load in cur_image one tile at a time
        cur_image = X[
            b, 0:1
        ]  # .reshape([n_tiles_c_in, nl.par_dim(c_in_pmax), filter_height, filter_width])
        for tile in nl.affine_range(n_tiles_c_in):

            t_start = tile * c_in_pmax
            t_end = (tile + 1) * c_in_pmax
            # i = nl.arange(t_start, t_end)[:, None, None]
            x[tile] = nl.load(X[b, t_start:t_end])

        for n_tile_out in nl.affine_range(n_tiles_c_out):
            per_tile_out = nl.ndarray(
                shape=(nl.par_dim(c_out_pmax), out_height, out_width),
                dtype=X.dtype,
                buffer=nl.sbuf,
            )

            for row in nl.affine_range(out_height):
                row_out = nl.zeros(
                    shape=(c_out_pmax, out_width), dtype=X.dtype, buffer=nl.psum
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
                per_tile_out[:, row] = nl.copy(row_out, dtype=X_out.dtype)
            # copy each tile back to hbm
            X_out[b, n_tile_out * c_out_pmax : (n_tile_out + 1) * c_out_pmax] = nl.copy(
                per_tile_out, dtype=X_out.dtype
            )

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
bias = np.zeros(output_channels).astype(np.float32)

fused_conv2d_maxpool(X, W, bias)
