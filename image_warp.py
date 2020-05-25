import numpy as np

def image_warp(im, flow, mode='bilinear'):
    """Performs a backward warp of an image using the predicted flow.
    numpy version

    Args:
        im: input image. ndim=2, 3 or 4, [[num_batch], height, width, [channels]]. num_batch and channels are optional, default is 1.
        flow: flow vectors. ndim=3 or 4, [[num_batch], height, width, 2]. num_batch is optional
        mode: interpolation mode. 'nearest' or 'bilinear'
    Returns:
        warped: transformed image of the same shape as the input image.
    """
    # assert im.ndim == flow.ndim, 'The dimension of im and flow must be equal '
    flag = 4
    if im.ndim == 2:
        height, width = im.shape
        num_batch = 1
        channels = 1
        im = im[np.newaxis, :, :, np.newaxis]
        flow = flow[np.newaxis, :, :]
        flag = 2
    elif im.ndim == 3:
        height, width, channels = im.shape
        num_batch = 1
        im = im[np.newaxis, :, :]
        flow = flow[np.newaxis, :, :]
        flag = 3
    elif im.ndim == 4:
        num_batch, height, width, channels = im.shape
        flag = 4
    else:
        raise AttributeError('The dimension of im must be 2, 3 or 4')

    max_x = width - 1
    max_y = height - 1
    zero = 0

    # We have to flatten our tensors to vectorize the interpolation
    im_flat = np.reshape(im, [-1, channels])
    flow_flat = np.reshape(flow, [-1, 2])

    # Floor the flow, as the final indices are integers
    flow_floor = np.floor(flow_flat).astype(np.int32)

    # Construct base indices which are displaced with the flow
    pos_x = np.tile(np.arange(width), [height * num_batch])
    grid_y = np.tile(np.expand_dims(np.arange(height), 1), [1, width])
    pos_y = np.tile(np.reshape(grid_y, [-1]), [num_batch])

    x = flow_floor[:, 0]
    y = flow_floor[:, 1]

    x0 = pos_x + x
    y0 = pos_y + y

    x0 = np.clip(x0, zero, max_x)
    y0 = np.clip(y0, zero, max_y)

    dim1 = width * height
    batch_offsets = np.arange(num_batch) * dim1
    base_grid = np.tile(np.expand_dims(batch_offsets, 1), [1, dim1])
    base = np.reshape(base_grid, [-1])

    base_y0 = base + y0 * width

    if mode == 'nearest':
        idx_a = base_y0 + x0
        warped_flat = im_flat[idx_a]
    elif mode == 'bilinear':
        # The fractional part is used to control the bilinear interpolation.
        bilinear_weights = flow_flat - np.floor(flow_flat)

        xw = bilinear_weights[:, 0]
        yw = bilinear_weights[:, 1]

        # Compute interpolation weights for 4 adjacent pixels
        # expand to num_batch * height * width x 1 for broadcasting in add_n below
        wa = np.expand_dims((1 - xw) * (1 - yw), 1) # top left pixel
        wb = np.expand_dims((1 - xw) * yw, 1) # bottom left pixel
        wc = np.expand_dims(xw * (1 - yw), 1) # top right pixel
        wd = np.expand_dims(xw * yw, 1) # bottom right pixel

        x1 = x0 + 1
        y1 = y0 + 1

        x1 = np.clip(x1, zero, max_x)
        y1 = np.clip(y1, zero, max_y)

        base_y1 = base + y1 * width
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        Ia = im_flat[idx_a]
        Ib = im_flat[idx_b]
        Ic = im_flat[idx_c]
        Id = im_flat[idx_d]

        warped_flat = wa * Ia + wb * Ib + wc * Ic + wd * Id
    warped = np.reshape(warped_flat, [num_batch, height, width, channels])

    if flag == 2:
        warped = np.squeeze(warped)
    elif flag == 3:
        warped = np.squeeze(warped, axis=0)
    else:
        pass
    warped = warped.astype(np.uint8)

    return warped