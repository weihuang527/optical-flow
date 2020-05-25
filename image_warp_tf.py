import tensorflow as tf
import numpy as np

def image_warp(im, flow):
    """Performs a backward warp of an image using the predicted flow.

    Args:
        im: Batch of images. [num_batch, height, width, channels]
        flow: Batch of flow vectors. [num_batch, height, width, 2]
    Returns:
        warped: transformed image of the same shape as the input image.
    """
    with tf.variable_scope('image_warp'):

        num_batch, height, width, channels = im.shape.as_list()
        max_x = tf.cast(width - 1, 'int32') # 数据格式转换

        max_y = tf.cast(height - 1, 'int32')
        zero = tf.zeros([], dtype='int32')

        # We have to flatten our tensors to vectorize the interpolation
        im_flat = tf.reshape(im, [-1, channels])
        flow_flat = tf.reshape(flow, [-1, 2])

        # Floor the flow, as the final indices are integers
        # The fractional part is used to control the bilinear interpolation.
        flow_floor = tf.to_int32(tf.floor(flow_flat))  #向下取整 并转换数据格式
        bilinear_weights = flow_flat - tf.floor(flow_flat) #偏移量

        # Construct base indices which are displaced with the flow
        pos_x = tf.tile(tf.range(width), [height * num_batch]) #数组复制
        grid_y = tf.tile(tf.expand_dims(tf.range(height), 1), [1, width])
        pos_y = tf.tile(tf.reshape(grid_y, [-1]), [num_batch])

        x = flow_floor[:, 0]
        y = flow_floor[:, 1]
        xw = bilinear_weights[:, 0]
        yw = bilinear_weights[:, 1]

        # Compute interpolation weights for 4 adjacent pixels
        # expand to num_batch * height * width x 1 for broadcasting in add_n below
        wa = tf.expand_dims((1 - xw) * (1 - yw), 1) # top left pixel
        wb = tf.expand_dims((1 - xw) * yw, 1) # bottom left pixel
        wc = tf.expand_dims(xw * (1 - yw), 1) # top right pixel
        wd = tf.expand_dims(xw * yw, 1) # bottom right pixel

        x0 = pos_x + x
        x1 = x0 + 1
        y0 = pos_y + y
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, zero, max_x) #使数组在min和max之间
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        dim1 = width * height
        batch_offsets = tf.range(num_batch) * dim1
        base_grid = tf.tile(tf.expand_dims(batch_offsets, 1), [1, dim1])
        base = tf.reshape(base_grid, [-1])

        base_y0 = base + y0 * width
        base_y1 = base + y1 * width
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        Ia = tf.gather(im_flat, idx_a) #用一个一维的索引数组，将张量中对应索引的向量提取出来
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)

        warped_flat = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id]) #数组相加
        warped = tf.reshape(warped_flat, [num_batch, height, width, channels])

        return warped

def deform(img, flow):
    height, width = img.shape
    if img.ndim == 2:
        img_dims = (1, height, width, 1)
        img = img[np.newaxis, :, :, np.newaxis]
    else:
        img_dims = (1, height, width, 3)
        img = img[np.newaxis, :, :]
    flow_dims = (1, height, width, 2)
    image_moving = tf.placeholder(tf.float32, shape=img_dims)
    flow_moving = tf.placeholder(tf.float32, shape=flow_dims)
    image_warped = image_warp(image_moving, flow_moving)
    flow = flow[np.newaxis, :, :, :]
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=sess_config) as sess:
        warped_label = sess.run(image_warped, 
                        feed_dict={image_moving: img,
                                    flow_moving: flow})
    warped_label = np.squeeze(warped_label).astype(np.uint8)
    return warped_label

def gen_flow_circle(center, height, width):
    x0, y0 = center
    if x0 >= height or y0 >= width:
        raise AttributeError('ERROR')
    flow = np.zeros((height, width, 2), dtype=np.float32)

    grid_x = np.tile(np.expand_dims(np.arange(width), 0), [height, 1])
    grid_y = np.tile(np.expand_dims(np.arange(height), 1), [1, width])

    grid_x0 = np.tile(np.array([x0]), [height, width])
    grid_y0 = np.tile(np.array([y0]), [height, width])

    flow[:,:,0] = grid_x0 - grid_x
    flow[:,:,1] = grid_y0 - grid_y

    return flow

if __name__ == "__main__":
    import cv2
    from PIL import Image
    case = 2  # 1 or 2
    height = 101
    width = 101
    if case == 1:
        flow = gen_flow_circle([50,50], height, width)
        flow = flow / 3
    else:
        flow = gen_flow_circle([0,0], height, width)
        flow = flow / 5
    
    img = np.zeros((height, width), dtype=np.uint8)
    img = cv2.circle(img, (width//2, height//2), radius=25, color=255, thickness=1)

    deformed_bilinear = deform(img, flow)

    img_cat = np.concatenate([img, deformed_bilinear], axis=1)
    _, w = img_cat.shape
    img_cat[:, width] = 255
    Image.fromarray(img_cat).save('./deformed.png')