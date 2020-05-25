import numpy as np 

def function_2(x):
    if x.ndim == 1:
        return np.sum(x **2)
    else:
        return np.sum(x**2,axis=1)

def _numerical_gradient_no_batch(f,x):
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_va1 = x[idx]
        x[idx] = float(tmp_va1) + h
        fxh1 =f(x)

        x[idx] = tmp_va1 -h
        fxh2 = f(x)
        grad[idx] = (fxh1- fxh2)/(2*h)

        x[idx] = tmp_va1
    # print("grad:"+str(grad))
    return grad

def numerical_gradient(f,X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f,X)
    else:
        grad = np.zeros_like(X)
        for idx,x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f,x)
        return grad

def gen_flow(arr, height, width):
    flow = numerical_gradient(function_2,np.array(arr))
    flow = flow.reshape((2, height, width))
    flow = np.transpose(flow, [1,2,0])
    return flow

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
    # Function: gen_flow_circle
    center = [0, 0]
    flow = gen_flow_circle(center, height=11, width=11)

    # Function: gen_flow
    # x0_s = 0
    # x0_e = 101
    # x1_s = 0
    # x1_e = 101
    # stride = 1
    # x0= np.arange(x0_s,x0_e,stride)
    # x1=np.arange(x1_s,x1_e,stride)
    # X,Y= np.meshgrid(x0,x1)
    # X = X.flatten()
    # Y = Y.flatten()

    # height = (x0_e - x0_s)
    # width = (x1_e - x1_s)
    # flow = gen_flow([X, Y], height=height, width=width)
    # flow = flow / 50000 # change the value of flow
    # flow = - flow