import torch
import numpy as np
import cv2


def transformer(point, source, target):
    """
    Thin Plate Spline Spatial Transformer Layer
  TPS control points are arranged in arbitrary positions given by `source`.
  U : float Tensor [num_batch, height, width, num_channels].
    Input Tensor.
  source : float Tensor [num_batch, num_point, 2]
    The source position of the control points.
  target : float Tensor [num_batch, num_point, 2]
    The target position of the control points.
  out_size: tuple of two integers [height, width]
    The size of the output of the network (height, width)
    """

    # point: bn, num_point, 2
    def _meshgrid_point(point, source):

        # x_t = torch.matmul(torch.ones([height, 1]), torch.unsqueeze(torch.linspace(-1.0, 1.0, width), 0))
        # y_t = torch.matmul(torch.unsqueeze(torch.linspace(-1.0, 1.0, height), 1), torch.ones([1, width]))
        # if torch.cuda.is_available():
        #     x_t = x_t.cuda()
        #     y_t = y_t.cuda()

        # x_t_flat = x_t.reshape([1, 1, -1])
        # y_t_flat = y_t.reshape([1, 1, -1])

        num_batch, num_point, _ = point.size()


        x_t_flat = point[:,:,0].view(num_batch, 1, num_point)   # bs, 1, num_point
        y_t_flat = point[:,:,1].view(num_batch, 1, num_point)

        num_batch = source.size()[0]
        px = torch.unsqueeze(source[:,:,0], 2)  # [bn, pn, 1]
        py = torch.unsqueeze(source[:,:,1], 2)  # [bn, pn, 1]
        if torch.cuda.is_available():
            px = px.cuda()
            py = py.cuda()
        d2 = torch.square(x_t_flat - px) + torch.square(y_t_flat - py)
        r = d2 * torch.log(d2 + 1e-6) # [bn, pn, h*w]
        # x_t_flat_g = x_t_flat.expand(num_batch, -1, -1)  # [bn, 1, h*w]
        # y_t_flat_g = y_t_flat.expand(num_batch, -1, -1)  # [bn, 1, h*w]
        ones = torch.ones_like(x_t_flat) # [bn, 1, h*w]
        if torch.cuda.is_available():
            ones = ones.cuda()

        grid = torch.cat((ones, x_t_flat, y_t_flat, r), 1) # [bn, 3+pn, num_point]


        return grid

    def _transform(T, source, point):
        num_batch, num_point, num_channels = point.size()

        #out_height, out_width = out_size[0], out_size[1]
        # grid = _meshgrid(out_height, out_width, source) # [bn, 3+pn, h*w]
        grid = _meshgrid_point(point, source) # [bn, 3+pn, num_point]

        # transform A x (1, x_t, y_t, r1, r2, ..., rn) -> (x_s, y_s)
        # [bn, 2, pn+3] x [bn, pn+3, h*w] -> [bn, 2, h*w]
        T_g = torch.matmul(T, grid)     # [bn, 2, num_point]
        # x_s = T_g[:,0,:]
        # y_s = T_g[:,1,:]
        # x_s_flat = x_s.reshape([-1])
        # y_s_flat = y_s.reshape([-1])

        # input_transformed = _interpolate(input_dim, x_s_flat, y_s_flat,out_size)

        # #output = input_transformed.reshape([num_batch, out_height, out_width, num_channels])

        # output = output.permute(0,3,1,2)
        T_g = T_g.permute(0,2,1)
        output = T_g.view(num_batch, num_point, num_channels)

        return output#, condition


    def _solve_system(source, target):
        num_batch  = source.size()[0]
        num_point  = source.size()[1]

        np.set_printoptions(precision=8)

        ones = torch.ones(num_batch, num_point, 1).float()
        if torch.cuda.is_available():
            ones = ones.cuda()
        p = torch.cat([ones, source], 2) # [bn, pn, 3]

        p_1 = p.reshape([num_batch, -1, 1, 3]) # [bn, pn, 1, 3]
        p_2 = p.reshape([num_batch, 1, -1, 3])  # [bn, 1, pn, 3]
        d2 = torch.sum(torch.square(p_1-p_2), 3) # p1 - p2: [bn, pn, pn, 3]   final output: [bn, pn, pn]
        #print("xxxxxxxxxxxxxxxxxxxx")
        #torch.set_printoptions(precision=8)
        #print(d2[0])
        #print(d2.dtype)
        r = d2 * torch.log(d2 + 1e-6) # [bn, pn, pn]


        zeros = torch.zeros(num_batch, 3, 3).float()
        if torch.cuda.is_available():
            zeros = zeros.cuda()
        W_0 = torch.cat((p, r), 2) # [bn, pn, 3+pn]
        W_1 = torch.cat((zeros, p.permute(0,2,1)), 2) # [bn, 3, pn+3]
        W = torch.cat((W_0, W_1), 1) # [bn, pn+3, pn+3]


        W_inv = torch.inverse(W.type(torch.float64))



        zeros2 = torch.zeros(num_batch, 3, 2)
        if torch.cuda.is_available():
            zeros2 = zeros2.cuda()
        tp = torch.cat((target, zeros2), 1) # [bn, pn+3, 2]

        T = torch.matmul(W_inv, tp.type(torch.float64)) # [bn, pn+3, 2]
        T = T.permute(0, 2, 1) # [bn, 2, pn+3]


        return T.type(torch.float32)

    T = _solve_system(source, target)

    output = _transform(T, source, point)

    return output#, condition