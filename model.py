
import numpy as np
import torch
import torch.nn as nn
from thop import profile
from torch import sigmoid, relu
import torch.nn.functional as F
from torch.nn.functional import conv2d

#
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





class TemporalAttention(nn.Module):
    '''
       compute temporal attention scores
       --------
       Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)  (1,5,26,9)
       Output: (batch_size, num_of_timesteps, num_of_timesteps)
    '''

    def __init__(self, num_of_timesteps, num_of_vertices, num_of_features):
        super(TemporalAttention, self).__init__()

        self.num_of_timesteps = num_of_timesteps
        self.num_of_vertices = num_of_vertices
        self.num_of_features = num_of_features
        self.U_1 = nn.Parameter(torch.zeros(self.num_of_vertices, 1))
        nn.init.xavier_uniform_(self.U_1.data, gain=1.414)
        self.U_2 = nn.Parameter(torch.zeros(self.num_of_features, self.num_of_vertices))
        nn.init.xavier_uniform_(self.U_2.data, gain=1.414)
        self.U_3 = nn.Parameter(torch.zeros(self.num_of_features, 1))
        nn.init.xavier_uniform_(self.U_3.data, gain=1.414)
        self.b_e = nn.Parameter(torch.zeros(1, self.num_of_timesteps, self.num_of_timesteps))
        nn.init.xavier_uniform_(self.b_e.data, gain=1.414)
        self.v_e = nn.Parameter(torch.zeros(self.num_of_timesteps, self.num_of_timesteps))
        nn.init.xavier_uniform_(self.v_e.data, gain=1.414)

    def forward(self, x):
        # shape of lhs is (batch_size, T, V)
        a = x.permute(0, 1, 3, 2)  # 1,5,90,90
        # print(a.shape)
        # print(self.U_1.shape)
        lhs = torch.matmul(a, self.U_1)
        # print(lhs.shape)

        lhs = lhs.reshape(x.shape[0], self.num_of_timesteps, self.num_of_features)
        lhs = torch.matmul(lhs, self.U_2)  # torch.Size([1, 5, 26])

        # shape of rhs is (batch_size, V, T)
        b = x.permute(2, 0, 3, 1)  # torch.Size([26, 1, 9, 5])
        zj = torch.squeeze(self.U_3)
        rhs = torch.matmul(zj, b)  # torch.Size([26, 1, 5])
        rhs = rhs.permute(1, 0, 2)  # torch.Size([1, 26, 5])

        # shape of product is (batch_size, T, T)
        product = torch.matmul(lhs, rhs)
        product = sigmoid(product + self.b_e).permute(1, 2, 0)  # torch.Size([5, 5, 1])
        product = torch.matmul(self.v_e, product)
        s = product.permute(2, 0, 1)

        # normalization
        s = s - torch.max(s, dim=1, keepdim=True)[0]
        exp = torch.exp(s)
        S_normalized = exp / torch.sum(exp, dim=1, keepdim=True)

        return S_normalized  # torch.Size([1, 5, 5])
        # return S_normalized,(x.shape[0], x.shape[1], x.shape[1])


# model = TemporalAttention( 6, 90, 240)
# inputs = torch.randn(1,6, 90, 240)
# out = model(inputs)
# print(out.shape)



def reshape_dot(x, TATT):
    outs = torch.matmul((x.permute(0, 2, 3, 1))
                        .reshape(x.shape[0], -1, x.shape[1]), TATT).reshape(-1, x.shape[1],
                                                                            x.shape[2],
                                                                            x.shape[3])
    return outs


def LayerNorm(x_residual, time_conv_output):
    '''
      Apply relu and layer normalization
    '''
    relu_x = F.relu(x_residual + time_conv_output)
    ln = torch.nn.functional.normalize(relu_x, p=1, dim=2)
    return ln


class SpatialAttention(nn.Module):
    def __init__(self, num_of_timesteps, num_of_vertices, num_of_features):
        super(SpatialAttention, self).__init__()

        self.num_of_timesteps = num_of_timesteps
        self.num_of_vertices = num_of_vertices
        self.num_of_features = num_of_features
        self.W_1 = nn.Parameter(torch.zeros(num_of_timesteps, 1))
        nn.init.xavier_uniform_(self.W_1.data, gain=1.414)
        self.W_2 = nn.Parameter(torch.zeros(self.num_of_features, self.num_of_timesteps))
        nn.init.xavier_uniform_(self.W_2.data, gain=1.414)
        self.W_3 = nn.Parameter(torch.zeros(self.num_of_features, 1))
        nn.init.xavier_uniform_(self.W_3.data, gain=1.414)
        self.b_s = nn.Parameter(torch.zeros(1, self.num_of_vertices, self.num_of_vertices))
        nn.init.xavier_uniform_(self.b_s.data, gain=1.414)
        self.v_s = nn.Parameter(torch.zeros(self.num_of_vertices, self.num_of_vertices))
        nn.init.xavier_uniform_(self.v_s.data, gain=1.414)

    def forward(self, x):
        # shape of lhs is (batch_size, V, T)
        lhs = torch.matmul(x.permute(0, 2, 3, 1), self.W_1)
        lhs = lhs.reshape(x.shape[0], self.num_of_vertices, self.num_of_features)
        lhs = torch.matmul(lhs, self.W_2)  # torch.Size([1, 26, 5])

        # shape of rhs is (batch_size, T, V)
        zj = torch.squeeze(self.W_3)
        rhs = torch.matmul(zj, x.permute(1, 0, 3, 2))
        rhs = rhs.permute(1, 0, 2)  # torch.Size([1, 5, 26])

        # shape of product is (batch_size, V, V)
        product = torch.matmul(lhs, rhs)  # torch.Size([1, 26, 26])
        product = sigmoid(product + self.b_s).permute(1, 2, 0)
        product = torch.matmul(self.v_s, product)
        s = product.permute(2, 0, 1)

        # normalization
        s = s - torch.max(s, dim=1, keepdim=True)[0]
        exp = torch.exp(s)
        S_normalized = exp / torch.sum(exp, dim=1, keepdim=True)

        return S_normalized  # torch.Size([2, 26, 26])


################K-order static GCN###################
class cheb_conv_with_SAt_static(nn.Module):
    '''
        K-order chebyshev graph convolution with static graph structure
        --------
        Input:  [x (batch_size, num_of_timesteps, num_of_vertices, num_of_features),
                 SAtt(batch_size, num_of_vertices, num_of_vertices)]
        Output: (batch_size, num_of_timesteps, num_of_vertices, num_of_filters)

    '''

    def __init__(self, num_of_filters, k, num_of_timesteps, num_of_vertices, num_of_features):

        super(cheb_conv_with_SAt_static, self).__init__()
        self.k = k
        self.num_of_filters = num_of_filters
        self.num_of_features = num_of_features
        self.num_of_timesteps = num_of_timesteps
        self.num_of_vertices = num_of_vertices
        # self.cheb_polynomials = cheb_polynomials
        self.Theta = nn.Parameter(torch.zeros(self.k, self.num_of_features, self.num_of_filters))
        nn.init.xavier_uniform_(self.Theta.data, gain=1.414)

    def forward(self, x, satt, cheb_polynomials):
        # _, num_of_timesteps, num_of_vertices, num_of_features = x.shape

        outputs = []
        for time_step in range(self.num_of_timesteps):
            # shape is (batch_size, V, F)
            graph_signal = x[:, time_step, :, :]
            # shape is (batch_size, V, F')
            output = torch.zeros(x.shape[0], self.num_of_vertices, self.num_of_filters)
            output = output.to(DEVICE)

            for kk in range(self.k):
                # shape of T_k is (V, V)
                T_k = cheb_polynomials[:, kk, :, :]

                # shape of T_k_with_at is (batch_size, V, V)
                T_k_with_at = T_k * satt
                # print(T_k_with_at.shape)

                # shape of theta_k is (F, num_of_filters)
                theta_k = self.Theta[kk]

                # shape is (batch_size, V, F)
                rhs = torch.matmul(T_k_with_at.permute(0, 2, 1), graph_signal)
                output = output + torch.matmul(rhs, theta_k)

            outputs.append(output.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        outputs = F.relu(outputs)
        return outputs


#################################TCN##############################
class TCN(nn.Module):
    def __init__(self, num_of_timesteps):
        super(TCN, self).__init__()
        self.num_of_timesteps = num_of_timesteps
        self.c1 = nn.Conv2d(self.num_of_timesteps, self.num_of_timesteps, 3, 1, padding='same')
        # self.c2 = nn.Conv2d(32, self.num_of_timesteps, 3, 1, padding='same')

    def forward(self, x):
        time_conv_output = self.c1(x)
        # time_conv_output = self.c2(time_conv_output)
        return time_conv_output  # 2,45,86,86


#
# model = TCN(5)
# a = torch.randn(2, 5, 90, 90)
# out = model(a)
# print(out.shape)


#############################MCSTGCN#########################
class san_stgcn(nn.Module):
    def __init__(self, num_of_timesteps, num_of_vertices, num_of_features, num_of_time_filters,
                 num_of_chev_filters, time_conv_kernel, time_conv_strides, k):
        super(san_stgcn, self).__init__()
        self.num_of_timesteps = num_of_timesteps
        self.num_of_vertices = num_of_vertices
        self.num_of_features = num_of_features
        # self.num_of_time_filters = num_of_time_filters
        self.time_conv_kernel = time_conv_kernel
        self.num_of_chev_filters = num_of_chev_filters
        self.time_conv_strides = time_conv_strides
        self.k = k

        self.temporal_At = TemporalAttention(self.num_of_timesteps, self.num_of_vertices,
                                             self.num_of_features)
        self.spatial_At = SpatialAttention(self.num_of_timesteps, self.num_of_vertices,
                                           self.num_of_features)
        self.spatial_gcn = cheb_conv_with_SAt_static(num_of_filters=self.num_of_chev_filters, k=self.k,
                                                     num_of_timesteps=self.num_of_timesteps,
                                                     num_of_vertices=self.num_of_vertices,
                                                     num_of_features=self.num_of_features)

        self.t1 = nn.Conv2d(self.num_of_timesteps, self.num_of_timesteps, 3, 1, padding='same')
        self.t2 = nn.Conv2d(self.num_of_timesteps, 1, 3, 1)

    def forward(self, x, che):
        # TemporalAttention 
        # output shape is (batch_size, T, T)
        temporal_Att = self.temporal_At(x)
        x_TAt = reshape_dot(x, temporal_Att)

        # SpatialAttention
        # output shape is (batch_size, V, V)
        spatial_Att = self.spatial_At(x_TAt)

        # Temporal  Convolution
        time_conv_output = self.t1(x_TAt)

        # Graph Convolution with spatial attention
        # output shape is (batch_size, T, V, F)
        spatial_gcns = self.spatial_gcn(time_conv_output, spatial_Att, che)  # torch.Size([2, 5, 90, 45])

        # Temporal  Convolution
        output = self.t2(spatial_gcns)

        return output


#
#
# cheb_polynomials = torch.randn(2, 3, 90, 90)
# sample_shape = torch.randn(2, 5, 90, 90)
# model = san_stgcn(5, 90, 90, 10, 45, 3, 1, 3)
# out = model(sample_shape, cheb_polynomials)
# print(out.shape)

class STGCN_model(nn.Module):
    def __init__(self, k, num_of_timesteps, num_of_chev_filters, num_of_time_filters, time_conv_strides,
                 time_conv_kernel,
                 num_of_vertices, num_of_features, nhid1, nhid2, n_class):
        super(STGCN_model, self).__init__()
        self.k = k
        self.num_of_timesteps = num_of_timesteps
        self.num_of_chev_filters = num_of_chev_filters
        self.num_of_time_filters = num_of_time_filters
        self.time_conv_strides = time_conv_strides
        self.time_conv_kernel = time_conv_kernel
        self.num_of_vertices = num_of_vertices
        self.num_of_features = num_of_features
        self.nhid1 = nhid1
        self.nhid2 = nhid2
        self.n_class = n_class
        self.st1 = san_stgcn(self.num_of_timesteps, self.num_of_vertices, self.num_of_features,
                             self.num_of_time_filters, self.num_of_chev_filters, self.time_conv_kernel,
                             self.time_conv_strides, self.k)

        self.f1 = nn.Flatten()
        self.l1 = nn.Linear(3784, 1024)  # 24300 3784
        self.bn1 = nn.BatchNorm1d(1024)
        self.d1 = nn.Dropout(p=0.3)
        self.l2 = nn.Linear(1024, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.d2 = nn.Dropout(p=0.3)
        self.l3 = nn.Linear(256, n_class)
        self.logs = nn.LogSoftmax(dim=1)

    def forward(self, x, che):
        block_out = self.st1(x, che)  # [20, 6, 90, 45]

        # block_out = self.bn1(block_out)
        block_outs = self.f1(block_out)
        block_outs = self.d1(self.bn1(self.l1(block_outs)))
        block_outs = self.d2(self.bn2(self.l2(block_outs)))
        block_outs = self.l3(block_outs)
        block_outs = self.logs(block_outs)

        return block_outs

cheb_polynomials = torch.randn(2,3, 90, 90)
sample_shape = torch.randn(2, 6, 90, 90)
# model = build_brainNet(3,5, 45, 10, 1, 3, 90, 90, 1024, 256, 2)
model = STGCN_model(3, 6, 45, 10, 1, 3, 90, 90, 1024, 256, 2)
model = model.cuda()
cheb_polynomials = cheb_polynomials.cuda()
sample_shape = sample_shape.cuda()
out = model(sample_shape, cheb_polynomials)
# print(out)

flops, params = profile(model, (sample_shape, cheb_polynomials,))
print('flops: ', flops, 'params: ', params)
print('flops: %.3f M, params: %.3f M' % (flops / 2000000.0, params / 1000000.0))

