import torch


dim1 = (20, 6, 90, 90)

dim2 = (20, 90, 90)

matrix1 = torch.rand(dim1)
matrix2 = torch.rand(dim2)
matrix2 = torch.unsqueeze(matrix2,1)

result = torch.matmul(matrix1, matrix2)

print(result.size())