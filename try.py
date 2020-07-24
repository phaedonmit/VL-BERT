import torch
import torch.nn as nn


# w = nn.Parameter(torch.ones(1))
# x = torch.ones(1) * 2
# y = torch.ones(1) * (2)

# for _ in range(3):
#     output = x * w
#     output.backward()
# for _ in range(3):
#     output = y * w
#     output.backward()
# print(w.grad)

x = torch.ones(1) *2
y = nn.Parameter(torch.ones(1))
z = y * x


w = torch.ones(1)*3
for _ in range(3):
    output = z * w
    # output.backward(retain_graph=True)
print(y.grad)
output = z * w
output.backward()
print(y.grad)

exit()

