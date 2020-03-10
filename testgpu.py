import torch
print(torch.version.cuda)
if torch.cuda.is_available():
    print('true')
else:
    print('false')