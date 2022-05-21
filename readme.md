# torch.nn.DataParallel()でmultiGPU
cifar-10で試したがあんま早くならん。。。
```
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ネットワーク宣言
net = ResNet18()
# cuda or cpu?
net = net.to(device)
# 複数GPU使用宣言
if device == 'cuda':
    net = torch.nn.DataParallel(net) # make parallel
    cudnn.benchmark = True
```

