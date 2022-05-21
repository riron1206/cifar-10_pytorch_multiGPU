# torch.nn.DataParallel()でmultiGPU
- cifar-10で試したがあんま早くならん。。。シングルGPUの精度より悪化してそうだし
  - 50sec（1GPU）-> 42sec（2GPU）-> 37sec（3GPU）になるレベル

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

