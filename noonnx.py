import torch

checkpoint = torch.load("logs/vcclothes/checkpoint.pth", map_location="cpu",  weights_only=False)
print(type(checkpoint))
print(checkpoint.keys())
