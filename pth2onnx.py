import argparse
import torch
import torch.onnx
# 1. 复用 train.py 里的模型构造器
from utils import model_manager_ft      # 和 train.py 完全一致
# 2. 复用 train.py 里的超参默认值
from easydict import EasyDict as edict   # 若没装 easydict 可把 cfg 写成类也行

# 和 train.py 默认参数保持一致
cfg = edict({
    'dataset': 'vcclothes',
    'model': 'hr',          # 训练时用的 backbone
    'height': 256,
    'width': 128,
})

def main():
    # ---- 构造模型（和 train.py 完全一致） ----
    from utils import data_manager
    dataset = data_manager.init_dataset(name=cfg.dataset)
    num_classes = dataset.train_data_ids

    model = model_manager_ft.init_model(name=cfg.model, class_num=num_classes)
    model = model.cuda()
    model.eval()

    # ---- 加载 checkpoint ----
    ckp_path = 'logs/vcclothes/checkpoint.pth'
    checkpoint = torch.load(ckp_path, map_location="cpu", weights_only=False)

    # 去掉 module. 前缀
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        new_state_dict[k.replace("module.", "")] = v
    model.load_state_dict(new_state_dict, strict=True)
    print('✅ checkpoint loaded')

    # ---- 构造 dummy input ----
    dummy = torch.randn(1, 3, cfg.height, cfg.width).cuda()

    # ---- 导出 ONNX ----
    onnx_path = 'screid_vcclothes.onnx'
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},
                      'output': {0: 'batch_size'}}
    )
    print(f'✅ ONNX 已生成：{onnx_path}')

if __name__ == '__main__':
    main()