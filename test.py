# File: convert_models.py
import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile



def convert_encoder():
    # 载入各组件权重（实际代码中需加载：ConditionAdaptor.pth, CustomConvNeXt.pth, UNet2DConditionModel.pth, vae.pth）
    # 在此示例中直接使用 Dummy 模型
    model = DummyVineREnc()
    # 假定已加载权重：model.load_state_dict(torch.load("your_encoder_weights.pth"))
    model.eval()
    # 构造示例输入：图像张量 shape [1,3,512,512] 和水印张量 shape [1,32]
    dummy_image = torch.rand(1, 3, 512, 512)
    dummy_watermark = torch.rand(1, 32)
    traced_model = torch.jit.trace(model, (dummy_image, dummy_watermark))
    traced_model_optimized = torch.utils.mobile_optimizer.optimize_for_mobile(traced_model)
    traced_model_optimized._save_for_lite_interpreter("vine_r_enc.pt")
    print("Encoder模型已转换并保存为 vine_r_enc.pt")

def convert_decoder():
    model = DummyVineRDec()
    # 假定已加载权重：model.load_state_dict(torch.load("your_decoder_weights.pth"))
    model.eval()
    dummy_image = torch.rand(1, 3, 512, 512)
    traced_model = torch.jit.trace(model, dummy_image)
    traced_model_optimized = torch.utils.mobile_optimizer.optimize_for_mobile(traced_model)
    traced_model_optimized._save_for_lite_interpreter("vine_r_dec.pt")
    print("Decoder模型已转换并保存为 vine_r_dec.pt")

if __name__ == "__main__":
    convert_encoder()
    convert_decoder()
