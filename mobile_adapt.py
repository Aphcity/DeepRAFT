import torch.utils.data.distributed

import model

torch.serialization.add_safe_globals([model.StegaStampEncoder])
torch.serialization.add_safe_globals([model.StampDetector])
torch.serialization.add_safe_globals([model.Dense])
torch.serialization.add_safe_globals([torch.nn.modules.linear.Linear])
torch.serialization.add_safe_globals([model.ImageEmbedder])
torch.serialization.add_safe_globals([torch.nn.modules.container.Sequential])
torch.serialization.add_safe_globals([model.Conv2D])
torch.serialization.add_safe_globals([torch.nn.modules.conv.Conv2d])
torch.serialization.add_safe_globals([torch.nn.modules.activation.ReLU])
torch.serialization.add_safe_globals([torch.nn.modules.upsampling.Upsample])

'定义转化后的模型名称'
model_en_pt ='model_en.pt'
model_de_pt ='model_de.pt'

'加载pytorch模型'
model_en = torch.load('results/test_mobile_4/saved_models/encoder.pth')
model_de = torch.load('results/test_mobile_4/saved_models/detector.pth')

'模型在cpu上运行'
device = torch.device('cpu')
model_en.to(device)
model_de.to(device)
model_en.eval()
model_de.eval()

'定义输入图片的大小'
input_tensor = torch.rand(1, 3, 400, 400)

'转化模型并存储'
mobile_en = torch.jit.trace(model_en, input_tensor)
model_de = torch.jit.trace(model_de, input_tensor)
mobile_en.save(model_en_pt)
model_de.save(model_de_pt)
