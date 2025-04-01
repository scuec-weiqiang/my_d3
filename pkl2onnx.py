import torch
import pickle
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")            #如果训练时用的GPU，必须还得使用GPU
# torch_model = torch.load("/home/wei/my_d3/policy.pkl") 											#加载.pkl文件
# batch_size = 1  										#batch_size需要定下来，可不为1
# input_shape = (3,208,976)                     												#模型的输入，根据训练时数据集的输入

# # set the model to inference mode
# torch_model.eval()            #切换到推理模式

# x = torch.randn(batch_size,*input_shape)		
# x = x.to(device)
# export_onnx_file = "torch-save.onnx"					
# torch.onnx.export(torch_model.module,
#                     x,
#                     export_onnx_file,
#                     opset_version=10,
#                     do_constant_folding=True,	
#                     input_names=["input"],	
#                     output_names=["output"],	
#                     dynamic_axes={"input":{0:"batch_size"},	
#                                     "output":{0:"batch_size"}})
