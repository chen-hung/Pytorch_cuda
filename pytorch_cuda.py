
import torch
import torchvision



print("cuda是否可用:",torch.cuda.is_available())
#cuda是否可用；
 
print("gpu数量；",torch.cuda.device_count())
#返回gpu数量；
 
print(torch.cuda.get_device_name(0))
#返回gpu名字，设备索引默认从0开始；
 
print("设备索引:",torch.cuda.current_device())
#返回当前设备索引

print(torch.cuda.get_device_properties(0))

#print("gpu容量:",torch.cuda.get_device_capability())
#查看gpu的容量
print("-"*60)
print("torchvision:",torchvision.__version__) #获取torchvision版本
print("torch:",torch.__version__) # 获取torch版本

torch.cuda.set_device(0) # 设定使用指定GPU：0号GPU
print(torch.cuda.is_available()) # 是否有已经配置好可以使用的GPU (若True则有)
print(torch.cuda.device_count())  # 可用GPU块数
print(torch.cuda.get_device_capability()) #获取所使用的GPU的计算力
print(torch.cuda.get_device_name()) #获取该GPU的名称
print(torch.cuda.get_device_properties(0)) #获取指定GPU的常见属性，必须指定GPU，否则报错

print("cuda:{}".format(torch.version.cuda))
print("cudnn:{}".format(torch.backends.cudnn.version()))

'''
t = torch.cuda.get_device_properties(0).total_memory
c = torch.cuda.memory_cached(0)
a = torch.cuda.memory_allocated(0)
f = c-a  # free inside cache
'''