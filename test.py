import torch
from module import DynamicsModel
import numpy as np
from torch.utils.data import DataLoader, TensorDataset,random_split
import matplotlib.pyplot as plt

dataset = np.load("/home/wei/my_d3/expert_data_with_next.npz")
states = torch.from_numpy(dataset["states"])
actions = torch.from_numpy(dataset["actions"])
next_states = torch.from_numpy(dataset["next_states"])

dataset = TensorDataset(states, actions, next_states)

dataloader = DataLoader(
    dataset, 
    batch_size=200, 
    pin_memory=True,  # 加速 CPU→GPU 传输
    num_workers=4     # 多进程加载数据
)

# 定义划分比例（80% 训练，20% 测试）
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DynamicsModel().to(device)  # 假设已定义 DynamicsModel
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()
losses = []
epochs = 1000

for epoch in range(epochs):
    for batch in train_loader:
        s, a, s_next = batch
        
        # 非阻塞传输到 GPU
        s = s.to(device, non_blocking=True)
        a = a.to(device, non_blocking=True)
        s_next = s_next.to(device, non_blocking=True)
        
        # 前向计算和损失
        pred = model(s, a)
        loss =  loss_fn(pred,s_next)
        # print(f"epoch:{epochs}------>loss:{loss}")
        losses.append(loss.item())
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch % 100) == 0:
            torch.save(model,f"./logs/model{epoch}.pth")

Fig = plt.figure()
plt.plot(range(epochs),losses)
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()

# with torch.no_grad():
#     Pred = model(X)
#     Pred[:,torch.argmax(Pred,axis=1)] = 1
#     Pred[Pred!=1]=0
#     correct = torch.sum((Pred==Y).all(1))
#     total = Y.size(0)
#     print(f'测试集精度：{100*correct/total}%')
