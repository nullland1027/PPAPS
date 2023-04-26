from tensorboardX import SummaryWriter
from dl_preds import AttentionNet
import torch


model = AttentionNet(1082, 512, 2)
writer = SummaryWriter('logs')

writer.add_graph(model, (torch.zeros(1, 1082), ))
writer.close()
