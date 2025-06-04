import torch
import torch.nn as nn
import torch.nn.functional as F

class AAMSoftmaxLoss(nn.Module):
     def __init__(self, hidden_dim, speaker_num, s=15, m=0.3, easy_margin=False, **kwargs):
          super(AAMSoftmaxLoss, self).__init__()
          import math

          self.test_normalize = True
          
          self.m = m
          self.s = s
          self.speaker_num = speaker_num
          self.hidden_dim = hidden_dim
          self.weight = torch.nn.Parameter(torch.FloatTensor(speaker_num, hidden_dim), requires_grad=True)
          self.ce = nn.CrossEntropyLoss()
          nn.init.xavier_normal_(self.weight, gain=1)

          self.easy_margin = easy_margin
          self.cos_m = math.cos(self.m)
          self.sin_m = math.sin(self.m)

          # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
          self.th = math.cos(math.pi - self.m)
          self.mm = math.sin(math.pi - self.m) * self.m

     def forward(self, x_BxH, labels_B):

          assert len(x_BxH) == len(labels_B)
          assert torch.min(labels_B) >= 0
          assert torch.max(labels_B) < self.speaker_num

          labels_B = labels_B.to(x_BxH.device)
          
          # cos(theta)
          cosine = F.linear(F.normalize(x_BxH), F.normalize(self.weight.to(x_BxH.device)))
          # cos(theta + m)
          sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
          phi = cosine * self.cos_m - sine * self.sin_m

          if self.easy_margin:
               phi = torch.where(cosine > 0, phi, cosine)
          else:
               phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

          #one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
          one_hot = torch.zeros_like(cosine)
          one_hot.scatter_(1, labels_B.view(-1, 1), 1)
          output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
          output = output * self.s

          loss    = self.ce(output, labels_B)
          return loss

if __name__ == "__main__": 

     input_vector = torch.randn(8, 256)  # Example batch of 32 vectors with 128 dimensions
     labels = torch.randint(0, 1000, (8,)) 

     print(labels)

     aam_softmax = AAMSoftmaxLoss(256, speaker_num=1000)
     loss = aam_softmax(input_vector, labels)
     print(loss)