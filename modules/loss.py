import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    This class is contrative loss function like InfoNCE loss.
    """
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, query: Tensor, pos_emb: Tensor, neg_emb: Tensor) -> Tensor:
        """
        Args:
            query(Tensor): Query embedding tensor (1, dim)
            pos_emb(Tensor): Positive embedding tensor (size_of_positive_samples, dim)
            neg_emb(Tensor): Negative embedding tensor (size_of_negative_samples, dim)
            
        Returns:
            loss(Tensor): Contrastive loss
        """

        query = F.normalize(query, p=2, dim=-1)
        pos_emb = F.normalize(pos_emb, p=2, dim=-1)
        neg_emb = F.normalize(neg_emb, p=2, dim=-1)
        
        pos_sim = torch.matmul(pos_emb, query.T) / self.temperature  # (size_of_positive_samples,)
        neg_sim = torch.matmul(neg_emb, query.T) / self.temperature  # (size_of_negative_samples,)

        all_sim = torch.sum(torch.exp(torch.cat([pos_sim, neg_sim])))

        loss = -torch.log(torch.exp(pos_sim) / all_sim).mean()

        return loss

