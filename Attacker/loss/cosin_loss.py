import torch
import torch.nn.functional as F
import torch.nn as nn

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, input, target):
        # Normalize input and target vectors
        input_norm = F.normalize(input, p=2, dim=-1)
        target_norm = F.normalize(target, p=2, dim=-1)
        
        # Compute cosine similarity
        cos_sim = torch.sum(input_norm * target_norm, dim=-1)
        
        # Compute cosine similarity loss (1 - cosine similarity)
        loss = 1 - cos_sim
        return loss.mean()  # Averaging over the batch

if __name__ == "__main__":
    # Create an instance of the loss class
    cosine_loss = CosineSimilarityLoss()

    # Example vectors
    input_vector = torch.randn(32, 128)  # Example batch of 32 vectors with 128 dimensions
    target_vector = torch.randn(32, 128) # Example target vectors of the same shape

    # Compute the loss
    loss = cosine_loss(input_vector, target_vector)
    print("Cosine Similarity Loss:", loss.item())