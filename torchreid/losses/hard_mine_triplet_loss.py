from __future__ import division, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
        mining_type (str, optional): type of mining strategy.
            Options: 'batch_all' (default), 'batch_hard', 'batch_semihard', 'batch_hard_soft'.
            Default is 'batch_all'.
    """

    def __init__(self, margin=0.3, mining_type='batch_all'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.mining_type = mining_type
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

        assert mining_type in ['batch_hard', 'batch_all', 'batch_semihard', 'batch_hard_soft'], \
            f"mining_type must be one of ['batch_hard', 'batch_all', 'batch_semihard', 'batch_hard_soft'], got {mining_type}"

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # Mask for positive pairs (same identity)
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())

        if self.mining_type == 'batch_hard':
            return self._batch_hard(dist, mask)
        elif self.mining_type == 'batch_all':
            return self._batch_all(dist, mask)
        elif self.mining_type == 'batch_semihard':
            return self._batch_semihard(dist, mask)
        elif self.mining_type == 'batch_hard_soft':
            return self._batch_hard_soft(dist, mask)

    def _batch_hard(self, dist, mask):
        """Batch hard mining: select hardest positive and hardest negative for each anchor.

        Args:
            dist (torch.Tensor): pairwise distance matrix (n, n).
            mask (torch.Tensor): boolean mask for positive pairs (n, n).
        """
        n = dist.size(0)
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)

    def _batch_all(self, dist, mask):
        """Batch all mining: use all valid triplets in the batch.

        Args:
            dist (torch.Tensor): pairwise distance matrix (n, n).
            mask (torch.Tensor): boolean mask for positive pairs (n, n).
        """
        n = dist.size(0)

        # For each anchor, get all positive and negative distances
        all_losses = []
        for i in range(n):
            pos_dists = dist[i][mask[i]]  # distances to all positives
            neg_dists = dist[i][~mask[i]]  # distances to all negatives

            if len(pos_dists) == 0 or len(neg_dists) == 0:
                continue

            # Compute loss for all combinations of positives and negatives
            # pos_dists: (num_pos,), neg_dists: (num_neg,)
            pos_dists = pos_dists.unsqueeze(1)  # (num_pos, 1)
            neg_dists = neg_dists.unsqueeze(0)  # (1, num_neg)

            # Triplet loss: max(0, d(a,p) - d(a,n) + margin)
            triplet_loss = F.relu(pos_dists - neg_dists + self.margin)  # (num_pos, num_neg)
            all_losses.append(triplet_loss)

        if len(all_losses) == 0:
            return torch.tensor(0.0, device=dist.device, requires_grad=True)

        # Average over all triplets
        return torch.cat([loss.flatten() for loss in all_losses]).mean()

    def _batch_semihard(self, dist, mask):
        """Batch semi-hard mining: select semi-hard negatives (harder than positive but within margin).

        Args:
            dist (torch.Tensor): pairwise distance matrix (n, n).
            mask (torch.Tensor): boolean mask for positive pairs (n, n).
        """
        n = dist.size(0)
        dist_ap, dist_an = [], []

        for i in range(n):
            # Get hardest positive
            hardest_pos = dist[i][mask[i]].max()
            dist_ap.append(hardest_pos.unsqueeze(0))

            # Get semi-hard negative: dist(a,n) > dist(a,p) but dist(a,n) < dist(a,p) + margin
            neg_dists = dist[i][mask[i] == 0]
            semihard_mask = (neg_dists > hardest_pos) & (neg_dists < hardest_pos + self.margin)

            if semihard_mask.any():
                # Use hardest semi-hard negative
                dist_an.append(neg_dists[semihard_mask].min().unsqueeze(0))
            else:
                # Fall back to hardest negative if no semi-hard negatives
                dist_an.append(neg_dists.min().unsqueeze(0))

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)

    def _batch_hard_soft(self, dist, mask):
        """Batch hard-soft mining: weighted combination using softmax over hard examples.

        Args:
            dist (torch.Tensor): pairwise distance matrix (n, n).
            mask (torch.Tensor): boolean mask for positive pairs (n, n).
        """
        n = dist.size(0)

        # Use log-sum-exp for numerical stability
        # For positives: weight harder (larger distance) positives more
        # For negatives: weight harder (smaller distance) negatives more
        dist_ap = []
        dist_an = []

        for i in range(n):
            pos_dists = dist[i][mask[i]]
            neg_dists = dist[i][mask[i] == 0]

            if len(pos_dists) == 0 or len(neg_dists) == 0:
                continue

            # Weighted positive: emphasize hard positives
            pos_weights = F.softmax(pos_dists, dim=0)
            weighted_pos = (pos_weights * pos_dists).sum()
            dist_ap.append(weighted_pos.unsqueeze(0))

            # Weighted negative: emphasize hard negatives (small distances)
            neg_weights = F.softmax(-neg_dists, dim=0)
            weighted_neg = (neg_weights * neg_dists).sum()
            dist_an.append(weighted_neg.unsqueeze(0))

        if len(dist_ap) == 0 or len(dist_an) == 0:
            return torch.tensor(0.0, device=dist.device, requires_grad=True)

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)
