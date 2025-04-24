import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss za rešavanje problema neizbalansiranih klasa.

    Formula: FL(p_t) = -alpha * (1 - p_t) ** gamma * log(p_t)

    gde je p_t verovatnoća za ciljnu klasu.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (Tensor, float, optional): težina za svaku klasu. Ukoliko je None, sve klase imaju težinu 1
            gamma (float): fokusirajući parametar (veća vrednost više fokusira na teške primere)
            reduction (str): 'none' | 'mean' | 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: predikcije modela (pre softmax-a)
            targets: stvarne oznake

        Returns:
            Calculated loss
        """
        # Konverzija inputs u verovatnoće primenom log_softmax
        log_probs = F.log_softmax(inputs, dim=1)

        # Odabir log verovatnoće za tačnu klasu
        targets = targets.view(-1, 1)
        log_p = log_probs.gather(1, targets)
        log_p = log_p.view(-1)

        # Računanje p_t
        probs = F.softmax(inputs, dim=1)
        p_t = probs.gather(1, targets).view(-1)

        # Primena formule focal loss-a
        batch_loss = -((1 - p_t) ** self.gamma) * log_p

        # Primena alpha težina ako postoje
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.gather(0, targets.view(-1))
                batch_loss = alpha_t * batch_loss
            else:
                alpha_t = self.alpha
                batch_loss = alpha_t * batch_loss

        # Primena redukcije
        if self.reduction == 'mean':
            loss = batch_loss.mean()
        elif self.reduction == 'sum':
            loss = batch_loss.sum()
        else:  # 'none'
            loss = batch_loss

        return loss