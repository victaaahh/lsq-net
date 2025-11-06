import torch.nn as nn
import torch.nn.functional as F


class MishraDistiller(nn.Module):
    def __init__(self, alpha=1, beta=1, gamma=1, temperature=1.0):
        super().__init__()

        # weight for teacher CE loss (usually 0 unless training teacher)
        self.alpha = alpha

        # weight for student CE loss
        self.beta = beta

        # weight for distillation loss (teacher logits to student probs)
        self.gamma = gamma

        self.temperature = temperature

    def forward(self, student_logits, teacher_logits, targets):
        T = self.temperature

        loss_teacher = F.cross_entropy(teacher_logits, targets) * self.alpha

        loss_student = F.cross_entropy(student_logits, targets) * self.beta

        # teacher and student probs from logits with temperature scaling
        p_T = F.softmax(teacher_logits / T, dim=1)
        p_A = F.log_softmax(student_logits / T, dim=1)

        # Cross entropy H(z_T, p_A)
        # Note: Using KL divergence scaled by T^2 = gamma * T^2 * KL, but KL is only equvalent for kd where the teacher is frozen.
        # loss_distill = F.sum(p_A, p_T, reduction="batchmean") * (T**2) * self.gamma
        loss_distill = -torch.sum(p_T * p_A, dim=1).mean() * (T**2) * self.gamma

        return loss_teacher + loss_student + loss_distill
