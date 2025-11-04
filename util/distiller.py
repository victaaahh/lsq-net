import torch
import torch.nn.functional as F

class MishraDistiller:
    def __init__(self, alpha=0, beta=1, gamma=1, temperature=1.0):
        self.alpha = alpha  # weight for teacher CE loss (usually 0 unless training teacher)
        self.beta = beta    # weight for student CE loss
        self.gamma = gamma  # weight for distillation loss (teacher logits to student probs)
        self.temperature = temperature

    def loss(self, student_logits, teacher_logits, targets):
        T = self.temperature

        # teacher and student probs from logits with temperature scaling
        p_T = F.softmax(teacher_logits / T, dim=1)
        p_A = F.log_softmax(student_logits / T, dim=1)

        # Cross entropy H(y, p_*) — H(y, p_T) part would usually be zero unless training teacher as in scheme A
        loss_teacher = 0
        if self.alpha > 0:
            loss_teacher = F.cross_entropy(teacher_logits, targets) * self.alpha

        loss_student = F.cross_entropy(student_logits, targets) * self.beta

        # Cross entropy H(z_T, p_A)
        # Note: Using KL divergence scaled by T^2 = gamma * T^2 * KL
        loss_distill = F.kl_div(p_A, p_T, reduction='batchmean') * (T ** 2) * self.gamma

        return loss_teacher + loss_student + loss_distill
