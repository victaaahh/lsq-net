import torch.nn as nn


class KDModel(nn.Module):
    def __init__(self, teacher, student, scheme):
        super().__init__()
        self.teacher = teacher
        if scheme == "B" or scheme == "C":
            self.teacher.requires_grad = False
        self.student = student

    def forward(self, x):
        teacher_out = self.teacher(x)
        student_out = self.student(x)
        return teacher_out, student_out
