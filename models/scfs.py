from numpy import append
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.ops as ops
import torchextractor as tx
import kornia
import numpy as np

from utils.utils import trunc_normal_


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, 
    hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class DINOShallowHead(nn.Module):
    def __init__(self, in_dim, out_dim, bottleneck_dim=256, hidden_dim=2048):
        super().__init__()
        #bottleneck
        self.conv1 = self.conv(in_dim, bottleneck_dim, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(bottleneck_dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.conv(bottleneck_dim, bottleneck_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_dim)
        self.conv3 = self.conv(bottleneck_dim, in_dim, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(in_dim)
        self.gap = nn.AdaptiveAvgPool2d((1,1))

        self.mlp = []
        self.mlp.append(nn.Linear(in_dim, hidden_dim))
        self.mlp.append(nn.BatchNorm1d(hidden_dim))
        self.mlp.append(nn.ReLU())
        self.mlp.append(nn.Linear(hidden_dim, out_dim))
        self.mlp.append(nn.BatchNorm1d(out_dim))
        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        out = self.gap(out).squeeze()

        out = self.mlp(out)
        out = nn.functional.normalize(out, dim=-1, p=2)
        return out

    def conv(self, in_chanel, out_channel, kernel_size, padding):
        return nn.Conv2d(
            in_chanel,
            out_channel,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DINOShallowLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_g, teacher_g, student_l, teacher_l, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """

        # global 
        student_out = student_g / self.student_temp
        student_out = student_out.chunk(2)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_g - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        loss_global1 = torch.sum(-teacher_out[0] * F.log_softmax(student_out[1], dim=-1), dim=-1).mean()
        loss_global2 = torch.sum(-teacher_out[1] * F.log_softmax(student_out[0], dim=-1), dim=-1).mean()
        loss_global = (loss_global1 + loss_global2) * 0.5

        # local
        student_out = student_l / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        teacher_out = F.softmax(teacher_l / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(self.ncrops)

        loss_local = 0
        n_loss_terms = 0
        for i in range(self.ncrops):
            loss = torch.sum(-teacher_out[i] * F.log_softmax(student_out[i], dim=-1), dim=-1)
            loss_local += loss.mean()
            n_loss_terms += 1

        loss_local /= n_loss_terms

        self.update_center(teacher_g)

        return loss_global, loss_local
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    
# global global local local
class StudentWrapper(nn.Module):
    def __init__(self, backbone, head, layer2_head, layer3_head, layer4_head):
        super(StudentWrapper, self).__init__()
        self.backbone = backbone
        self.backbone.fc = nn.Identity()
        self.backbone = tx.Extractor(self.backbone, ["layer2", "layer3", "layer4"])
        self.head = head
        self.layer2_head = layer2_head
        self.layer3_head = layer3_head
        self.layer4_head = layer4_head


    def forward(self, x):
        #import ipdb
        #ipdb.set_trace()
        f_gap, f_dict = self.backbone(torch.cat(x[:]))
        f_gap = f_gap.squeeze()
        f_layer2, f_layer3, f_layer4 = f_dict["layer2"],  f_dict["layer3"], f_dict["layer4"]
        return self.head(f_gap), self.layer2_head(f_layer2), self.layer3_head(f_layer3), \
            self.layer4_head(f_layer4), f_layer2, f_layer3, f_layer4


class TeacherWrapper(nn.Module):
    def __init__(self, backbone, head, layer2_head, layer3_head, layer4_head, num_crops):
        super(TeacherWrapper, self).__init__()
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.backbone = tx.Extractor(self.backbone, ["layer2", "layer3", "layer4"])
        self.head = head
        self.layer2_head = layer2_head
        self.layer3_head = layer3_head
        self.layer4_head = layer4_head
        self.num_crops = num_crops

        self.roi_align_layer2 = ops.RoIAlign(output_size=5,
                                          sampling_ratio=-1,
                                          spatial_scale=0.125, #28/224
                                          aligned=False)
        self.roi_align_layer3 = ops.RoIAlign(output_size=5,
                                          sampling_ratio=-1,
                                          spatial_scale=0.0625, #14/224
                                          aligned=False)

    def forward(self, x, stu_f_layer2, stu_f_layer3, stu_f_layer4, box_generator):
        f_gap, f_dict = self.backbone(torch.cat(x[:]))
        f_gap = f_gap.squeeze()
        f_layer2, f_layer3, f_layer4 = f_dict["layer2"],  f_dict["layer3"], f_dict["layer4"]

        # cal local for teacher
        with torch.no_grad():
            stu_f_layer2 = torch.mean(stu_f_layer2, dim=(2,3), keepdim=True)
            stu_f_layer3 = torch.mean(stu_f_layer3, dim=(2,3), keepdim=True)
            stu_f_layer4 = torch.mean(stu_f_layer4, dim=(2,3), keepdim=True)
            stu_f_layer2 = F.normalize(stu_f_layer2, dim=1)
            stu_f_layer3 = F.normalize(stu_f_layer3, dim=1)
            stu_f_layer4 = F.normalize(stu_f_layer4, dim=1)
            stu_f_layer2 = stu_f_layer2.chunk(self.num_crops)
            stu_f_layer3 = stu_f_layer3.chunk(self.num_crops)
            stu_f_layer4 = stu_f_layer4.chunk(self.num_crops)

            f_layer2_nom = F.normalize(f_layer2, dim=1).chunk(2)
            f_layer3_nom = F.normalize(f_layer3, dim=1).chunk(2)
            f_layer4_nom = F.normalize(f_layer4, dim=1).chunk(2)

        local_layer2 = torch.empty(0).to(x[0].device)
        local_layer3 = torch.empty(0).to(x[0].device)
        local_layer4 = torch.empty(0).to(x[0].device)
        attention_map_l2_list = []
        attention_map_l3_list = []
        attention_map_l4_list = []
        for i in range(self.num_crops):
            with torch.no_grad():
                attention_map_l2 = torch.sum(f_layer2_nom[0]*stu_f_layer2[i], dim=1, keepdim=True)
                attention_map_l3 = torch.sum(f_layer3_nom[0]*stu_f_layer3[i], dim=1, keepdim=True)
                attention_map_l4 = torch.sum(f_layer4_nom[0]*stu_f_layer4[i], dim=1, keepdim=True)

            attention_map_l2_list.append(attention_map_l2.squeeze())
            attention_map_l3_list.append(attention_map_l3.squeeze())
            attention_map_l4_list.append(attention_map_l4.squeeze())

            # roi pooling
            local_out_l2 = attention_map_l2 * f_layer2.chunk(2)[0]
            local_out_l3 = attention_map_l3 * f_layer3.chunk(2)[0]
            local_out_l4 = attention_map_l4 * f_layer4.chunk(2)[0]

            local_layer2 = torch.cat((local_layer2, local_out_l2), dim=0)
            local_layer3 = torch.cat((local_layer3, local_out_l3), dim=0)
            local_layer4 = torch.cat((local_layer4, local_out_l4), dim=0)


        return self.head(f_gap), self.layer2_head(f_layer2), self.layer3_head(f_layer3), self.layer4_head(f_layer4), \
            self.layer2_head(local_layer2), self.layer3_head(local_layer3),self.layer4_head(local_layer4), \
            attention_map_l2_list, attention_map_l3_list, attention_map_l4_list
