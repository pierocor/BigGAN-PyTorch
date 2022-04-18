import torch
import torch.nn.functional as F

# DCGAN loss
def loss_dcgan_dis(dis_fake, dis_real):
  L1 = torch.mean(F.softplus(-dis_real))
  L2 = torch.mean(F.softplus(dis_fake))
  return L1, L2


def loss_dcgan_gen(dis_fake):
  loss = torch.mean(F.softplus(-dis_fake))
  return loss


# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real, gy, dy, class_weights):
  gy_weighting = torch.index_select(class_weights, 0, gy)
  dy_weighting = torch.index_select(class_weights, 0, dy)
  loss_real = torch.mean(dy_weighting * F.relu(1. - dis_real))
  loss_fake = torch.mean(gy_weighting * F.relu(1. + dis_fake))
  return loss_real, loss_fake
# def loss_hinge_dis(dis_fake, dis_real): # This version returns a single loss
  # loss = torch.mean(F.relu(1. - dis_real))
  # loss += torch.mean(F.relu(1. + dis_fake))
  # return loss


def loss_hinge_gen(dis_fake, y, class_weights):
  y_weighting = torch.index_select(class_weights, 0, y)
  loss = -torch.mean(y_weighting * dis_fake)
  return loss

# Default to hinge loss
generator_loss = loss_hinge_gen
discriminator_loss = loss_hinge_dis