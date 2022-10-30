from AI.ML.ML_Loss import *
from AI.ML.ML_Optimizer import *


def test_init_parameters(model):
    model.init_parameters()
    model.init_optimizers()
    model.history = []


def test_init_loss(model, k):
    if k == 1:
        model.f_loss = loss_abs
        model.f_loss_d = loss_abs_
    elif k == 2:
        model.f_loss = loss_square
        model.f_loss_d = loss_square_
    elif k == 3:
        model.f_loss = loss_log
        model.f_loss_d = loss_log_


def test_init_optimizer(model, k):
    if k == 1:
        model.c_optimizer_W = GD(.1)
        model.c_optimizer_b = GD(.1)
    elif k == 2:
        model.c_optimizer_W = Momentum(0.1)
        model.c_optimizer_b = Momentum(0.1)
    elif k == 3:
        model.c_optimizer_W = RMSProp(0.1)
        model.c_optimizer_b = RMSProp(0.1)
    elif k == 4:
        model.c_optimizer_W = AdaM(0.1)
        model.c_optimizer_b = AdaM(0.1)

    elif k == 0:
        model.c_optimizer_W = OptT(.1, 1.)
        model.c_optimizer_b = OptT(.1, 1.)
