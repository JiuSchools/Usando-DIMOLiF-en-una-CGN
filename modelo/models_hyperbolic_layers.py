# -*- coding: utf-8 -*-
"""models/hyperbolic_layers.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1yIAV_xFoPL6VOzJqbOYHPhegSujnIJ9x
"""

import torch
from torch import nn
import geoopt

def create_ball(ball=None, c=None):
    if ball is None:
        assert c is not None
        ball = geoopt.PoincareBall(c)
    return ball

def mobius_linear(input, weight, bias=None, nonlin=None, *, ball: geoopt.PoincareBall):
    output = ball.mobius_matvec(weight, input)
    if bias is not None:
        output = ball.mobius_add(output, bias)
    if nonlin is not None:
        output = ball.logmap0(output)
        output = nonlin(output)
        output = ball.expmap0(output)
    return output

class MobiusLinear(nn.Linear):
    def __init__(self, *args, nonlin=None, ball=None, c=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ball = create_ball(ball, c)
        if self.bias is not None:
            self.bias = geoopt.ManifoldParameter(self.bias, manifold=self.ball)
        self.nonlin = nonlin
        self.reset_parameters()

    def forward(self, input):
        return mobius_linear(input, self.weight, self.bias, self.nonlin, ball=self.ball)

    @torch.no_grad()
    def reset_parameters(self):
        torch.nn.init.eye_(self.weight)
        self.weight.add_(torch.rand_like(self.weight).mul_(1e-3))
        if self.bias is not None:
            self.bias.zero_()