import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch
from torch import nn


class CondNet(nn.Module):
    def __init__(self, cond_features, horizon):
        super().__init__()
        self.horizon = horizon

        self.condition = nn.Sequential(nn.Linear(cond_features,  8),
                                       nn.Tanh(),
                                       nn.Linear(8, 4),
                                       )

    def forward(self, conds):
        return self.condition(conds)


def subnet(ch_in, ch_out):
    return nn.Sequential(nn.Linear(ch_in, 32),
                         nn.Tanh(),
                         nn.Linear(32, ch_out))


class INN(nn.Module):
    def __init__(self, lr, cond_features, horizon, n_layers_cond=5, n_layers_without_cond=0, subnet=subnet):
        super().__init__()
        self.horizon = horizon
        if cond_features > 0:
            self.cond_net = CondNet(cond_features, horizon=self.horizon)
        else:
            self.cond_net = None

        self.no_layer_cond = n_layers_cond
        self.no_layer_without_cond = n_layers_without_cond
        self.subnet = subnet
        self.cinn = self.build_inn()

        self.trainable_parameters = [p for p in self.cinn.parameters() if p.requires_grad]
        for p in self.trainable_parameters:
            p.data = 0.01 * torch.randn_like(p)
        if self.cond_net:
            self.trainable_parameters += list(self.cond_net.parameters())

        self.optimizer = torch.optim.Adam(self.trainable_parameters, lr=lr, weight_decay=1e-5)

    def build_inn(self):
        nodes = [Ff.InputNode(self.horizon)]

        if self.cond_net:
            cond = Ff.ConditionNode(4)
            for k in range(self.no_layer_cond):
                nodes.append(
                    Ff.Node(nodes[-1], Fm.GLOWCouplingBlock, {"subnet_constructor": self.subnet, "clamp": 0.5},
                            conditions=cond))
                nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {"seed": k}))
            for k in range(self.no_layer_without_cond):
                nodes.append(
                    Ff.Node(nodes[-1], Fm.GLOWCouplingBlock, {"subnet_constructor": self.subnet}))
                nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {"seed": k}))
            return Ff.ReversibleGraphNet(nodes + [cond, Ff.OutputNode(nodes[-1])], verbose=False)
        else:
            for k in range(self.no_layer_cond):
                nodes.append(
                    Ff.Node(nodes[-1], Fm.GLOWCouplingBlock, {"subnet_constructor": self.subnet}))
                nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {"seed": k}))
            return Ff.ReversibleGraphNet(nodes + [Ff.OutputNode(nodes[-1])], verbose=False)

    def forward(self, x, conds, rev=False):
        c = self._calculate_condition(conds)
        z, jac = self.cinn(x.float(), c=c, rev=rev)
        return z, jac

    def _calculate_condition(self, conds):
        return self.cond_net(conds)#.reshape((-1, self.horizon))

    def reverse_sample(self, z, conds):
        c = self._calculate_condition(conds)
        return self.cinn(z, c=c, rev=True)[0].detach().numpy()
