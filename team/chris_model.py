import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np

from kuma_utils.torch.hooks import SimpleHook


class Modelv16(nn.Module):
    def __init__(self, name, *, pretrained=False):
        """
        name (str): timm model name, e.g. tf_efficientnet_b2_ns
        """
        super().__init__()

        self.premodel = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=(3,31), stride=(1,2), padding=(3//2,31//2)),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=(5,5), stride=(1,2), padding=(5//2,5//2)),
            nn.GELU(),
        )
        torch.nn.init.xavier_uniform_(self.premodel[0].weight)
        torch.nn.init.xavier_uniform_(self.premodel[2].weight)
#         torch.nn.init.xavier_uniform_(self.premodel[4].weight)
        
        # Use timm
        model = timm.create_model(name, pretrained=pretrained, \
                                  in_chans=128, drop_rate=0.0)

        clsf = model.default_cfg['classifier']
        n_features = model._modules[clsf].in_features
        model._modules[clsf] = nn.Identity()

        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.2)
        self.drop3 = nn.Dropout(0.3)
        self.drop4 = nn.Dropout(0.4)
        self.drop5 = nn.Dropout(0.5)
        self.drop6 = nn.Dropout(0.1)
        self.drop7 = nn.Dropout(0.2)
        self.drop8 = nn.Dropout(0.3)
        self.drop9 = nn.Dropout(0.4)
        self.drop10 = nn.Dropout(0.5)
        
        self.fc = nn.Sequential(
            nn.Linear(n_features+3, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 2)
        )
        self.model = model
        
        torch.nn.init.xavier_uniform_(self.fc[0].weight)
        torch.nn.init.xavier_uniform_(self.fc[2].weight)
        torch.nn.init.xavier_uniform_(self.fc[4].weight)
        
    def forward(self, x, freq, mean0, std0, mean1, std1):
        
        x = self.premodel(x)
        x = self.model(x)
        
        x1 = self.drop1(x)
        x2 = self.drop2(x)
        x3 = self.drop3(x)
        x4 = self.drop4(x)
        x5 = self.drop5(x)
        x6 = self.drop6(x)
        x7 = self.drop7(x)
        x8 = self.drop8(x)
        x9 = self.drop9(x)
        x10 = self.drop10(x)

        x1 = torch.cat((x1, freq/500.0, std0, std1), dim=1)
        x2 = torch.cat((x2, freq/500.0, std0, std1), dim=1)
        x3 = torch.cat((x3, freq/500.0, std0, std1), dim=1)
        x4 = torch.cat((x4, freq/500.0, std0, std1), dim=1)
        x5 = torch.cat((x5, freq/500.0, std0, std1), dim=1)
        x6 = torch.cat((x6, freq/500.0, std0, std1), dim=1)
        x7 = torch.cat((x7, freq/500.0, std0, std1), dim=1)
        x8 = torch.cat((x8, freq/500.0, std0, std1), dim=1)
        x9 = torch.cat((x9, freq/500.0, std0, std1), dim=1)
        x10 = torch.cat((x10, freq/500.0, std0, std1), dim=1)
        
        x1 = self.fc(x1)
        x2 = self.fc(x2)
        x3 = self.fc(x3)
        x4 = self.fc(x4)
        x5 = self.fc(x5)
        x6 = self.fc(x6)
        x7 = self.fc(x7)
        x8 = self.fc(x8)
        x9 = self.fc(x9)
        x10 = self.fc(x10)

        return (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)


class ChrisLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs[:,0].view(-1), targets.view(-1))
        mse_loss = F.mse_loss(inputs[:,1].view(-1).sigmoid(), targets.view(-1))
        return bce_loss + mse_loss

    def __repr__(self):
        return f'ChrisLoss()'


class ChrisTrain(SimpleHook):

    def __init__(self, evaluate_in_batch=False,):
        super().__init__(evaluate_in_batch=evaluate_in_batch)

    def forward_train(self, trainer, inputs):
        target = inputs[-1]  
        y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6, y_pred7, y_pred8, y_pred9, y_pred10 = trainer.model(*inputs[:-1])
        approx = (y_pred1 + y_pred2 + y_pred3 + y_pred4 + y_pred5 + y_pred6 + y_pred7 + y_pred8 + y_pred9 + y_pred10) / 10.0
        loss = trainer.criterion(approx, target)
        return loss, approx[:, 0].view(-1, 1).detach() # drop mse output

    def forward_valid(self, trainer, inputs):
        return self.forward_train(trainer, inputs)
    
    def forward_test(self, trainer, inputs):
        y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6, y_pred7, y_pred8, y_pred9, y_pred10 = trainer.model(*inputs[:-1])
        approx = (y_pred1 + y_pred2 + y_pred3 + y_pred4 + y_pred5 + y_pred6 + y_pred7 + y_pred8 + y_pred9 + y_pred10) / 10.0
        return approx[:, 0].view(-1, 1)

    def __repr__(self) -> str:
        return f'ChrisTrain()'


def forward_test_chris(model, specs, inputs):
    y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6, y_pred7, y_pred8, y_pred9, y_pred10 = model(specs, *inputs[1:-1])
    approx = (y_pred1 + y_pred2 + y_pred3 + y_pred4 + y_pred5 + y_pred6 + y_pred7 + y_pred8 + y_pred9 + y_pred10) / 10.0
    return approx[:, 0].view(-1, 1)
