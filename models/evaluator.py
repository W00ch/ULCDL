import torch.nn as nn
from .fusion import Bottleneck
import torch


class MLP_block(nn.Module):

    def __init__(self, feature_dim, output_dim, dropout, attention_config):
        super(MLP_block, self).__init__()
        self.feature_dim = feature_dim
        self.activation = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.layer1 = nn.Linear(feature_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.layer2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.layer3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.layer4 = nn.Linear(64, output_dim)
        self.drop = nn.Dropout(dropout)
        self.layer5 = nn.Linear(64, 24)
        self.attention = Bottleneck(inplanes=attention_config['INPUT_DIM'], 
                                    planes=attention_config['HIDDEN_DIM'],
                                    base_width=attention_config['BASE_WIDTH'],
                                    fuse_type=attention_config['FUSE_TYPE'])

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.attention(x).view(B, -1)
        x = self.activation(self.bn1(self.layer1(x)))
        x = self.activation(self.bn2(self.layer2(x)))
        x = self.activation(self.bn3(self.layer3(x)))
        output = self.softmax(self.layer4(x))
        out_score = self.sigmoid(self.layer5(x))

        return output, out_score


class Evaluator(nn.Module):

    def __init__(self, feature_dim, output_dim, predict_type, dropout, attention_config, num_subscores=None):
        super(Evaluator, self).__init__()
        self.predict_type = predict_type
        if self.predict_type == 'subscores':
            self.evaluator = nn.ModuleList([MLP_block(feature_dim, output_dim, dropout, attention_config) for _ in range(num_subscores)])

        else:
            self.evaluator = MLP_block(feature_dim, output_dim, attention_config)

    def forward(self, feats_avg):
        out_probs, out_scores = [], []
        if self.predict_type == 'subscores':
            probs = [evaluator(feats_avg) for evaluator in self.evaluator]
        
        else:
            probs = self.evaluator(feats_avg)
        for i in range(len(probs)):
            out_probs.append(probs[i][0])
            out_scores.append(probs[i][1])
        # out_probs = torch.stack(out_probs).permute(1, 0, 2).mean(dim=1)
        out_scores = torch.stack(out_scores).permute(1, 0, 2).mean(dim=1)
        out = [out_probs, out_scores]

        return out
