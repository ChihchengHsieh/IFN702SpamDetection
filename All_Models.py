import torch
import torch.nn as nn
import gensim
from Constants import Constants
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


# Load Google's pre-trained Word2Vec model.
# model = gensim.models.KeyedVectors.load_word2vec_format(
#     './GoogleNews-vectors-negative300.bin', binary=True)

# word2vector = torch.FloatTensor(model.vectors)

class SSCL(nn.Module):

    ''' The Model from paper '''

    def __init__(self, args):
        super(SSCL, self).__init__()

        self.embed = nn.Embedding(
            args.vocab_size, args.embedding_dim, Constants.PAD)

        self.cnn = nn.Sequential(
            nn.Conv1d(args.embedding_dim, args.num_CNN_filter,
                      args.CNN_kernel_size, 1, 2),
            nn.BatchNorm1d(args.num_CNN_filter),
            nn.LeakyReLU(inplace=True),
        )

        self.rnn = nn.LSTM(args.num_CNN_filter, args.RNN_hidden,
                           batch_first=True, dropout=args.LSTM_dropout, num_layers=args.num_LSTM_layers)

        self.out_net = nn.Sequential(
            nn.Linear(args.RNN_hidden, 1),
        )

        self.h0 = nn.Parameter(torch.randn(1, args.RNN_hidden))
        self.c0 = nn.Parameter(torch.randn(1, args.RNN_hidden))

#         self.apply(self.weight_init)

    def forward(self, input, lengths=None):

        B = input.size(0)

        emb_out = self.embed(input).transpose(1, 2)

        out = self.cnn(emb_out).transpose(1, 2)

        if not lengths is None:
            out = pack_padded_sequence(out, lengths, batch_first=True)
            out, hidden = self.rnn(
                out, (self.h0.repeat(1, B, 1), self.c0.repeat(1, B, 1)))
            out = pad_packed_sequence(out, batch_first=True)[0][:, -1, :]
        else:
            #             out = self.rnn(out,(self.h0.repeat(1,B,1), self.c0.repeat(1,B,1)))[0][:, -1, :]
            # out = self.rnn(out)[0][:, -1, :]
            out = self.rnn(out)[0].sum(dim=1)

        out = self.out_net(out)

        return out

    def weight_init(self, m):

        if type(m) in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear]:
            nn.init.kaiming_normal_(m.weight, 0.2, nonlinearity='leaky_relu')
        elif type(m) in [nn.LSTM]:
            for name, value in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(value.data)
                if 'bias'in name:
                    value.data.normal_()



class GatedCNN(nn.Module):

    def __init__(self, args):
        super(GatedCNN, self).__init__()

        self.emb = nn.Embedding(args.vocab_size, args.GatedCNN_embedingDim)

        self.conv = nn.ModuleList([nn.Conv1d(args.GatedCNN_embedingDim, args.GatedCNN_convDim, args.GatedCNN_kernel, args.GatedCNN_stride, args.GatedCNN_pad)])
        self.conv.extend([nn.Conv1d(args.GatedCNN_convDim, args.GatedCNN_convDim, args.GatedCNN_kernel, args.GatedCNN_stride, args.GatedCNN_pad ) for _ in range(args.GatedCNN_layers)])
        self.conv_gate = nn.ModuleList([nn.Conv1d(args.GatedCNN_embedingDim, args.GatedCNN_convDim, args.GatedCNN_kernel, args.GatedCNN_stride, args.GatedCNN_pad )])        
        self.conv_gate.extend([nn.Conv1d(args.GatedCNN_convDim, args.GatedCNN_convDim, args.GatedCNN_kernel, args.GatedCNN_stride, args.GatedCNN_pad ) for _ in range(args.GatedCNN_layers)])

        self.fc = nn.Sequential(
            nn.Linear(args.GatedCNN_convDim, 1),
        ) 
        

    def forward(self, input, lengths = None):

        out_ = self.emb(input).transpose(1, 2)

        for i,(conv, conv_gate) in enumerate(zip(self.conv, self.conv_gate)):
            out_temp = out_ # Prepare for Residule
            out_a = conv(out_)
            out_b = conv_gate(out_)
            out_ = out_a * F.sigmoid(out_b)
            if out_temp.size()[1] ==  out_.size()[1]:
                out_ += out_temp # Residule

        out_ = out_.sum(dim=-1)

        out_ = self.fc(out_)

        return out_

    def weight_init(self, m):

        if type(m) in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.Conv1d]:
            nn.init.kaiming_normal_(m.weight, 0.2, nonlinearity='leaky_relu')
        elif type(m) in [nn.LSTM]:
            for name, value in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(value.data)
                if 'bias'in name:
                    value.data.normal_()



