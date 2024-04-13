import random

import torch.nn as nn
from models.vq.encdec import Encoder, Decoder
# from models.vq.residual_vq import ResidualVQ

from vector_quantize_pytorch import ResidualVQ,ResidualLFQ,LFQ,VectorQuantize,GroupedResidualVQ,RandomProjectionQuantizer

    
class RVQVAE(nn.Module):
    def __init__(self,
                 args,
                 input_width=263,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):

        super().__init__()
        assert output_emb_width == code_dim
        self.code_dim = code_dim
        self.num_code = nb_code
        # self.quant = args.quantizer
        self.encoder = Encoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        self.decoder = Decoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        # rvqvae_config = {
        #     'dim':code_dim, 
        #     'num_quantizers': args.num_quantizers,
        #     'shared_codebook': args.shared_codebook,
        #     'quantize_dropout_prob': args.quantize_dropout_prob,
        #     'quantize_dropout_cutoff_index': 0,
        #     'nb_code': nb_code,
        #     'code_dim':code_dim, 
        #     'args': args,
        # }
        rvqvae_config = {
            'dim':code_dim, 
            'num_quantizers': args.num_quantizers, 
            'codebook_size':nb_code, 
         
            'shared_codebook': args.shared_codebook,
            
            'quantize_dropout': args.quantize_dropout_prob > 0,
            'quantize_dropout_cutoff_index': 0,
           
          
        }
        self.quantizer = ResidualVQ(**rvqvae_config)

        self.quantizer =  VectorQuantize(
            dim = code_dim,
            use_cosine_sim = True ,
            codebook_size = nb_code,     # codebook size
            decay = 0.8,             # the exponential moving average decay, lower means the dictionary will change faster
            commitment_weight = 1.   # the weight on the commitment loss
        )
        # Orthogonal regularization loss
        # self.quantizer = VectorQuantize(
        #     dim = code_dim,
        #     codebook_size = nb_code,  
        #     # accept_image_fmap = True,                   # set this true to be able to pass in an image feature map
        #     orthogonal_reg_weight = 10,                 # in paper, they recommended a value of 10
        #     orthogonal_reg_max_codes = 128,             # this would randomly sample from the codebook for the orthogonal regularization loss, for limiting memory usage
        #     orthogonal_reg_active_codes_only = False    # set this to True if you have a very large codebook, and would only like to enforce the loss on the activated codes per batch
        # )

        # Multi-headed VQ
        # self.quantizer  = VectorQuantize( 
        #     dim = code_dim,
        #     codebook_size = 8196,                   # a number of papers have shown smaller codebook dimension to be acceptable
        #     heads = 8,                          # number of heads to vector quantize, codebook shared across all heads
        #     separate_codebook_per_head = True,  # whether to have a separate codebook per head. False would mean 1 shared codebook
            
        # )

        # lfq
        # self.quantizer = ResidualLFQ(
        #                     dim=512,#quantize_dim
        #                     codebook_size = 2**14, # 2**16
        #                     num_quantizers = 6 #,
        #                     # **vq_kwargs
        #                 )
        self.quantizer = LFQ(dim=code_dim, codebook_size=nb_code)###denemece 



        # self.quantizer = GroupedResidualVQ(
        #     dim = code_dim,
        #     num_quantizers = 8,      # specify number of quantizers
        #     groups = 2,
        #     codebook_size = 1024,    # codebook size
        # )


    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        # print(x_encoder.shape)
        code_idx, all_codes = self.quantizer.quantize(x_encoder, return_latent=True)
        # print(code_idx.shape)
        # code_idx = code_idx.view(N, -1)
        # (N, T, Q)
        # print()
        return code_idx, all_codes

    def forward(self, x):
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)

        #new  ht 
        x_encoder = self.preprocess(x_encoder)

        ## quantization
        # x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5,
        #                                                                 force_dropout_index=0) #TODO hardcode
        # x_quantized, code_idx, all_loss = self.quantizer(x_encoder, sample_codebook_temp=0.5)# for residual vq
        x_quantized, code_idx, all_loss = self.quantizer(x_encoder)
# 
        #new  ht 
        x_quantized = self.preprocess(x_quantized)

        # print(code_idx[0, :, 1])
        ## decoder
        x_out = self.decoder(x_quantized)
        # x_out = self.postprocess(x_decoder)
        return x_out, all_loss

    def forward_decoder(self, x):
        x_d = self.quantizer.get_codes_from_indices(x)
        # x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        x = x_d.sum(dim=0).permute(0, 2, 1)

        # decoder
        x_out = self.decoder(x)
        # x_out = self.postprocess(x_decoder)
        return x_out

class LengthEstimator(nn.Module):
    def __init__(self, input_size, output_size):
        super(LengthEstimator, self).__init__()
        nd = 512
        self.output = nn.Sequential(
            nn.Linear(input_size, nd),
            nn.LayerNorm(nd),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(0.2),
            nn.Linear(nd, nd // 2),
            nn.LayerNorm(nd // 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(0.2),
            nn.Linear(nd // 2, nd // 4),
            nn.LayerNorm(nd // 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(nd // 4, output_size)
        )

        self.output.apply(self.__init_weights)

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, text_emb):
        return self.output(text_emb)