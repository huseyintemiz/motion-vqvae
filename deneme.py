# add main function for evaluation
from models.vq.model import RVQVAE
import pydantic

from pydantic import BaseModel

class ArgsModel(BaseModel):
    num_quantizers: int
    vq_group: int
    vq_arch_option: str


if __name__ == '__main__':
    print('deneme')
    import torch
    # self.quantizer = GroupedResidualVQ(
    #             dim = code_dim,
    #             num_quantizers = args.num_quantizers,      # specify number of quantizers
    #             groups = args.vq_group, #2,
    #             codebook_size = nb_code,    # codebook size
    #         )
    args = {'num_quantizers': 8, 'vq_group': 2, 'vq_arch_option': 'group_residual_vq'}
    
    # args.num_quantizers = 8
    # args.vq_group = 2
    # args.vq_arch_option = 'group_residual_vq'
    # Create an instance of ArgsModel using the args dictionary
    args_model = ArgsModel(**args)

    
    RVQVAE(args=args_model,  nb_code=1024, code_dim=512) 
    
    motion = torch.rand(32, 196, 263)
    