import os
import torch

from tqdm import tqdm

from os.path import join as pjoin
from options.vq_option import arg_parse

import utils.paramUtil as paramUtil
# from options.evaluate_options import TestT2MOptions
from utils.plot_script import *


# from networks.transformer import TransformerV1, TransformerV2
# from networks.quantizer import *
# from networks.modules import *
# from networks.trainers import TransformerT2MTrainer
# from data.dataset import Motion2TextEvalDataset
# from scripts.motion_process import *


from torch.utils.data import DataLoader
from utils.word_vectorizer import WordVectorizerV2
from utils.utils import *
from utils.motion_process import recover_from_ric
from models.vq.model import RVQVAE
from vector_quantize_pytorch import ResidualVQ,ResidualLFQ,LFQ,VectorQuantize,GroupedResidualVQ,RandomProjectionQuantizer

from utils.get_opt import get_opt


def plot_t2m(data, count):
    data = data * std + mean
    for i in range(len(data)):
        joint_data = data[i]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), args.joints_num).numpy()
        # joint = motion_temporal_filter(joint)
        # save_path = '%s_%02d.mp4' % (save_dir, i)
        np.save(pjoin(args.joint_dir, "%d.npy"%count), joint)
        plot_3d_motion(pjoin(args.animation_dir, "%d.mp4"%count),
                       kinematic_chain, joint, title=f"token_{count}", fps=fps, radius=radius)



def load_vq_model(vq_opt, which_epoch):
    # opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    # if args.arch_option == 'residual_vq':
    vq_model = RVQVAE(vq_opt,
                dim_pose,
                vq_opt.nb_code,
                vq_opt.code_dim,
                vq_opt.code_dim,
                vq_opt.down_t,
                vq_opt.stride_t,
                vq_opt.width,
                vq_opt.depth,
                vq_opt.dilation_growth_rate,
                vq_opt.vq_act,
                vq_opt.vq_norm)
  
    
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', which_epoch),
                            map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    vq_epoch = ckpt['ep'] if 'ep' in ckpt else -1
    print(f'Loading VQ Model {vq_opt.name} Completed!, Epoch {vq_epoch}')
    return vq_model, vq_epoch

if __name__ == '__main__':
    # parser = TestT2MOptions()
    # opt = parser.parse()
    args = arg_parse(False)
    args.device = torch.device("cpu" if args.gpu_id == -1 else "cuda:" + str(args.gpu_id))


    # opt.device = torch.device("cpu" if opt.gpu_id==-1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)
    # if opt.gpu_id != -1:
    #     torch.cuda.set_device(opt.gpu_id)

    args.result_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'eval')
    
    # args.result_dir = pjoin(args.result_path, args.dataset_name, args.name, args.ext)
    args.joint_dir = pjoin(args.result_dir, 'joints')
    args.animation_dir = pjoin(args.result_dir, 'animations')

    os.makedirs(args.joint_dir, exist_ok=True)
    os.makedirs(args.animation_dir, exist_ok=True)

    if args.dataset_name == 't2m':
        args.joints_num = 22
        args.max_motion_token = 55
        args.max_motion_frame = 196
        dim_pose = 263
        radius = 4
        fps = 20
        kinematic_chain = paramUtil.t2m_kinematic_chain
    elif args.dataset_name == 'kit':
        args.joints_num = 21
        radius = 240 * 8
        fps = 12.5
        dim_pose = 251
        args.max_motion_token = 55
        args.max_motion_frame = 196
        kinematic_chain = paramUtil.kit_kinematic_chain
    else:
        raise KeyError('Dataset Does Not Exist')

    mean = np.load(pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'meta', 'mean.npy'))
    std = np.load(pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'meta', 'std.npy'))

    enc_channels = [args.nb_code, args.code_dim]
    dec_channels = [args.code_dim, args.nb_code, dim_pose]
    print('donee')
    
        ##### ---- Network ---- #####
    vq_opt_path = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=args.device)
    
    if args.vq_arch_option == 'residual_vq':
        file = "/home/dipcik/PycharmPhd/motion-vqvae/checkpoints/t2m/residual_vq_replicate/model/E0034.tar"
    elif args.vq_arch_option == 'residual_lfq':
        file = "/home/dipcik/PycharmPhd/motion-vqvae/checkpoints/t2m/residual_lfq/model/E0024.tar"
       
    # file = "/home/dipcik/PycharmPhd/motion-vqvae/checkpoints/t2m/residual_vq_replicate/model/E0034.tar" 
    net, ep = load_vq_model(vq_opt, file)
    
    net.eval()
    net.cuda()
    

    vq_decoder = net.decoder
    quantizer = net.quantizer
  

    vq_decoder.eval()
    quantizer.eval()
    
    if args.vq_arch_option == 'residual_vq':
        with torch.no_grad():
            for i in tqdm(range(512)):
                m_token = torch.LongTensor(1, 1).fill_(i).to(args.device)
                # vq_latent = quantizer.get_codebook_entry(m_token)
                vq_latent = quantizer.get_codes_from_indices(m_token)
                vq_latent_ = vq_latent.sum(dim=0).unsqueeze(-1)
                gen_motion = vq_decoder(vq_latent_)
                
                plot_t2m(gen_motion.cpu().numpy(), i)
                print(f'done {i}')
                
    elif args.vq_arch_option == 'residual_lfq':
        
        with torch.no_grad():
            quantizer.quantize_dropout=0.1
            for i in tqdm(range(4096)):
                # i = 320
                m_token = torch.LongTensor(1, 1).fill_(i).to(args.device)
                # vq_latent = quantizer.get_codebook_entry(m_token)
                vq_latent = quantizer.get_output_from_indices(m_token)
                # print(vq_latent.shape)
                vq_latent_ = vq_latent.sum(dim=0).unsqueeze(-1).unsqueeze(0)
                gen_motion = vq_decoder(vq_latent_)
                
                plot_t2m(gen_motion.cpu().numpy(), i)
                print(f'done {i}')
