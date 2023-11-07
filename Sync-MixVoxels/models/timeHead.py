import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .sh import eval_sh_bases
import numpy as np
import time
from functools import reduce


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def positional_encoding(positions, freqs):
    freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
    pts =  torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts

def time_positional_encoding(positions, freqs):
    positions = positions.to(device)
    freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(positions.shape[:-1] + (freqs * positions.shape[-1], ))
    pts =  torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts


class TimeMLP(torch.nn.Module):
    def __init__(self,n_layer, hidden_dim, in_dim, out_dim, using_view, activation=False, camoffset=None, args=None):
        super(TimeMLP, self).__init__()
        self.using_view = using_view
        self.cam_offset = camoffset
        self.hidden_dim = hidden_dim
        self.args=args

        layers = []
        for i in range(n_layer):
            if i == 0:      # first layer
                layer = nn.Linear(in_dim, hidden_dim)
            elif i != (n_layer - 1):     # hidden layer
                layer = nn.Linear(hidden_dim, hidden_dim)
            else:       # final layer
                layer = nn.Linear(hidden_dim, out_dim)

            torch.nn.init.constant(layer.bias, 0)
            if i < n_layer-1:
                torch.nn.init.xavier_uniform_(layer.weight, gain=np.sqrt(2))
            else:
                torch.nn.init.xavier_uniform_(layer.weight, gain=0.3)

            layers.append(layer)

            if i != n_layer -1 and activation=='relu':
                layers.append(torch.nn.ReLU(inplace=True))
            elif i != n_layer -1 and activation=='leakyrelu':
                layers.append(torch.nn.LeakyReLU(inplace=True))

        self.time_mlp = torch.nn.Sequential(*layers)

    def forward(self, time_input, cam_id, time_freq, total_time=None, iteration=None, test_cam_offset=0.):
        out_dim = total_time*3 if self.using_view else total_time
        time_normalized = (torch.Tensor(time_input) / (total_time-1)).to(device)
        time_normalized = time_normalized + test_cam_offset

        if self.args.cam_offset and (iteration >= self.args.offset_start_iters): 
            if (iteration == self.args.offset_start_iters) and not self.args.render_only:
                print(f'########### {iteration} iters - time offset learning started ###########')

            cam_offset = self.cam_offset[cam_id.long()].expand(time_input.shape)
            time_plusoffset = time_normalized + cam_offset
            time_encoded = time_positional_encoding(time_plusoffset[...,None], time_freq)

        # no offset
        else:
            time_encoded = time_positional_encoding(time_normalized[...,None], time_freq)

        time_encoded = time_encoded / time_encoded.norm(dim=-1, keepdim=True)
        time_embedding = self.time_mlp(time_encoded).reshape(-1, out_dim, self.hidden_dim)
        return time_embedding



def generate_temporal_mask(temporal_mask, n_frames=300, n_frame_for_static=2):
    """
    temporal_mask: Ns
        true for select all frames to train
        false for random select one (or fixed small numbers) frame to train
    """
    Ns = temporal_mask.shape[0]
    keep = torch.ones(Ns, n_frames, device=temporal_mask.device)
    drop = torch.zeros(Ns, n_frames, device=temporal_mask.device)
    drop[:, 0] = 1
    # for i_choice in range(n_frame_for_static):
    #     drop[np.arange(Ns), np.random.choice(n_frames, Ns, replace=True)] = 1
    detail_temporal_mask = torch.where(temporal_mask.unsqueeze(dim=1).expand(-1, n_frames), keep, drop).bool()
    return detail_temporal_mask



class TimeMLPRender(torch.nn.Module):
    def __init__(self, inChanel, args=None, viewpe=6, using_view=False, n_time_embedding=6, 
                 total_time=300, featureD=128, time_embedding_type='abs', net_spec='i-d-d-o', gain=1.0, camoffset=None):
        super(TimeMLPRender, self).__init__()

        self.args = args
        self.in_mlpC = inChanel + using_view * (3 + 2*viewpe*3)
        self.viewpe = viewpe
        self.n_time_embedding = n_time_embedding
        self.time_embedding_type = time_embedding_type
        self.using_view = using_view
        self.time_pos_encoding = torch.nn.Parameter(
            0.1 * torch.randn(total_time, n_time_embedding)
        )
        self.total_time = total_time
        self.gain = gain

        self.time_freq = args.time_freq
        self.n_layer = args.n_layer
        self.activation = args.activation

        self.hidden_dim = 512 #*2 if using_view else 512
        self.in_dim = self.time_freq * 2 
        self.out_dim = 512 * 3 if using_view else 512
        self.camoffset = camoffset
        
        layers = []
        _net_spec = net_spec.split('-')
        for i_mk, mk in enumerate(_net_spec):
            if mk == 'i':
                continue
            if mk == 'd' and _net_spec[i_mk-1] == 'i':
                layer = torch.nn.Linear(self.in_mlpC, featureD)
            if mk == 'd' and _net_spec[i_mk-1] == 'd':
                layer = torch.nn.Linear(featureD, featureD)
            if mk == 'o' and _net_spec[i_mk-1] == 'i':
                layer = torch.nn.Linear(self.in_mlpC, self.out_dim)
            if mk == 'o' and _net_spec[i_mk-1] == 'd':
                layer = torch.nn.Linear(featureD, self.out_dim)

            torch.nn.init.constant_(layer.bias, 0)
            torch.nn.init.xavier_uniform_(layer.weight, gain=(self.gain if not using_view else 1))

            layers.append(layer)
            if mk != 'o':
                layers.append(torch.nn.ReLU(inplace=True))

        self.mlp = torch.nn.Sequential(*layers[:-1])
        self.time_mlp = TimeMLP(self.n_layer, self.hidden_dim, self.in_dim, self.out_dim, using_view, self.activation, self.camoffset, self.args)


        # load pretrain mlp weights without time query
        if self.args.no_load_timequery:
            pretrain = torch.load(self.args.ckpt)
            print(f'- timehead ckpt path {self.args.ckpt}')

            if using_view:
                rendermodule = {}
                for key in pretrain['state_dict'].keys():
                    if key.startswith('renderModule.mlp.') and key[17] != '4':
                        rendermodule[key[17:]] = pretrain['state_dict'][key]
                                                            
                self.mlp.load_state_dict(rendermodule, strict=False)
                print('- load rgb weights in timeHead.py without time query')

            else: 
                renderdenmodule = {}
                for key in pretrain['state_dict'].keys():
                    if key.startswith('renderDenModule.mlp.') and key[20] != '4':
                        renderdenmodule[key[20:]] = pretrain['state_dict'][key]
                        
                self.mlp.load_state_dict(renderdenmodule, strict=False)
                print('- load rgb weights in timeHead.py without time query')


    def forward(self, features, time=None, viewdirs=None, cam_id=None, spatio_temporal_sigma_mask=None, temporal_mask=None, temporal_indices=None, iteration=None, t_inputs=None, test_cam_offset=0.):
        Ns = features.shape[0]
        num_frames = self.total_time
        indata = [features, ]
        if self.using_view:
            indata += [viewdirs,]
            indata += [positional_encoding(viewdirs, self.viewpe),]
        mlp_in = torch.cat(indata, dim=-1)
        mlp_output = self.mlp(mlp_in)

        if type(t_inputs) == type(None):
            t_inputs = np.arange(num_frames)
        t_inputs = np.broadcast_to(t_inputs, (Ns, num_frames)) 
        time_mlp_output = self.time_mlp(t_inputs, cam_id, time_freq=self.time_freq, total_time=num_frames, iteration=iteration, test_cam_offset=test_cam_offset)
        cam_offset =  self.time_mlp.cam_offset

        output = torch.bmm(mlp_output.unsqueeze(1), time_mlp_output.permute(0,2,1))
        output = output.squeeze(1)

        if temporal_indices is not None and len(temporal_indices.shape) == 1:
            output = output.reshape(Ns, self.total_time, -1)
            output = output[:, temporal_indices, :]
        else:
            output = output.reshape(Ns, num_frames, -1)

        if self.using_view:
            output = torch.sigmoid(output)

        output = output.squeeze(dim=-1)
        return output, cam_offset


class DirectDyRender(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, using_view=False, n_time_embedding=6, args=None,
                 total_time=300, featureD=128, time_embedding_type='abs', net_spec='i-d-d-o', gain=1.0):
        super(DirectDyRender, self).__init__()

        self.in_mlpC = inChanel + using_view * (3 + 2*viewpe*3)
        self.viewpe = viewpe
        self.n_time_embedding = n_time_embedding
        self.time_embedding_type = time_embedding_type
        self.using_view = using_view
        self.time_pos_encoding = torch.nn.Parameter(
            0.1 * torch.randn(total_time, n_time_embedding)
        )
        self.total_time = total_time

        self.out_dim = 3*total_time if using_view else total_time
        self.gain = gain
        self.args = args

        layers = []
        _net_spec = net_spec.split('-')
        for i_mk, mk in enumerate(_net_spec):
            if mk == 'i':
                continue
            if mk == 'd' and _net_spec[i_mk-1] == 'i':
                layer = torch.nn.Linear(self.in_mlpC, featureD)
            if mk == 'd' and _net_spec[i_mk-1] == 'd':
                layer = torch.nn.Linear(featureD, featureD)
            if mk == 'o' and _net_spec[i_mk-1] == 'i':
                layer = torch.nn.Linear(self.in_mlpC, self.out_dim)
            if mk == 'o' and _net_spec[i_mk-1] == 'd':
                layer = torch.nn.Linear(featureD, self.out_dim)
            torch.nn.init.constant_(layer.bias, 0)
            torch.nn.init.xavier_uniform_(layer.weight, gain=(self.gain if not using_view else 1))

            layers.append(layer)
            if mk != 'o':
                layers.append(torch.nn.ReLU(inplace=True))

        self.mlp = torch.nn.Sequential(*layers)

        # load pretrain mlp weights without time query
        if self.args.no_load_timequery:
            pretrain = torch.load(self.args.ckpt)
            print(f'- timehead ckpt path {self.args.ckpt}')

            if using_view: 
                rendermodule = {}
                for key in pretrain['state_dict'].keys():
                    if key.startswith('renderModule.mlp.') and key[17] != '4':
                        rendermodule[key[17:]] = pretrain['state_dict'][key]
                                                            
                self.mlp.load_state_dict(rendermodule, strict=False)
                print('- load rgb weights in timeHead.py without time query')

        
                for name, params in self.mlp.named_parameters():
                    if not name.startswith('4'):
                        print(f'mlp key {name} is loaded with pretrain weight and frozen')
                        params.requires_grad=False 
        
            else: 
                renderdenmodule = {}
                for key in pretrain['state_dict'].keys():
                    if key.startswith('renderDenModule.mlp.') and key[20] != '4':
                        renderdenmodule[key[20:]] = pretrain['state_dict'][key]
                        
                self.mlp.load_state_dict(renderdenmodule, strict=False)
                print('- load rgb weights in timeHead.py without time query')

                for name, params in self.mlp.named_parameters():
                    if not name.startswith('4'):
                        print(f'{name} is loaded with pretrain weight and frozen')
                        params.requires_grad=False 


    def forward(self, features, time=None, viewdirs=None, spatio_temporal_sigma_mask=None, temporal_mask=None, temporal_indices=None):
        # spatio_temporal_sigma_mask: for rgb branch prunning
        # temporal_mask is for re-sampling temporal sequence by variance of training pixels.
        Ns = features.shape[0]
        num_frames = self.total_time
        indata = [features, ]
        if self.using_view:
            indata += [viewdirs,]
            indata += [positional_encoding(viewdirs, self.viewpe),]

        mlp_in = torch.cat(indata, dim=-1)
        output = self.mlp(mlp_in)

        if temporal_indices is not None and len(temporal_indices.shape) == 1:
            output = output.reshape(Ns, self.total_time, -1)
            output = output[:, temporal_indices, :]
        else:
            output = output.reshape(Ns, num_frames, -1)

        if self.using_view:
            output = torch.sigmoid(output)

        output = output.squeeze(dim=-1)
        return output
    

class DyRender(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, using_view=False, n_time_embedding=6,
                 total_time=300, featureD=128, time_embedding_type='abs'):
        super(DyRender, self).__init__()

        self.in_mlpC = n_time_embedding + inChanel + using_view * (3 + 2*viewpe*3)
        self.viewpe = viewpe
        self.n_time_embedding = n_time_embedding
        self.time_embedding_type = time_embedding_type
        self.using_view = using_view
        self.time_pos_encoding = torch.nn.Parameter(
            0.1 * torch.randn(total_time, n_time_embedding)
        )
        self.total_time = total_time

        layer1 = torch.nn.Linear(self.in_mlpC, featureD)
        layer2 = torch.nn.Linear(featureD, featureD)
        self.out_dim = 3 if using_view else 1
        layer3 = torch.nn.Linear(featureD, self.out_dim)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)

        if using_view:
            torch.nn.init.constant_(self.mlp[-1].bias, 0)
        else:
            torch.nn.init.constant_(self.mlp[0].bias, 0)
            torch.nn.init.constant_(self.mlp[2].bias, 0)
            torch.nn.init.constant_(self.mlp[4].bias, 0)
            torch.nn.init.xavier_uniform(self.mlp[0].weight)
            torch.nn.init.xavier_uniform(self.mlp[2].weight)
            torch.nn.init.xavier_uniform(self.mlp[4].weight)

    def forward_with_time(self, features, time=None, viewdirs=None):
        Ns = features.shape[0]
        time_embedding = self.time_pos_encoding[time].unsqueeze(0).expand(Ns, -1)
        indata = [features, time_embedding]
        if self.using_view:
            indata += [viewdirs]
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        # mlp_in Ns x (ds + dt)
        output = self.mlp(mlp_in)
        if self.using_view:
            output = torch.sigmoid(output)
        output = output.squeeze(dim=-1)
        return output

    def forward(self, features, time=None, viewdirs=None, spatio_temporal_sigma_mask=None, temporal_mask=None, temporal_indices=None):
        # spatio_temporal_sigma_mask: for rgb branch prunning
        # temporal_mask is for re-sampling temporal sequence by variance of training pixels.
        Ns = features.shape[0]
        if temporal_indices is None:
            num_frames = self.total_time
            time_embedding = self.time_pos_encoding.unsqueeze(0).expand(Ns, -1, -1)
        elif len(temporal_indices.shape) == 1:
            num_frames = self.total_time if temporal_indices is None else len(temporal_indices)
            time_embedding = self.time_pos_encoding[temporal_indices].unsqueeze(0).expand(Ns, -1, -1)
        else:
            # temporal_indices Ns x T_train
            num_frames = temporal_indices.shape[1]
            time_embedding = (self.time_pos_encoding[temporal_indices.reshape(-1)]).reshape(Ns, num_frames, -1)
        features = features.unsqueeze(1).expand(-1, num_frames, -1)
        assert len(features.shape) == 3
        indata = [features, time_embedding]
        if self.using_view:
            indata += [viewdirs.unsqueeze(dim=1).expand(-1, num_frames, -1)]
            indata += [positional_encoding(viewdirs, self.viewpe).unsqueeze(dim=1).expand(-1, num_frames, -1)]

        mlp_in = torch.cat(indata, dim=-1)

        origin_output = torch.zeros(Ns, num_frames, self.out_dim).to(features)
        st_mask = torch.ones(Ns, num_frames).to(features).bool()
        if spatio_temporal_sigma_mask is not None:
            # mlp_in Ns x T x (ds + dt)
            # spatio_temporal_sigma_mask: Ns x T
            st_mask = st_mask & spatio_temporal_sigma_mask
        if temporal_mask is not None:
            st_mask = st_mask & temporal_mask
        mlp_in = mlp_in[st_mask]
        output = self.mlp(mlp_in)
        if self.using_view:
            output = torch.sigmoid(output)

        origin_output[st_mask] = output
        output = origin_output
        output = output.squeeze(dim=-1)
        return output


class ForrierDyRender(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, using_view=False, n_time_embedding=60,
                 total_time=300, featureD=128, time_embedding_type='abs'):
        super(ForrierDyRender, self).__init__()

        self.in_mlpC = inChanel + using_view * (3 + 2*viewpe*3)
        self.viewpe = viewpe
        self.n_time_embedding = n_time_embedding
        self.time_embedding_type = time_embedding_type
        self.using_view = using_view
        self.total_time = total_time

        layer1 = torch.nn.Linear(self.in_mlpC, featureD)
        layer2 = torch.nn.Linear(featureD, featureD)
        self.out_dim = 3*(2*n_time_embedding+1) if using_view else 1*(2*n_time_embedding+1)
        layer3 = torch.nn.Linear(featureD, self.out_dim)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)

        def forrier_basis(t, n_basis):
            ret = [1, ]
            for n in range(1, n_basis + 1):
                ret.append(math.cos(n * 2*math.pi * t/self.total_time))
                ret.append(math.sin(n * 2*math.pi * t/self.total_time))
            return ret

        # norm_time = lambda T: (T - self.total_time//2)/(self.total_time//2)
        self.forrier_basis = np.stack([forrier_basis(T, self.n_time_embedding) for T in range(self.total_time)], axis=1)
        self.forrier_basis = torch.from_numpy(self.forrier_basis).to(torch.float16).cuda().detach()

        if using_view:
            torch.nn.init.constant_(self.mlp[-1].bias, 0)
        else:
            torch.nn.init.constant_(self.mlp[0].bias, 0)
            torch.nn.init.constant_(self.mlp[2].bias, 0)
            torch.nn.init.constant_(self.mlp[4].bias, 0)
            torch.nn.init.xavier_uniform(self.mlp[0].weight)
            torch.nn.init.xavier_uniform(self.mlp[2].weight)
            torch.nn.init.xavier_uniform(self.mlp[4].weight)

    def forward(self, features, time=None, viewdirs=None, spatio_temporal_sigma_mask=None, temporal_mask=None, temporal_indices=None):
        # spatio_temporal_sigma_mask: for rgb branch prunning
        # temporal_mask is for re-sampling temporal sequence by variance of training pixels.
        Ns = features.shape[0]
        num_frames = self.total_time
        indata = [features, ]
        if self.using_view:
            indata += [viewdirs,]
            indata += [positional_encoding(viewdirs, self.viewpe),]
        mlp_in = torch.cat(indata, dim=-1)

        output = self.mlp(mlp_in)
        frequency_output = output.reshape(Ns, 3 if self.using_view else 1, 2*self.n_time_embedding+1).transpose(1,2)
        output = output.reshape(-1, 2*self.n_time_embedding+1)
        basis = self.forrier_basis
        if temporal_indices is not None and len(temporal_indices.shape) == 1:
            basis = self.forrier_basis[:, temporal_indices]
            num_frames = temporal_indices.shape[0]
            output = output @ basis
            output = output.reshape(Ns, -1, num_frames).transpose(1,2)

        if temporal_indices is not None and len(temporal_indices.shape) == 2:
            output = output @ basis
            output = output.reshape(Ns, -1, self.total_time).transpose(1,2)
            output = torch.gather(output, dim=1, index=temporal_indices.unsqueeze(dim=-1).expand(-1, -1, output.shape[-1]))
            # output Ns,

        if temporal_indices is None:
            output = output @ basis
            output = output.reshape(Ns, -1, self.total_time).transpose(1,2)

        if self.using_view:
            output = torch.sigmoid(output)

        output = output.squeeze(dim=-1)
        return output, 0 