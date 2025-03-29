import copy
import warnings
import math
import torch
from models_mae.MAE_ViT_MsLd import MAE_ViT_MsLd, MAE_ViT_MsLd_PAIRED
from models_mae.MLP import MLP
from util.contrast_loss import NTXentLoss
from torch.nn.parallel import DistributedDataParallel
import torch.nn as nn
class MAE_ViT_Ours(MAE_ViT_MsLd):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
            self,
            loss_cd=None,
            predictor_hidden_size=2048,
            **kwargs,
    ):
        super().__init__(**kwargs)
        # If None, use the same loss as for the reconstruction (decoder projection head)
        self.loss_cd = loss_cd.lower() if loss_cd is not None else self.loss
        # get the loss function from class based on string
        self.__forward_loss_cd = getattr(self, f"forward_loss_{self.loss_cd}")
        print(f"__forward_loss_cd: {self.loss_cd} -> {self.__forward_loss_cd.__name__}")

        self.predictor = MLP(
            self.decoder_embed_dim, self.num_patches, predictor_hidden_size
        )
        self.head=DINOHead(
            768,
            768,
            use_bn=False,
            norm_last_layer=True,
        )
    def forward_head(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.head(output)
    def forward(
            self,
            imgs,
            mask_ratio=0.75,
            contr_bs=None,
            mask_seed: int = None,
            return_embeds=False,
            consistent_mask=False,
            is_teacher=False,
            **kwargs,
    ):  
        if is_teacher:
            enc_emb_orig= super().forward(
            imgs,
            mask_ratio=mask_ratio,
            mask_seed=mask_seed,
            return_embeds=True,
            consistent_mask=consistent_mask,
            is_teacher=True
        )
            f1 = torch.flatten(enc_emb_orig[:, 1:, :].mean(dim=1), 1)
            # f2 = torch.flatten(enc_emb_crop[:, 1:, :].mean(dim=1), 1)
            p1 = self.head(f1)
            # p2 = self.head(f2)


            return p1, f1
            # return f1
        (
            loss_d,
            pred_orig,
            mask_orig,
            (enc_emb_orig, enc_emb_crop),
            (dec_emb_orig, dec_emb_crop),
        # (
        #     loss_d,
        #     enc_emb_crop,
        #     dec_emb_crop,
        ) = super().forward(
            imgs,
            mask_ratio=mask_ratio,
            mask_seed=mask_seed,
            return_embeds=True,
            consistent_mask=consistent_mask,
        )

        if contr_bs:
            bs = contr_bs
        else:
            bs = imgs.shape[0]

        # Cross decoder loss between original and crop
        cross_pred = self.predictor(dec_emb_crop[:, 1:, :])
        cross_target = dec_emb_orig[:, 1:, :]
        loss_cd = self.__forward_loss_cd(cross_target, cross_pred)

        f1 = torch.flatten(enc_emb_orig[:, 1:, :].mean(dim=1), 1)
        f2 = torch.flatten(enc_emb_crop[:, 1:, :].mean(dim=1), 1)
        p1 = self.head(f1)
        p2 = self.head(f2)


        loss_d_cd_ce = loss_d + loss_cd
        # loss_d_cd_ce = loss_d 
        if not return_embeds:
            return loss_d_cd_ce, p1, p2, f1, f2

        return (
            loss_d_cd_ce,
            pred_orig,
            mask_orig,
            (enc_emb_orig, enc_emb_crop),
            (dec_emb_orig, dec_emb_crop),
        )
         

class MAE_ViT_MsLdCeCd_PAIRED(MAE_ViT_MsLd_PAIRED):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
            self,
            device="cuda:0",
            loss_cd=None,
            # bacth_size =128,
            predictor_hidden_size=2048,
            **kwargs,
    ):
        super().__init__(**kwargs)
        # If None, use the same loss as for the reconstruction (decoder projection head)
        self.loss_cd = loss_cd.lower() if loss_cd is not None else self.loss
        # get the loss function from class based on string
        self.__forward_loss_cd = getattr(self, f"forward_loss_{self.loss_cd}")
        print(f"__forward_loss_cd: {self.loss_cd} -> {self.__forward_loss_cd.__name__}")

        # self.batch_size = bacth_size
        self.device = device

        self.predictor = MLP(
            self.decoder_embed_dim, self.num_patches, predictor_hidden_size
        )


        # self.contrast_criterian = NTXentLoss(self.batch_size, self.device, 0.5, cos_sim=True)


    def forward(
            self,
            imgs1,
            imgs2,
            mask_ratio=0.75,
            contr_bs=None,
            mask_seed: int = None,
            return_embeds=True,
            consistent_mask=False,
            **kwargs,
    ):
        (
            loss_d,
            pred_orig,
            mask_orig,
            (enc_emb_orig, enc_emb_crop),
            (dec_emb_orig, dec_emb_crop),
        ) = super().forward(
            imgs1,
            imgs2,
            mask_ratio=mask_ratio,
            mask_seed=mask_seed,
            return_embeds=True,
            consistent_mask=consistent_mask,
        )

        if contr_bs:
            bs = contr_bs
        else:
            bs = imgs1.shape[0]

        # Cross decoder loss between original and crop
        cross_pred = self.predictor(dec_emb_crop[:, 1:, :])
        cross_target = dec_emb_orig[:, 1:, :]
        loss_cd = self.__forward_loss_cd(cross_target, cross_pred)

        # Contrastive encoder loss between original and crop

        contrast_criterian = NTXentLoss(self.device, bs, 0.5, cos_sim=True)

        f1 = torch.flatten(enc_emb_orig[:, 1:, :].mean(dim=1), 1)
        f2 = torch.flatten(enc_emb_crop[:, 1:, :].mean(dim=1), 1)
        # print('f1 shape:',f1.shape)
        # print('f2 shape:',f2.shape)

        loss_ce = contrast_criterian(f1, f2)

        #Reconstruction loss + cross decoder loss
        loss_d_cd_ce = loss_d + loss_cd + loss_ce
        # loss_d_cd_ce = loss_d + loss_cd

        if not return_embeds:
            return loss_d_cd_ce, pred_orig, mask_orig

        return (
            loss_d_cd_ce,
            pred_orig,
            mask_orig,
            (enc_emb_orig, enc_emb_crop),
            (dec_emb_orig, dec_emb_crop),
        )



def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x