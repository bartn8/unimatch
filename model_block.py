import numpy as np
import cv2
import torch
import torch.nn.functional as F
import gc

import importlib
unimatch = importlib.import_module("thirdparty.unimatch.unimatch.unimatch")
#dataloader_stereo = importlib.import_module("thirdparty.unimatch.dataloader.stereo")
utils = importlib.import_module("thirdparty.unimatch.utils.utils")


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class UniblockStereoParams:
    def __init__(self, inference_size = [1024,1536]) -> None:
        self.feature_channels = 128
        self.num_scales = 2
        self.task = "stereo"
        self.upsample_factor = 4
        self.num_head = 1
        self.ffn_dim_expansion = 4
        self.num_transformer_layers = 6
        self.reg_refine = True

        self.attn_type = "self_swin2d_cross_swin1d"
        self.attn_splits_list = [2,8]
        self.corr_radius_list = [-1,4]
        self.prop_radius_list = [-1, 1]
        self.num_reg_refine = 3

        self.padding_factor = 32
        #self.inference_size = [1024,1536]
        self.inference_size = inference_size

    def get_model_dict(self):
        return {
            "feature_channels": self.feature_channels,
            "num_scales": self.num_scales,
            "task": self.task,
            "upsample_factor": self.upsample_factor,
            "num_head": self.num_head,
            "ffn_dim_expansion": self.ffn_dim_expansion,
            "num_transformer_layers": self.num_transformer_layers,
            "reg_refine": self.reg_refine
        }

    def get_eval_dict(self, left, right):
        return {
            "img0":left,
            "img1":right,
            "attn_type": self.attn_type,
            "attn_splits_list": self.attn_splits_list,
            "corr_radius_list": self.corr_radius_list,
            "prop_radius_list": self.prop_radius_list,
            "num_reg_refine": self.num_reg_refine,
            "task": self.task
        }

class UnimatchBlock:

    def __init__(self, inference_size = [1024, 1536], device = "cpu", verbose=False):
        self.logName = "Unimatch Block"
        self.verbose = verbose
        self.device = device
        self.disposed = False
        self.model_params = UniblockStereoParams(inference_size)
        

    def log(self, x):
        if self.verbose:
            print(f"{self.logName}: {x}")

    def build_model(self):
        if self.disposed:
            self.log("Session disposed!")
            return

        self.log(f"Building Model...")
        #self.model = unimatch.UniMatch(**self.model_params.get_model_dict())
        self.model = unimatch.UniMatch(feature_channels=self.model_params.feature_channels,
                     num_scales=self.model_params.num_scales,
                     upsample_factor=self.model_params.upsample_factor,
                     num_head=self.model_params.num_head,
                     ffn_dim_expansion=self.model_params.ffn_dim_expansion,
                     num_transformer_layers=self.model_params.num_transformer_layers,
                     reg_refine=self.model_params.reg_refine,
                     task=self.model_params.task).to(self.device)
        
        #self.model = torch.nn.DataParallel(self.model).to(self.device)

    def load(self, model_path):
        # load the checkpoint file specified by model_path.loadckpt
        self.log("loading model {}".format(model_path))
        pretrained_dict = torch.load(model_path, map_location=torch.device(self.device))
        self.model.load_state_dict(pretrained_dict['model'],strict=False)

    def dispose(self):
        if not self.disposed:
            del self.model
            gc.collect()
            torch.cuda.empty_cache()
            self.disposed = True

    def _conv_image(self, img):
        if len(img.shape) < 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32)

        #h,w = img.shape[:2]
        #img = self.val_transform(img)

        #ToTensor
        img = np.transpose(img, (2, 0, 1))  # [3, H, W]
        img = torch.from_numpy(img) / 255.

        #Normalize
        for t, m, s in zip(img, IMAGENET_MEAN, IMAGENET_STD):
            t.sub_(m).div_(s)
        
        return img.unsqueeze(0).to(self.device)#[1,3,H,W]

    @torch.no_grad()
    def test(self, left_vpp, right_vpp):
        self.model.eval()
        #Input conversion
        left_vpp = self._conv_image(left_vpp)
        right_vpp = self._conv_image(right_vpp)

        inference_size = self.model_params.inference_size
        padding_factor = self.model_params.padding_factor
            
        #Padding or resize
        if inference_size is None:
            h,w = left_vpp.shape[-2:]
            pad_ht = (((h // padding_factor) + 1) * padding_factor - h) % 32
            pad_wd = (((w // padding_factor) + 1) * padding_factor - w) % 32
            _pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
            left_vpp = F.pad(left_vpp, _pad, mode="replicate")
            right_vpp = F.pad(right_vpp, _pad, mode="replicate")
        else:
            ori_size = left_vpp.shape[-2:]
            if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
                self.log(f"Image resized to {inference_size}")

                left_vpp = F.interpolate(left_vpp, size=inference_size,
                                    mode='bilinear',
                                    align_corners=True)
                right_vpp = F.interpolate(right_vpp, size=inference_size,
                                    mode='bilinear',
                                    align_corners=True)
        
        with torch.no_grad():
            #pred_disp = self.model(**self.model_params.get_eval_dict(left_vpp, right_vpp))['flow_preds'][-1]  # [1, H, W]

            pred_disp = self.model(left_vpp, right_vpp,
                              attn_type=self.model_params.attn_type,
                              attn_splits_list=self.model_params.attn_splits_list,
                              corr_radius_list=self.model_params.corr_radius_list,
                              prop_radius_list=self.model_params.prop_radius_list,
                              num_reg_refine=self.model_params.num_reg_refine,
                              task='stereo',
                              )['flow_preds'][-1]  # [1, H, W]

        if inference_size is None:
            ht, wd = pred_disp.shape[-2:]
            c = [_pad[2], ht-_pad[3], _pad[0], wd-_pad[1]]
            pred_disp = pred_disp[..., c[0]:c[1], c[2]:c[3]]
        else:
            if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
                # resize back
                pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size,
                                        mode='bilinear',
                                        align_corners=True).squeeze(1)  # [1, H, W]
                pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

        return pred_disp[0].cpu().numpy()