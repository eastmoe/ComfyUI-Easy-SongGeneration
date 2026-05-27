import torch
from model_septoken import PromptCondAudioDiffusion
import os
import math
import numpy as np
import gc
from tools.get_1dvae_large import _load_weights, get_model
from safetensors.torch import load_file


def _load_state_dict_assign(model, state_dict, *, strict=False):
    try:
        return model.load_state_dict(state_dict, strict=strict, assign=True)
    except TypeError:
        return model.load_state_dict(state_dict, strict=strict)


def _module_device_dtype(module, fallback_device, fallback_dtype=torch.float32):
    try:
        tensor = next(module.parameters())
    except StopIteration:
        try:
            tensor = next(module.buffers())
        except StopIteration:
            return torch.device(fallback_device), fallback_dtype
    return tensor.device, tensor.dtype


class Tango:
    def __init__(self, \
        model_path, \
        vae_config,
        vae_model,
        layer_vocal=7,\
        layer_bgm=3,\
        device="cuda:0"):
        
        self.sample_rate = 48000
        scheduler_name = "configs/scheduler/stable_diffusion_2.1_largenoise_sample.json"
        self.device = device
        self.model_path = model_path
        self.vae_config = vae_config
        self.vae_model = vae_model

        self.vae = get_model(vae_config, vae_model)
        self.vae = self.vae.to(device)
        self.vae=self.vae.eval()
        self.layer_vocal=layer_vocal
        self.layer_bgm=layer_bgm

        self.MAX_DURATION = 360
        main_config = {
            "num_channels":32,
            "unet_model_name":None,
            "unet_model_config_path":"configs/models/transformer2D_wocross_inch112_1x4_multi_large.json",
            "snr_gamma":None,
        }
        self.model = PromptCondAudioDiffusion(**main_config)
        if model_path.endswith(".safetensors"):
            main_weights = load_file(model_path)
        else:
            main_weights = _load_weights(model_path)
        _load_state_dict_assign(self.model, main_weights, strict=False)
        del main_weights
        gc.collect()
        self.model = self.model.to(device)
        print ("Successfully loaded checkpoint from:", model_path)
        
        self.model.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.model.eval()
        self.model.init_device_dtype(torch.device(device), torch.float32)
        
        # self.scheduler = DDIMScheduler.from_pretrained( \
        #     scheduler_name, subfolder="scheduler")
        # self.scheduler = DDPMScheduler.from_pretrained( \
        #     scheduler_name, subfolder="scheduler")
        print("Successfully loaded inference scheduler from {}".format(scheduler_name))


    @torch.no_grad()
    def sound2code(self, orig_vocal, orig_bgm, batch_size=8):
        if(orig_vocal.ndim == 2):
            audios_vocal = orig_vocal.unsqueeze(0).to(self.device)
        elif(orig_vocal.ndim == 3):
            audios_vocal = orig_vocal.to(self.device)
        else:
            assert orig_vocal.ndim in (2,3), orig_vocal.shape
        
        if(orig_bgm.ndim == 2):
            audios_bgm = orig_bgm.unsqueeze(0).to(self.device)
        elif(orig_bgm.ndim == 3):
            audios_bgm = orig_bgm.to(self.device)
        else:
            assert orig_bgm.ndim in (2,3), orig_bgm.shape

        
        audios_vocal = self.preprocess_audio(audios_vocal)
        audios_vocal = audios_vocal.squeeze(0)
        audios_bgm = self.preprocess_audio(audios_bgm)
        audios_bgm = audios_bgm.squeeze(0)
        if audios_vocal.shape[-1] > audios_bgm.shape[-1]:
            audios_vocal = audios_vocal[:,:audios_bgm.shape[-1]]
        else:
            audios_bgm = audios_bgm[:,:audios_vocal.shape[-1]]


        orig_length = audios_vocal.shape[-1]
        min_samples = int(40 * self.sample_rate)
        # 40秒对应10个token
        output_len = int(orig_length / float(self.sample_rate) * 25) + 1

        while(audios_vocal.shape[-1] < min_samples):
            audios_vocal = torch.cat([audios_vocal, audios_vocal], -1)
            audios_bgm = torch.cat([audios_bgm, audios_bgm], -1)
        int_max_len=audios_vocal.shape[-1]//min_samples+1
        audios_vocal = torch.cat([audios_vocal, audios_vocal], -1)
        audios_bgm = torch.cat([audios_bgm, audios_bgm], -1)
        audios_vocal=audios_vocal[:,:int(int_max_len*(min_samples))]
        audios_bgm=audios_bgm[:,:int(int_max_len*(min_samples))]
        codes_vocal_list=[]
        codes_bgm_list=[]

    

        audio_vocal_input = audios_vocal.reshape(2, -1, min_samples).permute(1, 0, 2).reshape(-1, 2, min_samples)
        audio_bgm_input = audios_bgm.reshape(2, -1, min_samples).permute(1, 0, 2).reshape(-1, 2, min_samples)

        for audio_inx in range(0, audio_vocal_input.shape[0], batch_size):
            [codes_vocal,codes_bgm], _, spk_embeds = self.model.fetch_codes_batch((audio_vocal_input[audio_inx:audio_inx+batch_size]), (audio_bgm_input[audio_inx:audio_inx+batch_size]), additional_feats=[],layer_vocal=self.layer_vocal,layer_bgm=self.layer_bgm)
            codes_vocal_list.append(codes_vocal)
            codes_bgm_list.append(codes_bgm)

        codes_vocal = torch.cat(codes_vocal_list, 0).permute(1,0,2).reshape(1, -1)[None]
        codes_bgm = torch.cat(codes_bgm_list, 0).permute(1,0,2).reshape(1, -1)[None]
        codes_vocal=codes_vocal[:,:,:output_len]
        codes_bgm=codes_bgm[:,:,:output_len]

        return codes_vocal, codes_bgm

    @torch.no_grad()
    def code2sound(self, codes, prompt_vocal=None, prompt_bgm=None, duration=40, guidance_scale=1.5, num_steps=20, disable_progress=False, chunked=False, chunk_size=128):
        codes_vocal,codes_bgm = codes
        codes_vocal = codes_vocal.to(self.device)
        codes_bgm = codes_bgm.to(self.device)

        min_samples = duration * 25 # 40ms per frame
        hop_samples = min_samples // 4 * 3
        ovlp_samples = min_samples - hop_samples
        hop_frames = hop_samples
        ovlp_frames = ovlp_samples
        first_latent = torch.randn(codes_vocal.shape[0], min_samples, 64).to(self.device)
        first_latent_length = 0
        first_latent_codes_length = 0


        if(isinstance(prompt_vocal, torch.Tensor) and isinstance(prompt_bgm, torch.Tensor)):
            # prepare prompt
            prompt_vocal = prompt_vocal.to(self.device)
            prompt_bgm = prompt_bgm.to(self.device)
            if(prompt_vocal.ndim == 3):
                assert prompt_vocal.shape[0] == 1, prompt_vocal.shape
                prompt_vocal = prompt_vocal[0]
                prompt_bgm = prompt_bgm[0]
            elif(prompt_vocal.ndim == 1):
                prompt_vocal = prompt_vocal.unsqueeze(0).repeat(2,1)
                prompt_bgm = prompt_bgm.unsqueeze(0).repeat(2,1)
            elif(prompt_vocal.ndim == 2):
                if(prompt_vocal.shape[0] == 1):
                    prompt_vocal = prompt_vocal.repeat(2,1)
                    prompt_bgm = prompt_bgm.repeat(2,1)

            if(prompt_vocal.shape[-1] < int(30 * self.sample_rate)):
                # if less than 30s, just choose the first 10s
                prompt_vocal = prompt_vocal[:,:int(10*self.sample_rate)] # limit max length to 10.24
                prompt_bgm = prompt_bgm[:,:int(10*self.sample_rate)] # limit max length to 10.24
            else:
                # else choose from 20.48s which might includes verse or chorus
                prompt_vocal = prompt_vocal[:,int(20*self.sample_rate):int(30*self.sample_rate)] # limit max length to 10.24
                prompt_bgm = prompt_bgm[:,int(20*self.sample_rate):int(30*self.sample_rate)] # limit max length to 10.24
            
            vae_device, vae_dtype = _module_device_dtype(
                self.vae,
                self.device,
                getattr(self, "vae_dtype", torch.float32),
            )
            prompt_audio = (prompt_vocal + prompt_bgm).to(device=vae_device, dtype=vae_dtype)
            true_latent = self.vae.encode_audio(prompt_audio).permute(0,2,1).to(
                device=first_latent.device,
                dtype=first_latent.dtype,
            )
            
            first_latent[:,0:true_latent.shape[1],:] = true_latent
            first_latent_length = true_latent.shape[1]
            first_latent_codes = self.sound2code(prompt_vocal, prompt_bgm)
            first_latent_codes_vocal = first_latent_codes[0]
            first_latent_codes_bgm = first_latent_codes[1]
            first_latent_codes_length = first_latent_codes_vocal.shape[-1]
            codes_vocal = torch.cat([first_latent_codes_vocal, codes_vocal], -1)
            codes_bgm = torch.cat([first_latent_codes_bgm, codes_bgm], -1)
            

        codes_len= codes_vocal.shape[-1]
        target_len = int((codes_len - first_latent_codes_length) / 100 * 4 * self.sample_rate)
        # target_len = int(codes_len / 100 * 4 * self.sample_rate)
        # code repeat
        if(codes_len < min_samples):
            while(codes_vocal.shape[-1] < min_samples):
                codes_vocal = torch.cat([codes_vocal, codes_vocal], -1)
                codes_bgm = torch.cat([codes_bgm, codes_bgm], -1)

            codes_vocal = codes_vocal[:,:,0:min_samples]
            codes_bgm = codes_bgm[:,:,0:min_samples]
        codes_len = codes_vocal.shape[-1]
        if((codes_len - ovlp_samples) % hop_samples > 0):
            len_codes=math.ceil((codes_len - ovlp_samples) / float(hop_samples)) * hop_samples + ovlp_samples
            while(codes_vocal.shape[-1] < len_codes):
                codes_vocal = torch.cat([codes_vocal, codes_vocal], -1)
                codes_bgm = torch.cat([codes_bgm, codes_bgm], -1)
            codes_vocal = codes_vocal[:,:,0:len_codes]
            codes_bgm = codes_bgm[:,:,0:len_codes]
        latent_length = min_samples
        output = None
        prev_latent_tail = None
        spk_embeds = torch.zeros([1, 32, 1, 32], device=codes_vocal.device)
        audio_min_samples = int(min_samples * self.sample_rate // 1000 * 40)
        audio_hop_samples = int(hop_samples * self.sample_rate // 1000 * 40)
        audio_ovlp_samples = audio_min_samples - audio_hop_samples
        ov_win = None
        if audio_ovlp_samples > 0:
            ov_win = torch.from_numpy(np.linspace(0, 1, audio_ovlp_samples)[None, :])
            ov_win = torch.cat([ov_win, 1 - ov_win], -1)
        for sinx in range(0, codes_vocal.shape[-1]-hop_samples, hop_samples):
            codes_vocal_input=codes_vocal[:,:,sinx:sinx+min_samples]
            codes_bgm_input=codes_bgm[:,:,sinx:sinx+min_samples]
            if(sinx == 0):
                incontext_length = first_latent_length
                context_latent = first_latent
            else:
                true_latent = prev_latent_tail.permute(0,2,1)
                len_add_to_1000 = min_samples - true_latent.shape[-2]
                incontext_length = true_latent.shape[-2]
                context_latent = torch.cat([true_latent, torch.randn(true_latent.shape[0],  len_add_to_1000, true_latent.shape[-1]).to(self.device)], -2)
            with torch.autocast(
                device_type="cuda",
                dtype=getattr(self, "diffusion_dtype", torch.float16),
                enabled=getattr(self, "diffusion_dtype", torch.float16) in (torch.float16, torch.bfloat16),
            ):
                latents = self.model.inference_codes([codes_vocal_input,codes_bgm_input], spk_embeds, context_latent, latent_length, incontext_length=incontext_length, additional_feats=[], guidance_scale=1.5, num_steps = num_steps, disable_progress=disable_progress, scenario='other_seg')
            prev_latent_tail = latents[:, :, -ovlp_frames:].detach()
            vae_device, vae_dtype = _module_device_dtype(
                self.vae,
                self.device,
                getattr(self, "vae_dtype", torch.float32),
            )
            decode_latent = latents.to(device=vae_device, dtype=vae_dtype)
            if sinx == 0:
                decode_latent = decode_latent[:,:,first_latent_length:]
            torch.cuda.empty_cache()
            cur_output = self.vae.decode_audio(decode_latent, chunked=chunked, chunk_size=chunk_size)[0].detach().cpu()
            if output is None:
                output = cur_output
            else:
                output[:, -audio_ovlp_samples:] = output[:, -audio_ovlp_samples:] * ov_win[:, -audio_ovlp_samples:] + cur_output[:, 0:audio_ovlp_samples] * ov_win[:, 0:audio_ovlp_samples]
                output = torch.cat([output, cur_output[:, audio_ovlp_samples:]], -1)
            del latents, decode_latent, cur_output
            torch.cuda.empty_cache()
        output = output[:, 0:target_len]
        return output

    @torch.no_grad()
    def preprocess_audio(self, input_audios_vocal, threshold=0.8):
        assert len(input_audios_vocal.shape) == 3, input_audios_vocal.shape
        nchan = input_audios_vocal.shape[1]
        input_audios_vocal = input_audios_vocal.reshape(input_audios_vocal.shape[0], -1)
        norm_value = torch.ones_like(input_audios_vocal[:,0])
        max_volume = input_audios_vocal.abs().max(dim=-1)[0]
        norm_value[max_volume>threshold] = max_volume[max_volume>threshold] / threshold
        return input_audios_vocal.reshape(input_audios_vocal.shape[0], nchan, -1)/norm_value.unsqueeze(-1).unsqueeze(-1)
    
    @torch.no_grad()
    def sound2sound(self, orig_vocal,orig_bgm, prompt_vocal=None,prompt_bgm=None, steps=50, disable_progress=False):
        codes_vocal, codes_bgm = self.sound2code(orig_vocal,orig_bgm)
        codes=[codes_vocal, codes_bgm]
        wave = self.code2sound(codes, prompt_vocal,prompt_bgm, guidance_scale=1.5, num_steps=steps, disable_progress=disable_progress)
        return wave
    
    def to(self, device=None, dtype=None, non_blocking=False):
        if device is not None:
            self.device = device
            self.model.device = device
        self.vae = self.vae.to(device, dtype, non_blocking)
        self.model = self.model.to(device, dtype, non_blocking)
        return self
