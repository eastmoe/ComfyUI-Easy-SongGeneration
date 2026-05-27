import torch
from model_1rvq import PromptCondAudioDiffusion
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
        vae_config="",
        vae_model="",
        layer_num=6, \
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
        self.layer_num = layer_num

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
        # print("Successfully loaded inference scheduler from {}".format(scheduler_name))

    # def sound2sound(self, orig_samples, lyric, st_et, batch_size=1, duration=40.96, steps=200, disable_progress=False,scenario = "start_seg"):
    #     """ Genrate audio without condition. """
    #     with torch.no_grad():
    #         if(orig_samples.shape[-1]<int(duration*48000)+480):
    #             orig_samples =  torch.cat([orig_samples, torch.zeros(orig_samples.shape[0], int(duration*48000+480)-orig_samples.shape[-1], \
    #                 dtype=orig_samples.dtype, device=orig_samples.device)], -1)

    #         orig_samples = orig_samples.to(self.device)
    #         saved_samples = orig_samples[:,0:40*48000].clamp(-1,1)
    #         orig_samples = orig_samples[:,0:40*48000].clamp(-1,1)
    #         max_volume = orig_samples.abs().max(dim=-1)[0]
    #         orig_samples = orig_samples/max_volume.unsqueeze(-1)
    #         print("orig_samples.shape", orig_samples.shape)

    #         latent_length = int((st_et[1] - st_et[0]) * 48000) // 1920 + 1

    #         true_latents = self.vae.encode_audio(orig_samples).permute(0,2,1)

    #         print("true_latents.shape", true_latents.shape)
    #         latents = self.model.inference(orig_samples.repeat(batch_size, 1), [lyric, ]*batch_size, true_latents, latent_length, additional_feats=[], guidance_scale=1.5, num_steps = steps, disable_progress=disable_progress,layer=6, scenario = scenario)
    #         print("latents.shape", latents.shape)
    #         print("latent_length", latent_length)

    #         latents = latents[:,:,:latent_length]
    #         audio = self.vae.decode_audio(latents)
    #         print("audio.shape:",audio.shape)
    #         audio = torch.cat((audio, torch.zeros(audio.shape[0],audio.shape[1], 48000*40 - audio.shape[-1], dtype=audio.dtype, device=audio.device)), dim=-1)
    #         print("audio.shape:",audio.shape)
    #         # audio = audio.reshape(audio.shape[0]//2, 2, -1)
    #         # audio = torch.from_numpy(audio)

    #         if(saved_samples.shape[-1]<audio.shape[-1]):
    #             saved_samples = torch.cat([saved_samples, torch.zeros(saved_samples.shape[0], audio.shape[-1]-saved_samples.shape[-1], dtype=saved_samples.dtype, device=saved_samples.device)],-1)
    #         else:
    #             saved_samples = saved_samples[:,0:audio.shape[-1]]
    #         output = torch.cat([saved_samples.detach().cpu(),audio[0].detach().cpu()],0)
    #     return output

    @torch.no_grad()
    def sound2code(self, orig_samples, batch_size=3):
        if(orig_samples.ndim == 2):
            audios = orig_samples.unsqueeze(0).to(self.device)
        elif(orig_samples.ndim == 3):
            audios = orig_samples.to(self.device)
        else:
            assert orig_samples.ndim in (2,3), orig_samples.shape
        audios = self.preprocess_audio(audios)
        audios = audios.squeeze(0)
        orig_length = audios.shape[-1]
        min_samples = int(40 * self.sample_rate)
        # 40秒对应10个token
        output_len = int(orig_length / float(self.sample_rate) * 25) + 1
        print("output_len: ", output_len)

        while(audios.shape[-1] < min_samples):
            audios = torch.cat([audios, audios], -1)
        int_max_len=audios.shape[-1]//min_samples+1
        audios = torch.cat([audios, audios], -1)
        audios=audios[:,:int(int_max_len*(min_samples))]
        codes_list=[]

        audio_input = audios.reshape(2, -1, min_samples).permute(1, 0, 2).reshape(-1, 2, min_samples)

        for audio_inx in range(0, audio_input.shape[0], batch_size):
            # import pdb; pdb.set_trace()
            codes, _, spk_embeds = self.model.fetch_codes_batch((audio_input[audio_inx:audio_inx+batch_size]), additional_feats=[],layer=self.layer_num)
            codes_list.append(torch.cat(codes, 1))
            # print("codes_list",codes_list[0].shape)

        codes = torch.cat(codes_list, 0).permute(1,0,2).reshape(1, -1)[None] # B 3 T -> 3 B T
        codes=codes[:,:,:output_len]

        return codes

    @torch.no_grad()
    def code2sound(self, codes, prompt=None, duration=40, guidance_scale=1.5, num_steps=20, disable_progress=False):
        codes = codes.to(self.device)

        min_samples = int(duration * 25) # 40ms per frame
        hop_samples = min_samples // 4 * 3
        ovlp_samples = min_samples - hop_samples
        hop_frames = hop_samples
        ovlp_frames = ovlp_samples
        first_latent = torch.randn(codes.shape[0], min_samples, 64).to(self.device)
        first_latent_length = 0
        first_latent_codes_length = 0

        if(isinstance(prompt, torch.Tensor)):
            # prepare prompt
            prompt = prompt.to(self.device)
            if(prompt.ndim == 3):
                assert prompt.shape[0] == 1, prompt.shape
                prompt = prompt[0]
            elif(prompt.ndim == 1):
                prompt = prompt.unsqueeze(0).repeat(2,1)
            elif(prompt.ndim == 2):
                if(prompt.shape[0] == 1):
                    prompt = prompt.repeat(2,1)

            if(prompt.shape[-1] < int(30 * self.sample_rate)):
                # if less than 30s, just choose the first 10s
                prompt = prompt[:,:int(10*self.sample_rate)] # limit max length to 10.24
            else:
                # else choose from 20.48s which might includes verse or chorus
                prompt = prompt[:,int(20*self.sample_rate):int(30*self.sample_rate)] # limit max length to 10.24
            
            vae_device, vae_dtype = _module_device_dtype(
                self.vae,
                self.device,
                getattr(self, "vae_dtype", torch.float32),
            )
            prompt_audio = prompt.to(device=vae_device, dtype=vae_dtype)
            true_latent = self.vae.encode_audio(prompt_audio).permute(0,2,1).to(
                device=first_latent.device,
                dtype=first_latent.dtype,
            )
            # print("true_latent.shape", true_latent.shape)
            # print("first_latent.shape", first_latent.shape)
            #true_latent.shape torch.Size([1, 250, 64])
            # first_latent.shape torch.Size([1, 1000, 64])
            
            first_latent[:,0:true_latent.shape[1],:] = true_latent
            first_latent_length = true_latent.shape[1]
            first_latent_codes = self.sound2code(prompt)
            first_latent_codes_length = first_latent_codes.shape[-1]
            codes = torch.cat([first_latent_codes, codes], -1)




        codes_len= codes.shape[-1]
        target_len = int((codes_len - first_latent_codes_length) / 100 * 4 * self.sample_rate)
        # target_len = int(codes_len / 100 * 4 * self.sample_rate)
        # code repeat
        if(codes_len < min_samples):
            while(codes.shape[-1] < min_samples):
                codes = torch.cat([codes, codes], -1)
            codes = codes[:,:,0:min_samples]
        codes_len = codes.shape[-1]
        if((codes_len - ovlp_samples) % hop_samples > 0):
            len_codes=math.ceil((codes_len - ovlp_samples) / float(hop_samples)) * hop_samples + ovlp_samples
            while(codes.shape[-1] < len_codes):
                codes = torch.cat([codes, codes], -1)
            codes = codes[:,:,0:len_codes]
        latent_length = min_samples
        output = None
        prev_latent_tail = None
        spk_embeds = torch.zeros([1, 32, 1, 32], device=codes.device)
        audio_min_samples = int(min_samples * self.sample_rate // 1000 * 40)
        audio_hop_samples = int(hop_samples * self.sample_rate // 1000 * 40)
        audio_ovlp_samples = audio_min_samples - audio_hop_samples
        ov_win = None
        if audio_ovlp_samples > 0:
            ov_win = torch.from_numpy(np.linspace(0, 1, audio_ovlp_samples)[None, :])
            ov_win = torch.cat([ov_win, 1 - ov_win], -1)
        for sinx in range(0, codes.shape[-1]-hop_samples, hop_samples):
            codes_input=[codes[:,:,sinx:sinx+min_samples]]
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
                latents = self.model.inference_codes(codes_input, spk_embeds, context_latent, latent_length, incontext_length=incontext_length, additional_feats=[], guidance_scale=1.5, num_steps = num_steps, disable_progress=disable_progress, scenario='other_seg')
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
            cur_output = self.vae.decode_audio(decode_latent)[0].detach().cpu()
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
    def preprocess_audio(self, input_audios, threshold=0.8):
        assert len(input_audios.shape) == 3, input_audios.shape
        nchan = input_audios.shape[1]
        input_audios = input_audios.reshape(input_audios.shape[0], -1)
        norm_value = torch.ones_like(input_audios[:,0])
        max_volume = input_audios.abs().max(dim=-1)[0]
        norm_value[max_volume>threshold] = max_volume[max_volume>threshold] / threshold
        return input_audios.reshape(input_audios.shape[0], nchan, -1)/norm_value.unsqueeze(-1).unsqueeze(-1)
    
    @torch.no_grad()
    def sound2sound(self, sound, prompt=None, steps=50, disable_progress=False):
        codes = self.sound2code(sound)
        # print(codes.shape)
        wave = self.code2sound(codes, prompt, guidance_scale=1.5, num_steps=steps, disable_progress=disable_progress)
        # print(fname, wave.shape)
        return wave

    @torch.no_grad()
    def sound2sound_vae(self, sound, prompt=None, steps=50, disable_progress=False):
        min_samples = int(40 * 25) # 40ms per frame
        hop_samples = min_samples // 4 * 3
        ovlp_samples = min_samples - hop_samples
        dur = 20

        latent_list = []
        for i in range(0, sound.shape[-1], dur*48000):
            if(i+dur*2*48000 > sound.shape[-1]):
                latent = tango.vae.encode_audio(sound.cuda()[None,:,i:])
                break
            else:
                latent = tango.vae.encode_audio(sound.cuda()[None,:,i:i+dur*48000])
            latent_list.append(latent)

        output = None
        for i in range(len(latent_list)):
            print(i)
            latent = latent_list[i]
            cur_output = self.vae.decode_audio(latent)[0].detach().cpu()
            if output is None:
                output = cur_output
            else:
                output = torch.cat([output, cur_output], -1)
        return output

    def to(self, device=None, dtype=None, non_blocking=False):
        if device is not None:
            self.device = device
            self.model.device = device
        self.vae = self.vae.to(device, dtype, non_blocking)
        self.model = self.model.to(device, dtype, non_blocking)
        return self
