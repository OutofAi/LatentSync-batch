# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py

import inspect
import os
import shutil
from typing import Callable, List, Optional, Union
import subprocess

import numpy as np
import torch
import torchvision

from diffusers.utils import is_accelerate_available
from packaging import version

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging

from einops import rearrange

from ..models.unet import UNet3DConditionModel
from ..utils.image_processor import ImageProcessor
from ..utils.util import read_video, read_audio, write_video, timer
from ..whisper.audio2feature import Audio2Feature
import tqdm
import soundfile as sf
from pathlib import Path
import torchvision.transforms.functional as TF
from einops import rearrange
from torchvision import transforms
import cv2
import math

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class LipsyncPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        audio_encoder: Audio2Feature,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()
            
        if scheduler == None:
            BASE_DIR = Path(__file__).resolve().parent
            config_dir = (BASE_DIR / ".." / "configs").resolve()
            scheduler = DDIMScheduler.from_pretrained(config_dir)

        if unet == None:
            
            unet, _ = UNet3DConditionModel.from_pretrained(device="cpu",)
            
            unet = unet.to(dtype=torch.float16)

        if audio_encoder == None:
            audio_encoder = Audio2Feature(device="cuda", num_frames=16)

        if vae == None:
            vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
            vae.config.scaling_factor = 0.18215
            vae.config.shift_factor = 0

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.set_progress_bar_config(desc="Steps")

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        decoded_latents = self.vae.decode(latents).sample
        return decoded_latents

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, height, width, callback_steps):
        assert height == width, "Height and width must be equal"

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_frames, num_channels_latents, height, width, dtype, device, generator):
        shape = (
            batch_size,
            num_channels_latents,
            1,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        rand_device = "cpu" if device.type == "mps" else device
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        latents = latents.repeat(1, 1, num_frames, 1, 1)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_mask_latents(
        self, mask, masked_image, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        masked_image = masked_image.to(device=device, dtype=dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
        masked_image_latents = (masked_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        mask = mask.to(device=device, dtype=dtype)

        # assume batch size = 1
        mask = rearrange(mask, "f c h w -> 1 c f h w")
        masked_image_latents = rearrange(masked_image_latents, "f c h w -> 1 c f h w")

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )
        return mask, masked_image_latents

    def prepare_image_latents(self, images, device, dtype, generator, do_classifier_free_guidance):
        images = images.to(device=device, dtype=dtype)
        image_latents = self.vae.encode(images).latent_dist.sample(generator=generator)
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        image_latents = rearrange(image_latents, "f c h w -> 1 c f h w")
        image_latents = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents

        return image_latents

    def set_progress_bar_config(self, **kwargs):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(kwargs)

    @staticmethod
    def paste_surrounding_pixels_back(decoded_latents, pixel_values, masks, device, weight_dtype):
        # Paste the surrounding pixels back, because we only want to change the mouth region
        pixel_values = pixel_values.to(device=device, dtype=weight_dtype)
        masks = masks.to(device=device, dtype=weight_dtype)
        combined_pixel_values = decoded_latents * masks + pixel_values * (1 - masks)
        return combined_pixel_values

    @staticmethod
    def pixel_values_to_images(pixel_values: torch.Tensor):
        pixel_values = rearrange(pixel_values, "f c h w -> f h w c")
        pixel_values = (pixel_values / 2 + 0.5).clamp(0, 1)
        images = (pixel_values * 255).to(torch.uint8)
        images = images.cpu().numpy()
        return images
    
    def affine_transform_video_batch(self, video_path, batch_size=512):
        video_frames = read_video(video_path, use_decord=True)

        faces = []
        boxes = []
        affine_matrices = []
        frames = []
        valid_frame_indices = []  

        for i in tqdm.tqdm(range(0, len(video_frames), batch_size)):
            batch = video_frames[i : i + batch_size]

            face_batch, box_batch, affine_batch, valid_batch, valid_local_indices = self.image_processor.affine_transform_batch(batch)

            for f, b, a, original_frame, valid_idx in zip(face_batch, box_batch, affine_batch, valid_batch, valid_local_indices):
                faces.append(f)
                boxes.append(b)
                affine_matrices.append(a)
                frames.append(original_frame)
                valid_frame_indices.append(i + valid_idx)

        faces = torch.stack(faces)
        # return:
        #   - faces/boxes/affine_matrices: only valid frames
        #   - video_frames: ALL original frames (for reinsertion later)
        #   - valid_frame_indices: mapping into video_frames
        return faces, frames, boxes, affine_matrices, video_frames, valid_frame_indices

    def affine_transform_video(self, video_path):
        video_frames = read_video(video_path, use_decord=True)
        faces = []
        boxes = []
        affine_matrices = []
        valid_frame_indices = []  

        print(f"Affine transforming {len(video_frames)} faces...")
        for idx, frame in enumerate(tqdm.tqdm(video_frames)):
            face, box, affine_matrix = self.image_processor.affine_transform(frame) 
            faces.append(face)
            boxes.append(box)
            affine_matrices.append(affine_matrix)
            valid_frame_indices.append(idx)

        faces = torch.stack(faces)
        return faces, video_frames, boxes, affine_matrices, video_frames, valid_frame_indices

    def restore_video(self, faces, video_frames, boxes, affine_matrices):
        video_frames = video_frames[: faces.shape[0]]
        out_frames = []
        for index, face in enumerate(faces):
            x1, y1, x2, y2 = boxes[index]
            height = int(y2 - y1)
            width = int(x2 - x1)
            face = torchvision.transforms.functional.resize(face, size=(height, width), antialias=True)
            face = rearrange(face, "c h w -> h w c")
            face = (face / 2 + 0.5).clamp(0, 1)
            face = (face * 255).to(torch.uint8).cpu().numpy()
            out_frame = self.image_processor.restorer.restore_img(video_frames[index], face, affine_matrices[index])
            out_frames.append(out_frame)
        return np.stack(out_frames, axis=0)
    
    def restore_video_batch(self, faces: torch.Tensor, video_frames: np.ndarray,
                    boxes: list, affine_matrices: list, chunk_size: int = 256):
        # Assume faces and video_frames correspond 1:1
        video_frames = video_frames[: len(faces)]
        out_frames_t = []

        # print(f"Restoring {len(faces)} faces...")

        # --- compute common height/width once ---
        x1, y1, x2, y2 = boxes[0]
        target_h = int(math.ceil(y2 - y1))
        target_w = int(math.ceil(x2 - x1))

        for start in range(0, len(faces), chunk_size):
            end = min(start + chunk_size, len(faces))

            faces_chunk = faces[start:end]         # (b, C, Hf, Wf)
            frames_chunk = video_frames[start:end] # (b, H, W, C)
            aff_chunk = affine_matrices[start:end] # len b

            # single batched resize on GPU/CPU (wherever faces_chunk is)
            faces_batch = TF.resize(
                faces_chunk,
                size=(target_h, target_w),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=False,
            )  # (b, C, target_h, target_w)

            with torch.no_grad():
                out_chunk = self.image_processor.restorer.restore_imgs(
                    input_imgs=frames_chunk,
                    faces=faces_batch,
                    affine_matrices=np.stack(aff_chunk, axis=0),
                )  # (b, H, W, C)

            out_frames_t.append(out_chunk)

        out_all_t = torch.cat(out_frames_t, dim=0)    # (N, H, W, C)
        # out_all_np = out_all_t.cpu().numpy()
        return out_all_t

    @torch.no_grad()
    def __call__(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        video_mask_path: str = None,
        num_frames: int = 16,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.5,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        mask: str = "fix_mask",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        is_train = self.unet.training
        self.unet.eval()

        # 0. Define call parameters
        batch_size = 1
        device = self._execution_device
        self.image_processor = ImageProcessor(height, mask=mask, device="cuda")
        # self.set_progress_bar_config(desc=f"Sample frames: {num_frames}")
        self.set_progress_bar_config(disable=True)
        with timer("doing it all"):

            with timer("affine_transform_video_batch"):
                cropped_face_frames, valid_video_frames, boxes, affine_matrices, all_video_frames, valid_frame_indices = self.affine_transform_video_batch(video_path)
            audio_samples = read_audio(audio_path)

            # 1. Default height and width to unet
            height = height or self.unet.config.sample_size * self.vae_scale_factor
            width = width or self.unet.config.sample_size * self.vae_scale_factor

            # 2. Check inputs
            self.check_inputs(height, width, callback_steps)

            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = guidance_scale > 1.0

            # 3. set timesteps
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

            # 4. Prepare extra step kwargs.
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            self.video_fps = video_fps

            if self.unet.add_audio_layer:
                whisper_feature = self.audio_encoder.audio2feat(audio_path)
                whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)

                whisper_chunks = [chunk for chunk, is_valid in zip(whisper_chunks, valid_frame_indices) if is_valid]
                whisper_chunks = whisper_chunks[: len(cropped_face_frames)]

                total_frames = min(len(cropped_face_frames), len(whisper_chunks))
            else:
                whisper_chunks = None
                total_frames = len(cropped_face_frames)

            # If we have no frames to process, fall back or raise a clear error
            if total_frames == 0:
                raise ValueError("No valid frames/audio chunks to process. "
                                "Check face detection / valid_indices / audio features.")

            # Make sure num_frames never exceeds what we have
            if num_frames > total_frames:
                num_frames = total_frames

            # How many chunks of size num_frames we’ll run
            num_inferences = max(1, total_frames // num_frames)
            
            synced_video_frames = []
            masked_video_frames = []

            num_channels_latents = self.vae.config.latent_channels

            # Prepare latent variables
            all_latents = self.prepare_latents(
                batch_size,
                num_frames * num_inferences,
                num_channels_latents,
                height,
                width,
                weight_dtype,
                device,
                generator,
            )

            for i in tqdm.tqdm(range(num_inferences), desc="Doing inference..."):
                if self.unet.add_audio_layer:
                    audio_embeds = torch.stack(whisper_chunks[i * num_frames : (i + 1) * num_frames])
                    audio_embeds = audio_embeds.to(device, dtype=weight_dtype)
                    if do_classifier_free_guidance:
                        empty_audio_embeds = torch.zeros_like(audio_embeds)
                        audio_embeds = torch.cat([empty_audio_embeds, audio_embeds])
                else:
                    audio_embeds = None
                inference_video_frames = cropped_face_frames[i * num_frames : (i + 1) * num_frames]
                latents = all_latents[:, :, i * num_frames : (i + 1) * num_frames]
                pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
                    inference_video_frames, affine_transform=False
                )

                # 7. Prepare mask latent variables
                mask_latents, masked_image_latents = self.prepare_mask_latents(
                    masks,
                    masked_pixel_values,
                    height,
                    width,
                    weight_dtype,
                    device,
                    generator,
                    do_classifier_free_guidance,
                )

                # 8. Prepare image latents
                image_latents = self.prepare_image_latents(
                    pixel_values,
                    device,
                    weight_dtype,
                    generator,
                    do_classifier_free_guidance,
                )

                # 9. Denoising loop
                num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
                with self.progress_bar(total=num_inference_steps) as progress_bar:
                    for j, t in enumerate(timesteps):
                        # expand the latents if we are doing classifier free guidance
                        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                        # concat latents, mask, masked_image_latents in the channel dimension
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                        latent_model_input = torch.cat(
                            [latent_model_input, mask_latents, masked_image_latents, image_latents], dim=1
                        )

                        # predict the noise residual
                        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=audio_embeds).sample

                        # perform guidance
                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)

                        # compute the previous noisy sample x_t -> x_t-1
                        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                        # call the callback, if provided
                        if j == len(timesteps) - 1 or ((j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0):
                            progress_bar.update()
                            if callback is not None and j % callback_steps == 0:
                                callback(j, t, latents)

                # Recover the pixel values
                decoded_latents = self.decode_latents(latents)
                decoded_latents = self.paste_surrounding_pixels_back(
                    decoded_latents, pixel_values, 1 - masks, device, weight_dtype
                )
                synced_video_frames.append(decoded_latents)


            faces_tensor = torch.cat(synced_video_frames, dim=0)  # (N, C, H, W)

            face_video_frames_np = np.stack(valid_video_frames, axis=0)  # (N, H, W, C)

            synced_video_frames_t = self.restore_video_batch(
                faces_tensor,
                face_video_frames_np,
                boxes,
                affine_matrices,
            )

            synced_video_frames = synced_video_frames_t.detach().cpu().numpy()

            # Start from original frames
            final_frames = list(all_video_frames)  # length N_total

            # Overwrite only the valid positions with your restored frames
            for restored_frame, idx in zip(synced_video_frames, valid_frame_indices):
                final_frames[idx] = restored_frame

            final_frames_np = np.stack(final_frames, axis=0)  # (N_total, H, W, C)

            audio_samples_remain_length = int(final_frames_np.shape[0] / video_fps * audio_sample_rate)
            audio_samples = audio_samples[:audio_samples_remain_length].cpu().numpy()

        if is_train:
            self.unet.train()

        temp_dir = "temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        write_video(os.path.join(temp_dir, "video.mp4"), final_frames_np, fps=25)
        # write_video(video_mask_path, masked_video_frames, fps=25)

        sf.write(os.path.join(temp_dir, "audio.wav"), audio_samples, audio_sample_rate)

        command = f"ffmpeg -y -loglevel error -nostdin -i {os.path.join(temp_dir, 'video.mp4')} -i {os.path.join(temp_dir, 'audio.wav')} -c:v libx264 -c:a aac -q:v 0 -q:a 0 {video_out_path}"
        subprocess.run(command, shell=True)