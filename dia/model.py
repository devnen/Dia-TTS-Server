import time
from enum import Enum

import dac
import numpy as np
import torch
import torchaudio

from .audio import (
    apply_audio_delay,
    build_delay_indices,
    build_revert_indices,
    decode,
    revert_audio_delay,
)
from .config import DiaConfig
from .layers import DiaModel
from .state import DecoderInferenceState, DecoderOutput, EncoderInferenceState


DEFAULT_SAMPLE_RATE = 44100


def _get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sample_next_token(
    logits_BCxV: torch.Tensor,
    temperature: float,
    top_p: float,
    cfg_filter_top_k: int | None = None,
    generator: torch.Generator = None,  # Added generator parameter
) -> torch.Tensor:
    if temperature == 0.0:
        return torch.argmax(logits_BCxV, dim=-1)

    logits_BCxV = logits_BCxV / temperature
    if cfg_filter_top_k is not None:
        _, top_k_indices_BCxV = torch.topk(logits_BCxV, k=cfg_filter_top_k, dim=-1)
        mask = torch.ones_like(logits_BCxV, dtype=torch.bool)
        mask.scatter_(dim=-1, index=top_k_indices_BCxV, value=False)
        logits_BCxV = logits_BCxV.masked_fill(mask, -torch.inf)

    if top_p < 1.0:
        probs_BCxV = torch.softmax(logits_BCxV, dim=-1)
        sorted_probs_BCxV, sorted_indices_BCxV = torch.sort(
            probs_BCxV, dim=-1, descending=True
        )
        cumulative_probs_BCxV = torch.cumsum(sorted_probs_BCxV, dim=-1)

        sorted_indices_to_remove_BCxV = cumulative_probs_BCxV > top_p
        sorted_indices_to_remove_BCxV[..., 1:] = sorted_indices_to_remove_BCxV[
            ..., :-1
        ].clone()
        sorted_indices_to_remove_BCxV[..., 0] = 0

        indices_to_remove_BCxV = torch.zeros_like(sorted_indices_to_remove_BCxV)
        indices_to_remove_BCxV.scatter_(
            dim=-1, index=sorted_indices_BCxV, src=sorted_indices_to_remove_BCxV
        )
        logits_BCxV = logits_BCxV.masked_fill(indices_to_remove_BCxV, -torch.inf)

    final_probs_BCxV = torch.softmax(logits_BCxV, dim=-1)

    sampled_indices_BC = torch.multinomial(
        final_probs_BCxV,
        num_samples=1,
        generator=generator,  # Pass generator to multinomial
    )
    sampled_indices_C = sampled_indices_BC.squeeze(-1)
    return sampled_indices_C


class ComputeDtype(str, Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"

    def to_dtype(self) -> torch.dtype:
        if self == ComputeDtype.FLOAT32:
            return torch.float32
        elif self == ComputeDtype.FLOAT16:
            return torch.float16
        elif self == ComputeDtype.BFLOAT16:
            return torch.bfloat16
        else:
            raise ValueError(f"Unsupported compute dtype: {self}")


class Dia:
    def __init__(
        self,
        config: DiaConfig,
        compute_dtype: str | ComputeDtype = ComputeDtype.FLOAT32,
        device: torch.device | None = None,
    ):
        """Initializes the Dia model.

        Args:
            config: The configuration object for the model.
            device: The device to load the model onto. If None, will automatically select the best available device.

        Raises:
            RuntimeError: If there is an error loading the DAC model.
        """
        super().__init__()
        self.config = config
        self.device = device if device is not None else _get_default_device()
        if isinstance(compute_dtype, str):
            compute_dtype = ComputeDtype(compute_dtype)
        self.compute_dtype = compute_dtype.to_dtype()
        self.model = DiaModel(config, self.compute_dtype)
        self.dac_model = None

    @classmethod
    def from_local(
        cls,
        config_path: str,
        checkpoint_path: str,
        compute_dtype: str | ComputeDtype = ComputeDtype.FLOAT32,
        device: torch.device | None = None,
    ) -> "Dia":
        """Loads the Dia model from local configuration and checkpoint files.

        Args:
            config_path: Path to the configuration JSON file.
            checkpoint_path: Path to the model checkpoint (.pth) file.
            device: The device to load the model onto. If None, will automatically select the best available device.

        Returns:
            An instance of the Dia model loaded with weights and set to eval mode.

        Raises:
            FileNotFoundError: If the config or checkpoint file is not found.
            RuntimeError: If there is an error loading the checkpoint.
        """
        config = DiaConfig.load(config_path)
        if config is None:
            raise FileNotFoundError(f"Config file not found at {config_path}")

        dia = cls(config, compute_dtype, device)

        try:
            state_dict = torch.load(checkpoint_path, map_location=dia.device)
            dia.model.load_state_dict(state_dict)
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(
                f"Error loading checkpoint from {checkpoint_path}"
            ) from e

        dia.model.to(dia.device)
        dia.model.eval()
        dia._load_dac_model()
        return dia

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "nari-labs/Dia-1.6B",
        compute_dtype: str | ComputeDtype = ComputeDtype.FLOAT32,
        device: torch.device | None = None,
    ) -> "Dia":
        """Loads the Dia model from a Hugging Face Hub repository.

        Downloads the configuration and checkpoint files from the specified
        repository ID and then loads the model.

        Args:
            model_name: The Hugging Face Hub repository ID (e.g., "nari-labs/Dia-1.6B").
            compute_dtype: The computation dtype to use.
            device: The device to load the model onto. If None, will automatically select the best available device.

        Returns:
            An instance of the Dia model loaded with weights and set to eval mode.

        Raises:
            FileNotFoundError: If config or checkpoint download/loading fails.
            RuntimeError: If there is an error loading the checkpoint.
        """
        if isinstance(compute_dtype, str):
            compute_dtype = ComputeDtype(compute_dtype)
        loaded_model = DiaModel.from_pretrained(
            model_name, compute_dtype=compute_dtype.to_dtype()
        )
        config = loaded_model.config
        dia = cls(config, compute_dtype, device)

        dia.model = loaded_model
        dia.model.to(dia.device)
        dia.model.eval()
        dia._load_dac_model()
        return dia

    def _load_dac_model(self):
        try:
            dac_model_path = dac.utils.download()
            dac_model = dac.DAC.load(dac_model_path).to(self.device)
        except Exception as e:
            raise RuntimeError("Failed to load DAC model") from e
        self.dac_model = dac_model

    def _prepare_text_input(self, text: str) -> torch.Tensor:
        """Encodes text prompt, pads, and creates attention mask and positions."""
        text_pad_value = self.config.data.text_pad_value
        max_len = self.config.data.text_length

        byte_text = text.encode("utf-8")
        replaced_bytes = byte_text.replace(b"[S1]", b"\x01").replace(b"[S2]", b"\x02")
        text_tokens = list(replaced_bytes)

        current_len = len(text_tokens)
        padding_needed = max_len - current_len
        if padding_needed <= 0:
            text_tokens = text_tokens[:max_len]
            padded_text_np = np.array(text_tokens, dtype=np.uint8)
        else:
            padded_text_np = np.pad(
                text_tokens,
                (0, padding_needed),
                mode="constant",
                constant_values=text_pad_value,
            ).astype(np.uint8)

        src_tokens = (
            torch.from_numpy(padded_text_np).to(torch.long).to(self.device).unsqueeze(0)
        )  # [1, S]
        return src_tokens

    def _prepare_audio_prompt(
        self, audio_prompt: torch.Tensor | None
    ) -> tuple[torch.Tensor, int]:
        num_channels = self.config.data.channels
        audio_bos_value = self.config.data.audio_bos_value
        audio_pad_value = self.config.data.audio_pad_value
        delay_pattern = self.config.data.delay_pattern
        max_delay_pattern = max(delay_pattern)

        prefill = torch.full(
            (1, num_channels),
            fill_value=audio_bos_value,
            dtype=torch.int,
            device=self.device,
        )

        prefill_step = 1

        if audio_prompt is not None:
            prefill_step += audio_prompt.shape[0]
            prefill = torch.cat([prefill, audio_prompt], dim=0)

        delay_pad_tensor = torch.full(
            (max_delay_pattern, num_channels),
            fill_value=-1,
            dtype=torch.int,
            device=self.device,
        )
        prefill = torch.cat([prefill, delay_pad_tensor], dim=0)

        delay_precomp = build_delay_indices(
            B=1,
            T=prefill.shape[0],
            C=num_channels,
            delay_pattern=delay_pattern,
        )

        prefill = apply_audio_delay(
            audio_BxTxC=prefill.unsqueeze(0),
            pad_value=audio_pad_value,
            bos_value=audio_bos_value,
            precomp=delay_precomp,
        ).squeeze(0)

        return prefill, prefill_step

    def _prepare_generation(
        self, text: str, audio_prompt: str | torch.Tensor | None, verbose: bool
    ):
        enc_input_cond = self._prepare_text_input(text)
        enc_input_uncond = torch.zeros_like(enc_input_cond)
        enc_input = torch.cat([enc_input_uncond, enc_input_cond], dim=0)

        if isinstance(audio_prompt, str):
            audio_prompt = self.load_audio(audio_prompt)
        prefill, prefill_step = self._prepare_audio_prompt(audio_prompt)

        if verbose:
            print("generate: data loaded")

        enc_state = EncoderInferenceState.new(self.config, enc_input_cond)
        encoder_out = self.model.encoder(enc_input, enc_state)

        # Clean up inputs after encoding
        del enc_input_uncond, enc_input

        dec_cross_attn_cache = self.model.decoder.precompute_cross_attn_cache(
            encoder_out, enc_state.positions
        )
        dec_state = DecoderInferenceState.new(
            self.config,
            enc_state,
            encoder_out,
            dec_cross_attn_cache,
            self.compute_dtype,
        )

        # Can delete encoder output after it's used by decoder state
        del dec_cross_attn_cache

        dec_output = DecoderOutput.new(self.config, self.device)
        dec_output.prefill(prefill, prefill_step)

        dec_step = prefill_step - 1
        if dec_step > 0:
            dec_state.prepare_step(0, dec_step)
            tokens_BxTxC = (
                dec_output.get_tokens_at(0, dec_step).unsqueeze(0).expand(2, -1, -1)
            )
            self.model.decoder.forward(tokens_BxTxC, dec_state)

        return dec_state, dec_output

    def _decoder_step(
        self,
        tokens_Bx1xC: torch.Tensor,
        dec_state: DecoderInferenceState,
        cfg_scale: float,
        temperature: float,
        top_p: float,
        cfg_filter_top_k: int,
        generator: torch.Generator = None,  # Added generator parameter
    ) -> torch.Tensor:
        audio_eos_value = self.config.data.audio_eos_value
        logits_Bx1xCxV = self.model.decoder.decode_step(tokens_Bx1xC, dec_state)

        logits_last_BxCxV = logits_Bx1xCxV[:, -1, :, :]
        # ADD: Remove the full logits tensor
        del logits_Bx1xCxV

        uncond_logits_CxV = logits_last_BxCxV[0, :, :]
        cond_logits_CxV = logits_last_BxCxV[1, :, :]
        # ADD: Remove the combined logits
        del logits_last_BxCxV

        logits_CxV = cond_logits_CxV + cfg_scale * (cond_logits_CxV - uncond_logits_CxV)
        # ADD: Remove component logits
        del uncond_logits_CxV, cond_logits_CxV

        logits_CxV[:, audio_eos_value + 1 :] = -torch.inf
        logits_CxV[1:, audio_eos_value:] = -torch.inf

        pred_C = _sample_next_token(
            logits_CxV.float(),
            temperature=temperature,
            top_p=top_p,
            cfg_filter_top_k=cfg_filter_top_k,
            generator=generator,  # Pass generator to _sample_next_token
        )
        # ADD: Remove final logits
        del logits_CxV

        return pred_C

    def _generate_output(self, generated_codes: torch.Tensor) -> np.ndarray:
        num_channels = self.config.data.channels
        seq_length = generated_codes.shape[0]
        delay_pattern = self.config.data.delay_pattern
        audio_pad_value = self.config.data.audio_pad_value
        max_delay_pattern = max(delay_pattern)

        revert_precomp = build_revert_indices(
            B=1,
            T=seq_length,
            C=num_channels,
            delay_pattern=delay_pattern,
        )

        codebook = revert_audio_delay(
            audio_BxTxC=generated_codes.unsqueeze(0),
            pad_value=audio_pad_value,
            precomp=revert_precomp,
            T=seq_length,
        )[:, :-max_delay_pattern, :]

        # ADD: Clean up intermediate tensors
        del revert_precomp, generated_codes

        min_valid_index = 0
        max_valid_index = 1023
        invalid_mask = (codebook < min_valid_index) | (codebook > max_valid_index)
        codebook[invalid_mask] = 0

        # Process final audio
        audio = decode(self.dac_model, codebook.transpose(1, 2))

        # ADD: Clean up codebook
        del codebook, invalid_mask

        result = audio.squeeze().cpu().numpy()

        # ADD: Clean up audio tensor
        del audio
        torch.cuda.empty_cache()

        return result

    def load_audio(self, audio_path: str) -> torch.Tensor:
        audio, sr = torchaudio.load(audio_path, channels_first=True)  # C, T
        if sr != DEFAULT_SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, DEFAULT_SAMPLE_RATE)
        audio = audio.to(self.device).unsqueeze(0)  # 1, C, T
        audio_data = self.dac_model.preprocess(audio, DEFAULT_SAMPLE_RATE)
        _, encoded_frame, _, _, _ = self.dac_model.encode(audio_data)  # 1, C, T

        # Clean up intermediate tensors
        del audio_data
        if sr != DEFAULT_SAMPLE_RATE:
            del audio
        encoded_result = encoded_frame.squeeze(0).transpose(0, 1)
        del encoded_frame

        return encoded_result

    def save_audio(self, path: str, audio: np.ndarray):
        import soundfile as sf

        sf.write(path, audio, DEFAULT_SAMPLE_RATE)

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        max_tokens: int | None = None,
        cfg_scale: float = 3.0,
        temperature: float = 1.3,
        top_p: float = 0.95,
        use_torch_compile: bool = False,
        cfg_filter_top_k: int = 35,
        audio_prompt: str | torch.Tensor | None = None,
        audio_prompt_path: str | None = None,
        use_cfg_filter: bool | None = None,
        seed: int = 42,  # Added seed parameter
        verbose: bool = True,
        text_to_generate_size: int | None = None,
    ) -> np.ndarray:
        # Import tqdm here to avoid requiring it as a dependency if progress bar isn't used
        from tqdm import tqdm

        audio_eos_value = self.config.data.audio_eos_value
        audio_pad_value = self.config.data.audio_pad_value
        delay_pattern = self.config.data.delay_pattern

        # Estimate tokens based on text length (using the 6.19 tokens per char ratio)
        if audio_prompt:
            estimated_tokens = int(text_to_generate_size * 3.4)
        else:
            estimated_tokens = int(text_to_generate_size * 6.7)

        current_step = 0

        # Cap at model maximum
        max_tokens = self.config.data.audio_length if max_tokens is None else max_tokens

        max_delay_pattern = max(delay_pattern) if delay_pattern else 0
        self.model.eval()

        if audio_prompt_path:
            print("Warning: audio_prompt_path is deprecated. Use audio_prompt instead.")
            audio_prompt = audio_prompt_path
        if use_cfg_filter is not None:
            print("Warning: use_cfg_filter is deprecated.")

        # Create generator from seed if provided
        generator = None
        if seed >= 0:
            if verbose:
                print(f"Using seed: {seed} for generation")
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
        else:
            if verbose:
                print("Using random seed for generation")

        if verbose:
            total_start_time = time.time()

        # Prepare generation states
        dec_state, dec_output = self._prepare_generation(text, audio_prompt, verbose)
        dec_step = dec_output.prefill_step - 1

        bos_countdown = max_delay_pattern
        eos_detected = False
        eos_countdown = -1

        if use_torch_compile:
            step_fn = torch.compile(self._decoder_step, mode="default")
        else:
            step_fn = self._decoder_step

        if verbose:
            print("generate: starting generation loop")
            if use_torch_compile:
                print(
                    "generate: by using use_torch_compile=True, the first step would take long"
                )
            start_time = time.time()

        token_history = []  # Track generated tokens

        try:
            while dec_step < max_tokens:
                dec_state.prepare_step(dec_step)
                tokens_Bx1xC = (
                    dec_output.get_tokens_at(dec_step).unsqueeze(0).expand(2, -1, -1)
                )
                pred_C = step_fn(
                    tokens_Bx1xC,
                    dec_state,
                    cfg_scale,
                    temperature,
                    top_p,
                    cfg_filter_top_k,
                    generator=generator,  # Pass generator to step_fn
                )
                # Clean up tokens after use
                del tokens_Bx1xC

                if (
                    not eos_detected and pred_C[0] == audio_eos_value
                ) or dec_step == max_tokens - max_delay_pattern - 1:
                    eos_detected = True
                    eos_countdown = max_delay_pattern

                if eos_countdown > 0:
                    step_after_eos = max_delay_pattern - eos_countdown
                    for i, d in enumerate(delay_pattern):
                        if step_after_eos == d:
                            pred_C[i] = audio_eos_value
                        elif step_after_eos > d:
                            pred_C[i] = audio_pad_value
                    eos_countdown -= 1

                bos_countdown = max(0, bos_countdown - 1)
                dec_output.update_one(pred_C, dec_step + 1, bos_countdown > 0)

                # Add to token history
                token_history.append(pred_C.detach().clone())

                if eos_countdown == 0:
                    break

                dec_step += 1

                if verbose and dec_step % 86 == 0:
                    duration = time.time() - start_time
                    print(
                        f"generate step {dec_step}: speed={86 / duration:.3f} tokens/s, realtime factor={1 / duration:.3f}x"
                    )
                    start_time = time.time()

                    # Periodic memory cleanup during long generations
                    if torch.cuda.is_available() and dec_step > 500:
                        torch.cuda.empty_cache()

        except Exception as e:
            # Ensure cleanup even on error
            if "dec_state" in locals():
                del dec_state
            if "dec_output" in locals():
                del dec_output
            if "token_history" in locals():
                del token_history
            self.reset_state()
            raise e

        finally:
            # Close progress bar in finally block to ensure it's closed even on error
            # pbar.close()

            # Check if dec_output was created
            if "dec_output" in locals() and dec_output is not None:
                if dec_output.prefill_step >= dec_step + 1:
                    print("Warning: Nothing generated")
                    # Cleanup on early return
                    if "dec_state" in locals():
                        del dec_state
                    if "dec_output" in locals():
                        del dec_output
                    if "token_history" in locals():
                        del token_history
                    self.reset_state()
                    return None

                generated_codes = dec_output.generated_tokens[
                    dec_output.prefill_step : dec_step + 1, :
                ]

                # Clean up state objects
                if "dec_state" in locals():
                    del dec_state
                if "dec_output" in locals():
                    del dec_output
                if "token_history" in locals():
                    del token_history

                if verbose:
                    if "dec_output" in locals():
                        total_step = dec_step + 1 - dec_output.prefill_step
                        total_duration = time.time() - total_start_time
                        print(
                            f"generate: total step={total_step}, total duration={total_duration:.3f}s"
                        )

                # Process output
                output = self._generate_output(generated_codes)

                # Final cleanup
                del generated_codes
                self.reset_state()

                return output
            else:
                # Handle case where dec_output wasn't created
                self.reset_state()
                return None

    def reset_state(self):
        """Reset internal model state and clear CUDA cache to prevent memory leaks."""
        # 1. Clear any cached states in the model
        if hasattr(self, "dac_model") and hasattr(self.dac_model, "reset"):
            self.dac_model.reset()

        # 2. Clear any encoder/decoder buffers
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "buffers"):
            for buffer in self.model.encoder.buffers():
                if buffer.device.type == "cuda":
                    buffer.data = buffer.data.clone()  # Force copy to break references

        if hasattr(self.model, "decoder") and hasattr(self.model.decoder, "buffers"):
            for buffer in self.model.decoder.buffers():
                if buffer.device.type == "cuda":
                    buffer.data = buffer.data.clone()  # Force copy to break references

        # 3. Force garbage collection first
        import gc

        gc.collect()

        # 4. Then clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return True
