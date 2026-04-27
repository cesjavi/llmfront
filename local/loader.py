"""
local/loader.py – Hilo de carga de modelos locales con Transformers/PyTorch.
"""
import logging
from pathlib import Path
from typing import Optional

from state import _model_lock, _download_lock, _download_state, _loaded_models

logger = logging.getLogger("llmfront")


def _load_model_thread(
    model_id: str,
    quantization: str,
    device: str,
    model_key_fn,
    model_local_dir_fn,
    dir_size_gb_fn,
    is_partial_fn,
    extract_meta_fn,
    system_info_fn,
):
    """Carga un modelo en memoria. Recibe helpers como args para evitar imports circulares."""
    key = model_key_fn(model_id)
    local_dir = model_local_dir_fn(model_id)

    try:
        if is_partial_fn(model_id):
            size_saved = dir_size_gb_fn(local_dir)
            with _model_lock:
                _download_state[key + "_load"] = {
                    "status": "error",
                    "progress": 0,
                    "message": f"Descarga incompleta ({size_saved} GB guardados). Reanudá la descarga primero.",
                    "model_id": model_id,
                }
            return

        with _model_lock:
            _download_state[key + "_load"] = {
                "status": "loading",
                "progress": 10,
                "message": "Importando transformers...",
                "model_id": model_id,
            }

        # Parche para bug de transformers leyendo la versión de gguf
        try:
            import gguf
            gguf.__version__ = "0.18.0"
        except ImportError:
            pass

        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:
            AutoModelForImageTextToText = None

        config = AutoConfig.from_pretrained(str(local_dir), trust_remote_code=True)
        model_meta = extract_meta_fn(model_id, config=config.to_dict())
        supports_vision = model_meta["supports_vision"]
        if supports_vision and AutoModelForImageTextToText is None:
            logger.warning(
                "Installed transformers does not provide AutoModelForImageTextToText; "
                "falling back to text-only loading for %s", model_id,
            )
            supports_vision = False
            model_meta["supports_vision"] = False
            model_meta["capability_label"] = "Solo texto"

        use_cuda = torch.cuda.is_available() and device != "cpu"
        if use_cuda:
            target_device = torch.device("cuda")
            dtype = torch.float16
        else:
            target_device = torch.device("cpu")
            dtype = torch.float32
            try:
                import transformers.utils.import_utils as _tf_import_utils
                _tf_import_utils._bitsandbytes_available = False
            except Exception:
                pass

        # Hardware safety check
        sys_info = system_info_fn()
        vram_avail = sys_info.get("vram_gb", 0) if sys_info.get("cuda_available", False) else 0
        ram_avail = sys_info.get("ram_available_gb", 0)
        size_gb = dir_size_gb_fn(local_dir)

        if quantization == "8bit":
            required_gb = size_gb * 0.55
        elif quantization == "4bit":
            required_gb = size_gb * 0.3
        else:
            required_gb = size_gb
        required_gb += 1.0

        if use_cuda:
            if quantization == "none":
                if required_gb > ram_avail:
                    raise RuntimeError(f"RAM insuficiente: Requiere ~{required_gb:.1f} GB, disponible {ram_avail:.1f} GB.")
                if required_gb > vram_avail:
                    raise RuntimeError(f"VRAM insuficiente: Requiere ~{required_gb:.1f} GB, disponible {vram_avail:.1f} GB. Usa cuantización 4bit u 8bit.")
            else:
                if required_gb > vram_avail:
                    raise RuntimeError(f"VRAM insuficiente para {quantization}: Requiere ~{required_gb:.1f} GB, disponible {vram_avail:.1f} GB.")
        else:
            if quantization in ("4bit", "8bit"):
                raise RuntimeError("La cuantización 4bit/8bit requiere una GPU.")
            if required_gb > ram_avail:
                raise RuntimeError(f"RAM insuficiente: Requiere ~{required_gb:.1f} GB, disponible {ram_avail:.1f} GB.")

        # Detección de archivos incompatibles
        litert_files = list(local_dir.rglob("*.litertlm"))
        safetensors = list(local_dir.rglob("*.safetensors"))
        if litert_files and not safetensors:
            raise RuntimeError("Este es un modelo LiteRT (.litertlm). No puede cargarse con transformers.")

        with _model_lock:
            _download_state[key + "_load"]["message"] = "Cargando tokenizer/procesador..."
            _download_state[key + "_load"]["progress"] = 25

        gguf_files = list(local_dir.rglob("*.gguf"))
        gguf_kwargs = {}
        is_gguf = len(gguf_files) > 0
        if is_gguf:
            selected_gguf = gguf_files[0]
            if quantization == "4bit":
                q4 = [f for f in gguf_files if "q4" in f.name.lower()]
                if q4: selected_gguf = q4[0]
            elif quantization == "8bit":
                q8 = [f for f in gguf_files if "q8" in f.name.lower()]
                if q8: selected_gguf = q8[0]
            gguf_kwargs["gguf_file"] = selected_gguf.name
            logger.info(f"Detectado modelo GGUF: usando archivo {selected_gguf.name}")

        processor = None
        tokenizer = None
        if supports_vision and not is_gguf:
            processor = AutoProcessor.from_pretrained(str(local_dir), trust_remote_code=True)
            tokenizer = getattr(processor, "tokenizer", None)
        else:
            tokenizer = AutoTokenizer.from_pretrained(str(local_dir), trust_remote_code=True, **gguf_kwargs)

        with _model_lock:
            _download_state[key + "_load"]["message"] = "Cargando modelo (puede tardar varios minutos)..."
            _download_state[key + "_load"]["progress"] = 40

        model_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
        model_kwargs.update(gguf_kwargs)

        if quantization in ("4bit", "8bit") and not is_gguf:
            try:
                from transformers import BitsAndBytesConfig
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=(quantization == "4bit"),
                    load_in_8bit=(quantization == "8bit"),
                    bnb_4bit_compute_dtype=torch.float16 if quantization == "4bit" else None,
                )
                model_kwargs["quantization_config"] = bnb_cfg
                model_kwargs["device_map"] = "auto"
                target_device = None
            except ImportError:
                logger.warning("bitsandbytes no instalado, cargando sin cuantización")
                quantization = "none"

        if quantization == "none" or is_gguf:
            model_kwargs["torch_dtype"] = dtype

        try:
            model_cls = AutoModelForImageTextToText if supports_vision and not is_gguf else AutoModelForCausalLM
            model = model_cls.from_pretrained(str(local_dir), **model_kwargs)
        except Exception as load_err:
            logger.warning(f"First load attempt failed ({load_err}), retrying without dtype hints...")
            model_cls = AutoModelForImageTextToText if supports_vision and not is_gguf else AutoModelForCausalLM
            model = model_cls.from_pretrained(str(local_dir), trust_remote_code=True)

        if target_device is not None:
            logger.info(f"Moving model to {target_device}")
            try:
                model = model.to(target_device)
            except Exception as move_err:
                logger.warning(f"Could not move model to {target_device}: {move_err}. Staying on CPU.")

        model.eval()

        with _model_lock:
            _download_state[key + "_load"]["message"] = "Finalizando..."
            _download_state[key + "_load"]["progress"] = 90

        with _model_lock:
            _loaded_models[key] = {
                "tokenizer": tokenizer,
                "processor": processor,
                "model": model,
                "model_id": model_id,
                "quantization": quantization,
                "supports_vision": supports_vision,
                "model_type": model_meta["model_type"],
            }
            _download_state[key + "_load"] = {
                "status": "loaded",
                "progress": 100,
                "message": "✓ Modelo cargado y listo" + (" (vision + texto)" if supports_vision else ""),
                "model_id": model_id,
            }
        logger.info(f"Model loaded: {model_id}")

    except Exception as e:
        logger.error(f"Load error [{model_id}]: {e}")
        with _model_lock:
            _download_state[key + "_load"] = {
                "status": "error",
                "progress": 0,
                "message": str(e),
                "model_id": model_id,
            }
