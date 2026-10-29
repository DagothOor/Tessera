# -*- coding: utf-8 -*-
"""
Texelith — SuperMat worker. Runs INSIDE the SuperMat environment, not Texelith's.

SuperMat (Hong et al., ICCV 2025; code https://github.com/hyj542682306/SuperMat,
weights https://huggingface.co/oyiya/SuperMat, both MIT) splits a picture into
albedo, roughness and metalness in one denoising step on top of Stable
Diffusion 2.1 (base model ``sd2-community/stable-diffusion-2-1``, CreativeML
Open RAIL++-M licence). It does not give a normal map.

This script follows the authors' inference_supermat.py exactly in how the
model is built and called (SuperMatStableDiffusionPipeline +
SuperMatAdapterWrapper + the UNet state dict, DDIM with trailing spacing, one
step, empty prompt, source image as a 1×3×H×W tensor 0…1), and adds two
things Texelith needs:

  * one process serves many textures (loading takes far longer than a pass);
  * seamless output. The model was trained on objects, so like any image
    network it treats the frame edge as an edge. The texture is therefore run
    four times, rolled by 0 or half its size along each axis, and the four
    answers are blended with the weight sin²(π·y/S)·sin²(π·x/S) in each
    pass's own frame: full weight in the middle, zero at the frame edges.
    The four weights add up to exactly 1, so every seam of one pass is
    covered by the middle of another.

Protocol:

  argv: <SuperMat repo folder> [<supermat.pth>] [<base model id or folder>]
        an empty checkpoint means <repo>/checkpoints/supermat.pth, or, if that
        is missing, a download from Hugging Face (oyiya/SuperMat);
        an empty base means sd2-community/stable-diffusion-2-1 (downloaded
        into the Hugging Face cache on first use, about 5 GB).
  stdout: lines starting with '@@' are for Texelith, the rest is library
  output. '@@{"ready": true, "device": "cuda"}' once loaded; then for each
  stdin line '{"in": a.npy, "out": b.npz}': '@@{"ok": true}' or
  '@@{"ok": false, "error": "..."}'.

Input .npy: S×S×3 float32 sRGB 0…1 (Texelith resizes on the torus first).
Output .npz, same S×S: basecolor (×3), roughness, metalness — float32 0…1.

Texelith — рабочий процесс SuperMat. Запускается питоном окружения SuperMat.
Модель собирается ровно как в inference_supermat.py авторов; плюс четыре
прохода со сдвигом на полпериода, чтобы карты были бесшовными.
"""

import json
import os
import sys
import traceback

BASE_DEFAULT = 'sd2-community/stable-diffusion-2-1'


def say(obj):
    """Print one protocol line (prefixed with '@@') and flush."""
    sys.stdout.write('@@' + json.dumps(obj) + '\n')
    sys.stdout.flush()


def find_checkpoint(repo, given):
    """Path of supermat.pth: the given file, the repo's checkpoints folder,
    or a download from Hugging Face (into its cache)."""
    if given:
        return given
    local = os.path.join(repo, 'checkpoints', 'supermat.pth')
    if os.path.isfile(local):
        return local
    from huggingface_hub import hf_hub_download
    return hf_hub_download('oyiya/SuperMat', 'supermat.pth')


def build(repo, ckpt, base, dev, dtype=None):
    """The pipeline, built as in the authors' inference_supermat.py.

    dtype: torch.float16 on a graphics card (about twice as fast and half the
    memory; one_pass falls back to float32 if a pass comes out broken), else
    float32. TEXELITH_SM_FP32=1 in the environment forces float32."""
    import torch
    if dtype is None:
        dtype = torch.float32
    from diffusers import DDIMScheduler
    sys.path.insert(0, repo)
    from src.adapters import SuperMatAdapterWrapper
    from src.pipelines.pipeline_supermat_stable_diffusion import \
        SuperMatStableDiffusionPipeline

    pipe = SuperMatStableDiffusionPipeline.from_pretrained(
        base, safety_checker=None, requires_safety_checker=False, torch_dtype=dtype).to(dev)
    pipe = SuperMatAdapterWrapper.convert(pipe, use_camera_embeddings=False,
                                          camera_embeddings_dim=16)
    try:
        state = torch.load(ckpt, map_location='cpu', weights_only=True)
    except Exception:
        # файл не из одних тензоров: читаем как авторы в старых torch (полным
        # распаковщиком). Это веса из репозитория авторов, не чужой файл
        state = torch.load(ckpt, map_location='cpu', weights_only=False)
    pipe.unet.load_state_dict(state, strict=False)
    pipe.unet.to(dtype)
    pipe.unet.eval()
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config,
                                               timestep_spacing='trailing')
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(dev)


def one_pass(pipe, img, dev):
    """One SuperMat pass on an S×S×3 array 0…1 → (albedo S×S×3, orm S×S×3).

    In half precision a pass can overflow and come out as NaN; then the
    pipeline is switched to float32 for good and the pass repeated."""
    import numpy as np
    import torch
    out = _call(pipe, img, dev)
    if not all(bool(torch.isfinite(o).all()) for o in out[:2]) and pipe.unet.dtype != torch.float32:
        pipe.to(torch.float32)
        say({'note': 'float16 gave NaN, switched to float32'})
        out = _call(pipe, img, dev)

    def hwc(x):
        """Batch or single CHW tensor → H×W×C float32 numpy, clipped 0…1."""
        x = x.detach().float().cpu().clamp(0.0, 1.0)
        if x.ndim == 4:
            x = x[0]
        return x.permute(1, 2, 0).numpy()
    return hwc(out[0]), hwc(out[1])


def _call(pipe, img, dev):
    """Run the pipeline once on an S×S×3 array; returns its output list."""
    import numpy as np
    import torch
    t = torch.from_numpy(np.ascontiguousarray(img, np.float32)).permute(2, 0, 1)[None]
    t = t.to(dev, dtype=pipe.unet.dtype)
    # use_fp32_input: the authors' pipeline turns the text features into
    # float32 by default; with a float16 model that clashes, so keep them as is
    with torch.no_grad():
        return pipe(prompt='', num_inference_steps=1, source_image=t, output_type='pt',
                    use_fp32_input=pipe.unet.dtype == torch.float32)


def seamless(pipe, x, dev):
    """Four passes rolled by half a period, blended so the result tiles."""
    import numpy as np
    S, T = x.shape[:2]
    wy = np.sin(np.pi * (np.arange(S) + 0.5) / S)[:, None] ** 2
    wx = np.sin(np.pi * (np.arange(T) + 0.5) / T)[None, :] ** 2
    w = (wy * wx)[..., None].astype(np.float32)
    alb = np.zeros((S, T, 3), np.float32)
    orm = np.zeros((S, T, 3), np.float32)
    for sy in (0, S // 2):
        for sx in (0, T // 2):
            a, o = one_pass(pipe, np.roll(x, (-sy, -sx), (0, 1)), dev)
            # обратно в координаты текстуры: вес и ответ едут вместе
            alb += np.roll(a * w, (sy, sx), (0, 1))
            orm += np.roll(o * w, (sy, sx), (0, 1))
    # сумма четырёх весов sin²·sin² со сдвигами на полпериода равна 1
    return alb, orm


def main():
    """Load SuperMat once, then serve requests from stdin until EOF."""
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    repo = os.path.abspath(sys.argv[1])
    ckpt_arg = sys.argv[2] if len(sys.argv) > 2 else ''
    base = (sys.argv[3] if len(sys.argv) > 3 else '') or BASE_DEFAULT
    try:
        import numpy as np
        import torch
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        half = dev.type == 'cuda' and os.environ.get('TEXELITH_SM_FP32') != '1'
        pipe = build(repo, find_checkpoint(repo, ckpt_arg), base, dev,
                     torch.float16 if half else torch.float32)
    except Exception as e:                      # noqa: BLE001 — отдаём текст в Texelith
        traceback.print_exc()
        say({'ready': False, 'error': '%s: %s' % (e.__class__.__name__, e)})
        return 1
    name = torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'
    name += ' (fp16)' if half else ''
    say({'ready': True, 'device': name})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            x = np.clip(np.load(req['in']).astype(np.float32)[..., :3], 0, 1)
            alb, orm = seamless(pipe, x, dev)
            # ORM: зелёный — шероховатость, синий — металличность (src/utils.py авторов)
            np.savez(req['out'], basecolor=alb, roughness=orm[..., 1],
                     metalness=orm[..., 2])
            say({'ok': True})
        except Exception as e:                  # noqa: BLE001
            traceback.print_exc()
            say({'ok': False, 'error': '%s: %s' % (e.__class__.__name__, e)})
    return 0


if __name__ == '__main__':
    sys.exit(main())
