"""ComfyUI-Krea2Edit — in-context edit forward for the Krea2 model.

ComfyUI's native Krea2 `_forward` is text-to-image only: it builds the sequence
`[text | target]`. The krea2_edit LoRA (trained in ai-toolkit) needs the *appearance
path*: the VAE-encoded SOURCE latent prepended as a block of clean tokens, distinguished
from the (noisy) target purely by the 3-axis RoPE frame index (source=1..N, target=0, h/w
aligned). This node adds that by wrapping the model's DIFFUSION_MODEL forward and rebuilding
the sequence as `[text | sources(frame=1..N) | target(frame=0)]`, keeping only the target tokens
out — mirroring ai-toolkit's `predict_velocity_edit` exactly, using the model's own submodules.
"""
import math

import torch
import torch.nn.functional as F
from einops import rearrange

import comfy.patcher_extension
import comfy.utils
import comfy.ldm.common_dit
from comfy.ldm.flux.layers import timestep_embedding


def _imgids(bs, frame, h_, w_, device):
    ids = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
    ids[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
    return ids.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)


def _imgids_offset(bs, frame, gh, gw, th, tw, device):
    """Stride-1 integer positions at a centered continuous offset.

    Fractional center: integer floor placed odd-gap refs 8px off their true center.
    RoPE is continuous, so half-token positions are exact.
    """
    off_h, off_w = max(0.0, (th - gh) / 2.0), max(0.0, (tw - gw) / 2.0)
    ids = torch.zeros(gh, gw, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = (torch.arange(gh, device=device, dtype=torch.float32) + off_h)[:, None]
    ids[..., 2] = (torch.arange(gw, device=device, dtype=torch.float32) + off_w)[None, :]
    return ids.reshape(1, gh * gw, 3).repeat(bs, 1, 1)


REF_POSITIONS = ["aligned (trained)", "beside (right of target)", "bag of features"]


def _place_refs(ref_ids, modes, h_, w_):
    """Reference placement on the h/w RoPE axes (frame axis untouched: refs keep 1..N)."""
    if isinstance(modes, str):
        modes = [modes] * len(ref_ids)
    out, x0 = [], float(w_)
    for ids, mode in zip(ref_ids, modes):
        if mode.startswith("aligned"):
            out.append(ids)
            continue
        ids = ids.clone()
        if mode.startswith("beside"):
            ids[..., 2] = ids[..., 2] - ids[..., 2].min() + x0
            x0 = float(ids[..., 2].max()) + 1.0
        elif mode.startswith("bag"):
            ids[..., 1] = (h_ - 1) / 2.0
            ids[..., 2] = (w_ - 1) / 2.0
        else:
            raise ValueError(f"unknown ref_position {mode!r}")
        out.append(ids)
    return out


def _to_4d(v):
    if v.ndim == 5:
        b, c, t, h, w = v.shape
        return v.reshape(b * t, c, h, w)
    return v


def _fit_src(src, H, W):
    sh, sw = src.shape[-2:]
    if (sh, sw) == (H, W):
        return src
    s = max(H / sh, W / sw)
    ch, cw = min(sh, int(round(H / s))), min(sw, int(round(W / s)))
    y0, x0 = (sh - ch) // 2, (sw - cw) // 2
    src = src[..., y0:y0 + ch, x0:x0 + cw]
    return F.interpolate(src.float(), size=(H, W), mode="bilinear")


def _fit_encode_image(image, vae, H, W, cache, key, fit_mode="fit"):
    key = key + (fit_mode,)
    if key in cache:
        return cache[key]
    print(f"[krea2edit] _fit_encode_image: mode={fit_mode} in={tuple(image.shape)} target_latent={H}x{W}", flush=True)
    px_h, px_w = H * 8, W * 8
    img = image.movedim(-1, 1)  # B,H,W,C -> B,C,H,W
    ih, iw = img.shape[-2:]

    if fit_mode == "fit":
        sc = min(px_h / ih, px_w / iw)
        CROP_TOL = 0.08
        if ih * sc >= px_h * (1 - CROP_TOL) and iw * sc >= px_w * (1 - CROP_TOL):
            s = max(px_h / ih, px_w / iw)
            ch, cw = min(ih, int(round(px_h / s))), min(iw, int(round(px_w / s)))
            y0, x0 = (ih - ch) // 2, (iw - cw) // 2
            img = img[..., y0:y0 + ch, x0:x0 + cw]
            nh, nw = px_h, px_w
        else:
            nh = min(max(16, int(ih * sc) // 16 * 16), max(16, px_h // 16 * 16))
            nw = min(max(16, int(iw * sc) // 16 * 16), max(16, px_w // 16 * 16))
            ch2, cw2 = min(ih, max(1, int(round(nh / sc)))), min(iw, max(1, int(round(nw / sc))))
            y0, x0 = (ih - ch2) // 2, (iw - cw2) // 2
            img = img[..., y0:y0 + ch2, x0:x0 + ch2]

        img = F.interpolate(img.float(), size=(nh, nw), mode="bicubic", antialias=True)
        lat = vae.encode(img.movedim(1, -1)[..., :3].clamp(0, 1))
        cache[key] = lat
        return lat

    # Crop mode (legacy)
    s = max(px_h / ih, px_w / iw)
    ch, cw = min(ih, int(round(px_h / s))), min(iw, int(round(px_w / s)))
    y0, x0 = (ih - ch) // 2, (iw - cw) // 2
    img = img[..., y0:y0 + ch, x0:x0 + cw]
    img = F.interpolate(img.float(), size=(px_h, px_w), mode="bicubic", antialias=True)
    lat = vae.encode(img.movedim(1, -1)[..., :3].clamp(0, 1))
    cache[key] = lat
    return lat


def _ref_attn_bias(boosts, boost_mask, txtlen, slens, tgtlen, mask_hw, device, dtype):
    nsrc = len(slens)
    offs = [txtlen]
    for sl in slens:
        offs.append(offs[-1] + sl)
    rows0 = offs[-1]
    L = rows0 + tgtlen
    bias = torch.zeros(1, 1, L, L, device=device, dtype=dtype)

    for i, b in enumerate(boosts):
        if i >= nsrc:
            break
        if b is None or math.isnan(b) or b == 1.0:
            continue
        off, sl = offs[i], slens[i]

        if boost_mask is not None and i == nsrc - 1 and mask_hw is not None:
            mask = boost_mask[:1]
            if mask.ndim == 2:
                mask = mask[None]
            mask = F.interpolate(mask[None].float(), size=mask_hw[i], mode="area")[0, 0]
            cols = off + torch.nonzero(mask.reshape(-1) > 0.5, as_tuple=True)[0].to(device)
        else:
            cols = torch.arange(off, off + sl, device=device)

        bias[:, :, rows0:, cols] = math.log(max(b, 1e-4))
    return bias


def krea2_edit_forward(m, x, timesteps, context, src_latent, transformer_options,
                       ref_boosts=None, ref_boost_mask=None,
                       ref_native=False, pos_mode="anchor",
                       ref_positions=None):
    if ref_boosts is None:
        ref_boosts = [1.0]

    patch = m.patch
    temporal = x.ndim == 5
    if temporal:
        b5, c5, t5, h5, w5 = x.shape
    x = _to_4d(x)
    bs, c, H_orig, W_orig = x.shape

    x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch, patch), padding_mode="replicate")
    H, W = x.shape[-2], x.shape[-1]
    h_, w_ = H // patch, W // patch

    src_list = src_latent if isinstance(src_latent, (list, tuple)) else [src_latent]
    srcs = []
    for sl in src_list:
        src = _to_4d(sl).to(x.device, x.dtype)
        if src.shape[0] != bs:
            src = src[:1].expand(bs, *src.shape[1:])
        if not ref_native and src.shape[-2:] != (H, W):
            print(f"[krea2edit] LATENT-PATH fit_src (crop): src={tuple(src.shape[-2:])} -> {H}x{W}", flush=True)
            src = _fit_src(src, H, W).to(x.dtype)
        srcs.append(comfy.ldm.common_dit.pad_to_patch_size(src, (patch, patch), padding_mode="replicate"))
    src_grids = [(s_.shape[-2] // patch, s_.shape[-1] // patch) for s_ in srcs]

    context = m._unpack_context(context)

    tgt_img = m.first(rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch))
    src_imgs = [m.first(rearrange(s_, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch))
                for s_ in srcs]

    t = m.tmlp(timestep_embedding(timesteps, m.tdim).unsqueeze(1).to(tgt_img.dtype))
    tvec = m.tproj(t)

    context = m.txtfusion(context, mask=None, transformer_options=transformer_options)
    context = m.txtmlp(context)

    txtlen, tgtlen = context.shape[1], tgt_img.shape[1]
    srclen = sum(si.shape[1] for si in src_imgs)
    combined = torch.cat([context] + src_imgs + [tgt_img], dim=1)

    device = combined.device
    if pos_mode == "stride1" and ref_native:
        print(f"[krea2edit] STRIDE1-POS fit: ref grids {src_grids} centered in ({h_},{w_})", flush=True)
        ref_ids = [_imgids_offset(bs, i + 1, gh, gw, h_, w_, device)
                   for i, (gh, gw) in enumerate(src_grids)]
    else:
        ref_ids = [_imgids(bs, i + 1, gh, gw, device) for i, (gh, gw) in enumerate(src_grids)]

    # Применяем позиции для активных референсов
    if ref_positions is not None:
        modes = ref_positions[:len(ref_ids)]
        ref_ids = _place_refs(ref_ids, modes, h_, w_)

    pos = torch.cat([
        torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)]
        + ref_ids
        + [_imgids(bs, 0, h_, w_, device)],
        dim=1)
    freqs = m.pe_embedder(pos)

    attn_bias = None
    clean_boosts = [1.0 if (b is None or math.isnan(b)) else b for b in ref_boosts]
    if any(b != 1.0 for b in clean_boosts):
        attn_bias = _ref_attn_bias(clean_boosts, ref_boost_mask, txtlen,
                                   [si.shape[1] for si in src_imgs], tgtlen,
                                   src_grids, combined.device, combined.dtype)

    for block in m.blocks:
        combined = block(combined, tvec, freqs, attn_bias, transformer_options=transformer_options)

    final = m.last(combined, t)
    out = final[:, txtlen + srclen: txtlen + srclen + tgtlen, :]
    out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                    h=h_, w=w_, ph=patch, pw=patch, c=m.channels)
    out = out[:, :, :H_orig, :W_orig]
    if temporal:
        out = out.reshape(b5, t5, m.channels, H_orig, W_orig).movedim(1, 2)
    return out


class Krea2EditModelPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
        }, "optional": {
            "source_latent": ("LATENT", {"tooltip": "1st reference (scene)"}),
            "source_latent_b": ("LATENT", {"tooltip": "2nd reference (subject photo)"}),
            "source_latent_c": ("LATENT", {"tooltip": "3rd reference"}),
            "source_latent_d": ("LATENT", {"tooltip": "4th reference"}),
            "source_latent_e": ("LATENT", {"tooltip": "5th reference"}),
            "ref_boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01, "round": 0.001, "tooltip": "Boost for 1st reference"}),
            "ref_boost_b": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01, "round": 0.001, "tooltip": "Boost for 2nd reference"}),
            "ref_boost_c": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01, "round": 0.001, "tooltip": "Boost for 3rd reference"}),
            "ref_boost_d": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01, "round": 0.001, "tooltip": "Boost for 4th reference"}),
            "ref_boost_e": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01, "round": 0.001, "tooltip": "Boost for 5th reference"}),
            "fit_mode": (["fit", "crop (legacy)"], {"default": "fit"}),
            "ref_boost_mask": ("MASK", {"tooltip": "Region on the last active reference to boost"}),
            "vae": ("VAE", {"tooltip": "RECOMMENDED with source_image: blur-proof pixel-space path"}),
            "source_image": ("IMAGE", {"tooltip": "1st reference as IMAGE"}),
            "source_image_b": ("IMAGE", {"tooltip": "2nd reference as IMAGE"}),
            "source_image_c": ("IMAGE", {"tooltip": "3rd reference as IMAGE"}),
            "source_image_d": ("IMAGE", {"tooltip": "4th reference as IMAGE"}),
            "source_image_e": ("IMAGE", {"tooltip": "5th reference as IMAGE"}),
            "target_latent": ("LATENT", {"tooltip": "RECOMMENDED with vae + source_image: wire the SAME latent you feed KSampler.latent_image"}),
            "ref_position": (REF_POSITIONS, {"default": REF_POSITIONS[0], "tooltip": "Placement for 1st reference"}),
            "ref_position_b": (REF_POSITIONS, {"default": REF_POSITIONS[0], "tooltip": "Placement for 2nd reference"}),
            "ref_position_c": (REF_POSITIONS, {"default": REF_POSITIONS[0], "tooltip": "Placement for 3rd reference"}),
            "ref_position_d": (REF_POSITIONS, {"default": REF_POSITIONS[0], "tooltip": "Placement for 4th reference"}),
            "ref_position_e": (REF_POSITIONS, {"default": REF_POSITIONS[0], "tooltip": "Placement for 5th reference"}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "krea2edit"
    DESCRIPTION = "Adds the krea2_edit in-context source-preservation path to a Krea2 model (supports up to 5 references)."

    def patch(self, model, source_latent=None, source_latent_b=None, source_latent_c=None,
              source_latent_d=None, source_latent_e=None, ref_boost=1.0, ref_boost_b=1.0,
              ref_boost_c=1.0, ref_boost_d=1.0, ref_boost_e=1.0,
              ref_boost_mask=None, vae=None, source_image=None, source_image_b=None,
              source_image_c=None, source_image_d=None, source_image_e=None,
              fit_mode="fit", target_latent=None,
              ref_position="aligned (trained)", ref_position_b="aligned (trained)",
              ref_position_c="aligned (trained)", ref_position_d="aligned (trained)",
              ref_position_e="aligned (trained)", **_future):

        # Обработка обратной совместимости для старого ref_position_a из воркфлоу на 2 референса
        ref_position_a = _future.pop("ref_position_a", None)

        if _future:
            print(f"[krea2edit] WARNING: workflow provides inputs this node version does not "
                  f"know ({', '.join(sorted(_future))}). Continuing without them.", flush=True)

        m = model.clone()
        mm = model.model

        # Collect latents
        lats_raw = [source_latent, source_latent_b, source_latent_c, source_latent_d, source_latent_e]
        active_lats = [l for l in lats_raw if l is not None]

        src_samples = []
        for l in active_lats:
            src_samples.append(mm.process_latent_in(l["samples"]))
        if len(src_samples) == 1:
            src_samples = src_samples[0]

        # Collect inputs & positions
        raw_inputs = [
            (source_image, ref_boost, ref_position),
            (source_image_b, ref_boost_b, ref_position_b),
            (source_image_c, ref_boost_c, ref_position_c),
            (source_image_d, ref_boost_d, ref_position_d),
            (source_image_e, ref_boost_e, ref_position_e),
        ]

        active_imgs = []
        active_boosts = []
        active_positions = []

        for idx, (img, boost, pos) in enumerate(raw_inputs):
            # Проверка, активен ли текущий слот (через image или latent)
            is_active = (img is not None) or (idx < len(active_lats))
            if is_active:
                b_val = 1.0 if (boost is None or math.isnan(boost)) else boost
                active_boosts.append(b_val)
                active_positions.append(pos)
                if img is not None:
                    active_imgs.append(img)

        # Если использовался старый параметр ref_position_a, переназначаем позиции для ранних референсов
        if ref_position_a is not None and len(active_positions) > 1:
            for i in range(len(active_positions) - 1):
                active_positions[i] = ref_position_a

        px_cache = {}
        state = {"announced": False}

        if fit_mode == "fit" and (vae is None or len(active_imgs) == 0):
            print(f"[krea2edit] WARNING: fit_mode='fit' has NO EFFECT without both 'vae' and 'source_image' connected. Falling back to latent path.", flush=True)

        # Pre-encode outside sampling loop
        primed = None
        if vae is not None and len(active_imgs) > 0:
            if target_latent is not None:
                Hh, Ww = target_latent["samples"].shape[-2], target_latent["samples"].shape[-1]
                print(f"[krea2edit] pre-encoding {len(active_imgs)} sources at target {Hh * 8}x{Ww * 8}px (fit_mode={fit_mode})", flush=True)
                for idx, img in enumerate(active_imgs):
                    _fit_encode_image(img, vae, Hh, Ww, px_cache, (f"ref_{idx}", Hh, Ww), fit_mode)
                primed = (Hh, Ww)
            else:
                print("[krea2edit] NOTE: connect 'target_latent' to pre-encode sources before sampling and avoid VRAM thrashing.", flush=True)

        def wrapper(executor, x, timesteps, context, *wargs, **kwargs):
            transformer_options = kwargs.pop("transformer_options", None)
            if transformer_options is None:
                transformer_options = {}
                for a in reversed(wargs):
                    if isinstance(a, dict):
                        transformer_options = a
                        break
            dm = executor.class_obj
            src = src_samples

            if vae is not None and len(active_imgs) > 0:
                xx = _to_4d(x)
                Hh, Ww = xx.shape[-2], xx.shape[-1]
                if not state["announced"]:
                    state["announced"] = True
                    print(f"[krea2edit] pixel path ACTIVE ({len(active_imgs)} refs, fit_mode={fit_mode})", flush=True)
                    if primed is not None and primed != (Hh, Ww):
                        print(f"[krea2edit] WARNING: 'target_latent' pre-encode misaligned with sampling resolution ({primed} vs {Hh},{Ww}).", flush=True)

                encoded_list = []
                for idx, img in enumerate(active_imgs):
                    encoded = _fit_encode_image(img, vae, Hh, Ww, px_cache, (f"ref_{idx}", Hh, Ww), fit_mode)
                    encoded_list.append(mm.process_latent_in(encoded))
                src = encoded_list if len(encoded_list) > 1 else encoded_list[0]

            v = krea2_edit_forward(dm, x, timesteps, context, src, transformer_options,
                                   ref_boosts=active_boosts,
                                   ref_boost_mask=ref_boost_mask,
                                   ref_native=(fit_mode == "fit" and vae is not None and len(active_imgs) > 0),
                                   pos_mode=("stride1" if fit_mode == "fit" else "anchor"),
                                   ref_positions=active_positions)
            return v

        to = m.model_options.setdefault("transformer_options", {})
        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "krea2_edit", wrapper, to
        )
        return (m,)


class Krea2EditGroundedEncode:
    DEFAULT_SYSTEM = (
        "Describe the image by detailing the color, shape, size, "
        "texture, quantity, text, spatial relationships of the objects and background:"
    )

    @classmethod
    def _template(cls, nimg, system_prompt=""):
        sp = system_prompt.strip() or cls.DEFAULT_SYSTEM
        vis = "<|vision_start|><|image_pad|><|vision_end|>" * nimg
        return ("<|im_start|>system\n" + sp + "<|im_end|>\n<|im_start|>user\n"
                + vis + "{}<|im_end|>\n<|im_start|>assistant\n")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "1st reference image"}),
                "image_b": ("IMAGE", {"tooltip": "2nd reference image"}),
                "image_c": ("IMAGE", {"tooltip": "3rd reference image"}),
                "image_d": ("IMAGE", {"tooltip": "4th reference image"}),
                "image_e": ("IMAGE", {"tooltip": "5th reference image"}),
                "grounding_px": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 64}),
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "krea2edit"
    DESCRIPTION = "Encodes the edit instruction grounded on up to 5 source images."

    def _prep(self, image, grounding_px):
        samples = image.movedim(-1, 1)
        h, w = samples.shape[2], samples.shape[3]
        if grounding_px and max(h, w) > grounding_px:
            s = grounding_px / max(h, w)
            samples = comfy.utils.common_upscale(samples, round(w * s), round(h * s), "area", "disabled")
        return samples.movedim(1, -1)[:, :, :, :3]

    def encode(self, clip, prompt, image=None, image_b=None, image_c=None,
               image_d=None, image_e=None, grounding_px=768, system_prompt="", **_future):

        if _future:
            print(f"[krea2edit] WARNING: workflow provides inputs this node version does not "
                  f"know ({', '.join(sorted(_future))}). Continuing without them.", flush=True)

        raw_imgs = [image, image_b, image_c, image_d, image_e]
        raw_imgs = [i for i in raw_imgs if i is not None]

        if not raw_imgs:
            tokens = clip.tokenize(prompt)
            return (clip.encode_from_tokens_scheduled(tokens),)

        imgs = [self._prep(img, grounding_px) for img in raw_imgs]
        template = self._template(len(imgs), system_prompt)
        tokens = clip.tokenize(prompt, images=imgs, llama_template=template)
        return (clip.encode_from_tokens_scheduled(tokens),)


NODE_CLASS_MAPPINGS = {
    "Krea2EditModelPatch": Krea2EditModelPatch,
    "Krea2EditGroundedEncode": Krea2EditGroundedEncode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2EditModelPatch": "Krea2 Edit (source patch)",
    "Krea2EditGroundedEncode": "Krea2 Edit (grounded encode)",
}


def _pack_version():
    try:
        import os, re
        p = os.path.join(os.path.dirname(__file__), "pyproject.toml")
        with open(p) as f:
            m = re.search(r'^version\s*=\s*"([^"]+)"', f.read(), re.M)
        return m.group(1) if m else "unknown"
    except Exception:
        return "unknown"


print(f"[krea2edit] nodes v{_pack_version()} loaded (5-ref multi-input with individual ref_position)", flush=True)
