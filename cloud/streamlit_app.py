"""Community Cloud entry point; run from the repository root."""

from pathlib import Path
import sys
from threading import RLock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import torch

import demo_app


@st.cache_resource(show_spinner=False)
def inference_lock():
    # The cached model and its temporary Grad-CAM hooks are shared by viewers.
    # Serialize runs to avoid overlapping backward passes and memory spikes.
    return RLock()


# Community Cloud has CPUs, so keep each request's thread use modest.
torch.set_num_threads(2)

# results/ is a local handoff directory. Use the checkpoint tracked in Git LFS
# directly so a fresh cloud checkout needs no symlinks or manual file copies.
demo_app.MODEL_REGISTRY["ViT (frequency-hybrid)"]["default_checkpoint"] = str(
    ROOT / "artifacts/vit_wish_s42_ddp2_v1/best_model.pt"
)

with inference_lock():
    demo_app.main()
