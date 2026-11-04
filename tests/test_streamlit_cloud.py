"""Deployment smoke checks; run from the repository root with Git LFS pulled."""

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class CloudSmokeTest(unittest.TestCase):
    def test_upload_screen(self):
        entrypoint = Path(__file__).resolve().parents[1] / "cloud/streamlit_app.py"
        app = AppTest.from_file(str(entrypoint)).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(
            app.title[0].value,
            "Cross-Generator Deepfake Face Detection — Live Demo",
        )

    def test_vit_predictions_and_gradcam_cleanup(self):
        app = AppTest.from_string('''
import runpy
runpy.run_path("cloud/streamlit_app.py", run_name="__main__")
from unittest.mock import patch
import torch
from PIL import Image
import demo_app
from models.vit.runtime import predict_image

name = "ViT (frequency-hybrid)"
assert demo_app.MODEL_REGISTRY[name]["default_checkpoint"].endswith(
    "artifacts/vit_wish_s42_ddp2_v1/best_model.pt"
)
demo_app.load_model.clear()
with patch.object(demo_app, "get_device", return_value=torch.device("cpu")):
    model, device, cfg = demo_app.load_model(name)
image = Image.new("RGB", (299, 299), (100, 140, 170))
reference = predict_image(
    model, image, cfg[demo_app.VIT_METADATA_KEY]["preprocessing"], device
)
original = demo_app.vit_probabilities
observed = []
def capture(logits):
    result = original(logits)
    observed.append(float(result[0]))
    return result
layer = dict(model.named_modules())[demo_app.MODEL_REGISTRY[name]["gradcam_layer"]]
before = (len(layer._forward_hooks), len(layer._backward_hooks))
with patch.object(demo_app, "vit_probabilities", side_effect=capture), \
     patch.object(demo_app.st, "image", wraps=demo_app.st.image) as render:
    for _ in range(2):
        demo_app.run_inference(name, image)
        assert (len(layer._forward_hooks), len(layer._backward_hooks)) == before
        assert all(p.grad is None for p in model.parameters())
    assert render.call_count == 2
    assert all(isinstance(call.args[0], Image.Image) for call in render.call_args_list)
assert len(observed) == 2
assert all(abs(value - reference) < 1e-6 for value in observed)
demo_app.load_model.clear()
''').run(timeout=90)
        self.assertFalse(app.exception)
        self.assertFalse(any("unavailable" in caption.value.lower()
                             for caption in app.caption))


if __name__ == "__main__":
    unittest.main()
