# Streamlit Community Cloud

This entry point runs the existing Streamlit interface with inference-only,
CPU dependencies. It loads the trained ViT checkpoint directly from the
Git LFS artifact, without relying on the ignored results/ symlink.

## Deploy

Commit and push the deployment files, tests, and demo_app.py hook cleanup first.
In https://share.streamlit.io/, sign in with the GitHub account `devvicha`
and create an app with:

- Repository: `devvicha/Vision-Transformer-Fine-Tuning`
- Branch: `main`
- Main file path: `cloud/streamlit_app.py`
- Advanced settings / Python: `3.12`
- Secrets: none required

Community Cloud supports Git LFS. The model artifact must be uploaded to the
same repository, not just its pointer. For a local checkout, use `git lfs pull`.

## Local preview

From the repository root, in your existing Python environment:

```bash
python -m streamlit run cloud/streamlit_app.py
```

The cloud requirements pin Linux CPU wheels; keep the existing environment's
platform-appropriate PyTorch installation when previewing on macOS.

Run the startup, ViT probability parity, and repeated Grad-CAM cleanup checks:

```bash
python -m unittest discover -s tests -p test_streamlit_cloud.py
```

These checks use a synthetic image for pipeline consistency, not an accuracy
benchmark. The trained Git LFS checkpoint is required.

## Scope and limits

- Only ViT weights are currently committed. The other three models keep their
  existing disabled/missing-checkpoint behavior until their trained weights
  are supplied separately. No replacement or random weights are used.
- The optional MTCNN package is not installed. The existing OpenCV face detector
  and center-crop fallback are used, as in the current local environment.
- UI labels, layout, model outputs, threshold, and crop behavior are unchanged.
  The legacy frequency-hybrid label still refers to the plain trained ViT.
- Runs are serialized because viewers share the cached model and Grad-CAM hooks.
  ViT Grad-CAM hooks and parameter gradients are released after each run.
- This is a CPU demo, not a throughput guarantee. Check cloud memory and latency
  with representative uploads before sharing widely. Initial model loading and
  waking a sleeping app take longer than subsequent requests.
- To diagnose deployment failures, open Manage app and inspect the build/runtime
  logs. Check Git LFS download errors if the checkpoint cannot be loaded.

Official documentation:

- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization
