# Learning 3D Gaussian Splatting and Change Detection on Kaggle

This notebook documents a step-by-step exploration of **3D Gaussian Splatting (3DGS)** using Nerfstudio’s Splatfacto model on Kaggle.

Starting with environment setup and CUDA compilation, we trained a reconstruction, rendered the scene, inspected its Gaussian representation, and performed controlled experiments to understand how geometry, color, camera position, and lighting affect change detection.

The experiments use a chair scene from the Nerfstudio `poster` dataset. They are educational tests, not a validated system for measuring physical change.

## Environment

| Component | Version / hardware |
|---|---|
| Platform | Kaggle Notebook |
| Python | 3.12.13 |
| PyTorch | 2.10.0+cu128 |
| CUDA toolkit | 12.8 |
| GPUs | 2 × NVIDIA Tesla T4 |
| GPU memory | Approximately 14.56 GiB per GPU |
| System RAM | Approximately 31.3 GiB |
| Nerfstudio | 1.1.5 |
| gsplat | 1.4.0 |
| NumPy | 1.26.4 |

Training and rendering used one GPU. The two GPUs’ memory was not combined.

## 1. Environment and GPU Verification

We inspected Python, PyTorch, CUDA, GPU memory, system RAM, and the CUDA compiler.

A small matrix multiplication on the GPU verified that CUDA computation worked before attempting training.

## 2. Installation and Dependency Checks

We installed the reconstruction dependencies and checked imports for PyTorch, OpenCV, NumPy, and gsplat.

Installation reported conflicts with several packages preinstalled on Kaggle. Successful imports did not guarantee that every package in the environment was compatible, so we tested the components needed by this notebook directly.

## 3. CUDA Compilation and Renderer Verification

The first Gaussian render triggered compilation of gsplat’s CUDA extension.

An initial timeout interrupted the build. Inspection of compiler processes, object files, lock files, and build logs showed that compilation had been progressing.

We resumed compilation with limited parallelism and reused completed build files. After the compiled library was available, a small renderer test passed both:

- Gaussian rendering.
- Backward gradient computation.

This established that the rendering backend worked before training.

## 4. Dataset Inspection and Preparation

The dataset metadata referenced **226 frames**, but the available dataset contained only **100 original images and 100 reduced-resolution images**.

We checked:

- Referenced image paths.
- Missing images.
- Original and reduced image dimensions.
- Camera intrinsics.
- Availability of the sparse point cloud.

We prepared a dataset containing only frames whose required images were present.

The original images were 1080 × 1920 pixels; the reduced images were 270 × 480 pixels, corresponding to a downscale factor of four.

## 5. Training Smoke Test

We ran a short **100-iteration Splatfacto training test** and confirmed that it produced a configuration file and checkpoint.

A command-line argument error was resolved by using the correct `nerfstudio-data` subcommand and placing its options after that subcommand.

## 6. Initial Reconstruction Training

We completed a **3,000-iteration training run** using the prepared dataset and reduced-resolution images.

This produced a trained Gaussian reconstruction and a checkpoint for subsequent rendering and analysis.

This was an initial learning run, rather than a claim of fully converged reconstruction quality.

## 7. Dataset Rendering and Video Rendering

We rendered held-out dataset views and compared reconstructed RGB images with their reference images.

We also rendered an interpolated camera trajectory around the chair. The video showed a recognizable chair and surrounding scene, with occasional floating or cloudy artifacts.

Compatibility fixes included:

- Using `--rendered-output-names` for the installed rendering CLI.
- Allowing full checkpoint loading for our own trusted checkpoints.
- Using fresh Python subprocesses to avoid mixed NumPy imports after package changes.

## 8. Inspecting the Learned Gaussian Parameters

We inspected the checkpoint’s Gaussian parameters:

| Parameter | Meaning |
|---|---|
| `means` | Gaussian center positions in 3D |
| `scales` | Learned log-scale parameters |
| `quats` | Gaussian orientations |
| `opacities` | Learned opacity logits |
| `features_dc` | Base color coefficients |
| `features_rest` | Additional spherical-harmonic coefficients for view-dependent color |

One inspected training run contained **260,058 Gaussians**. Counts can vary between runs.

## 9. Visualizing Gaussian Centers

We created interactive plots of sampled Gaussian centers.

We experimented with height-based coloring, learned base colors, point filtering, and smaller samples to improve responsiveness.

These plots helped reveal the chair and its surroundings, but they displayed Gaussian centers—not the complete rendered representation.

**No surface mesh was produced.**

## 10. RGB and Depth Inspection

We rendered RGB and depth views to examine the reconstructed scene.

The depth render helped distinguish the chair from surrounding surfaces, while also exposing imperfect background reconstruction.

Depth values were treated as reconstruction units, not calibrated distances in metres.

## 11. Gaussian Size Experiment

We temporarily scaled Gaussian sizes by factors of **0.25, 1, and 2**.

Smaller Gaussians produced more visible gaps and speckling. Larger Gaussians increased overlap and blur.

This demonstrated how Gaussian size affects image coverage and appearance.

## 12. Gaussian Opacity Experiment

We temporarily adjusted Gaussian opacity using factors of **0.1, 1, and 3**, with valid opacity limits.

Lower opacity made the scene more transparent and ghost-like. Higher opacity changed its coverage and appearance.

The original parameters were restored after the experiment.

## 13. Reconstruction Difference Experiment

We compared a reference image with its reconstructed RGB render using a pixelwise difference heatmap.

Differences appeared around edges, fine details, and imperfectly reconstructed regions.

This showed that reconstruction error can resemble scene change even when no physical change has occurred.

## 14. Camera Movement Experiment

We slightly shifted the rendering camera while leaving the Gaussian model unchanged.

The resulting differences demonstrated how viewpoint changes and parallax can generate apparent change.

This motivated using identical cameras for controlled before/after comparisons.

## 15. Selecting a Local 3D Region

We used rendered depth and camera parameters to estimate a 3D location, then selected nearby Gaussian centers inside a small sphere.

We temporarily recolored the selection red to inspect its location beneath the chair seat.

This was a spatial selection, not semantic segmentation of the entire chair.

## 16. Controlled Geometry Edit

We temporarily moved the selected Gaussians sideways while keeping the rendering camera fixed.

The RGB difference map responded mainly around the patch’s original and new locations.

We saved before/after RGB images and depth arrays, then restored the original Gaussian positions.

## 17. Depth Difference and Change Masks

We compared the saved depth renders and created masks using relative-depth thresholds.

| Threshold | Flagged pixels |
|---|---:|
| 0.5% | 486 |
| 1.0% | 400 |
| 2.0% | 306 |

The maximum rendered depth difference was **0.440682 model units**.

This value describes a pixelwise depth difference, including surfaces revealed or covered by the edit. It is not the patch’s displacement or an erosion measurement.

## 18. Multiple Viewpoints

We rendered the same fixed 3D edit from three dataset cameras.

| View | Pixels exceeding the 1% depth threshold |
|---|---:|
| 0 | 400 |
| 5 | 0 |
| 9 | 0 |

This demonstrated that detection depends on visibility and viewpoint.

Zero detected pixels can mean that the patch is hidden, outside the image, or produces an effect below the threshold.

## 19. Color-Only Edit

We changed the selected patch’s color without changing its geometry or opacity.

Results:

- Maximum mean-channel RGB difference: **0.317288**.
- Maximum depth difference: **0**.
- Pixels exceeding the depth-change threshold: **0**.

This separated an appearance edit from a geometry edit in our controlled model.

## 20. Uniform Brightness Change

We darkened the original image by 35%, without changing geometry.

An RGB difference threshold of 0.05 flagged **126,373 of 129,600 pixels—97.5% of the image**.

This demonstrated how lighting alone can dominate an RGB change mask.

## 21. Global Brightness Correction

We estimated a single brightness correction multiplier from aligned image pixels.

| Case | Before correction | After correction |
|---|---:|---:|
| Lighting only | 126,373 | 0 |
| Moved patch + lighting | 126,349 | 159 |

The correction removed uniform darkening while retaining a small response to the edit.

## 22. Shadows and Color Casts

We tested a smooth local shadow, warmer colors, and a shadow combined with the moved patch.

A single global brightness correction left many false alarms:

| Case | Before correction | After correction |
|---|---:|---:|
| Local shadow only | 64,443 | 46,762 |
| Warmer colors only | 24,302 | 24,637 |
| Moved patch + local shadow | 64,395 | 46,667 |

A global multiplier could not adequately correct spatially varying lighting or channel-specific color changes.

## 23. Per-Channel Color Correction

We estimated separate correction gains for red, green, and blue.

| Case | Before correction | After correction |
|---|---:|---:|
| Warmer colors only | 24,302 | 0 |
| Moved patch + warmer colors | 24,498 | 159 |
| Local shadow only | 64,443 | 46,823 |

Channel correction handled the global color cast, but the local shadow remained problematic.

## 24. Local Brightness Correction

We estimated a smooth brightness correction field using Gaussian-blurred brightness images.

| Case | Before correction | After correction |
|---|---:|---:|
| Shadow only | 64,443 | 0 |
| Moved patch + shadow | 64,395 | 150 |

The edit-only RGB reference mask contained **159 pixels**.

After local correction, **142 reference pixels were retained**, approximately **89.3%**, with 8 additional pixels outside that reference mask.

This worked well for the smooth synthetic shadow, while showing that correction can also weaken some edit evidence.

## Main Lessons

- A functioning GPU does not guarantee that CUDA extensions are already compiled.
- Dataset metadata must match the images actually available.
- Gaussian center plots are not meshes or complete Gaussian renders.
- RGB differences respond to geometry, color, lighting, and reconstruction errors.
- Depth comparisons provide complementary evidence, but depend on reconstruction quality and visibility.
- Global brightness and color corrections can handle simple appearance shifts.
- Local correction can reduce smooth shadow effects, but may also suppress genuine appearance changes.
- Pixel counts depend on thresholds and do not represent physical area or volume.

## Limitations and Next Direction

These experiments used controlled edits to one trained reconstruction and synthetic image transformations.

They do not establish performance on real temporal changes, complex weather, independently trained reconstructions, or physical measurements.

The next stage is a structured real before/after evaluation with camera alignment, visibility checks, reference annotations, and uncertainty assessment.

For coastal monitoring, additional validation and metric scale would be needed before reporting erosion or volume change.

## Outputs

The notebook generates training checkpoints, rendered images, an interpolated video, Gaussian-center visualizations, difference maps, binary masks, and experiment comparison figures.

Experiments temporarily modifying model parameters restore the originals afterward and do not overwrite the trained checkpoint.
