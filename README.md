<img src="logo.png" alt="drawing"/>

**Yes, it's an AI!**

A tool that automatically detects and counts colonies on images of agar plates.

Think automated colony counting isn't for you 🤨? Your plates are too "wild" for a machine?
Take a photo with your smartphone and let agar3000 prove you wrong.


- No fine-tuning  or supervision required. Just an input path with your images and an output path for the results
- It can analyse up to 20 images per minute in CPU-only mode and up to 65 with GPU acceleration.


## Requirements
<details>
  <summary markdown="span">Click to expand</summary>

### Images
1. Photos may be captured using any camera, including **a smartphone camera**.
2. Images should be taken against a uniform background and under adequate lighting. The plate must be in sharp focus, and occupy the majority of the image (at least two-thirds of the shorter side of the image) and be in sharp focus.
3. Condensation, glare, and other imperfections on the plate's lid may lead to inaccurate results. The tool has some tolerance for bubbles in the media.
4. The tool may have difficulty detecting very small colonies, colonies with complex structures, or colonies grown on unusually looking media.
5. The recommended minimum image resolution is 2048 × 2048 pixels. Higher resolutions do not affect the results, as images are downscaled internally.
6. Supported image formats are JPG, PNG, TIFF, BMP (any formats supported by **opencv**)

### Hardware
agar3000 runs on any x86-based system with **Windows or Linux** and uses up to 1G of RAM. It works without a GPU but can also utilise a GPU to significantly accelerate computations.

- **NVIDIA GPUs** are supported via CUDA (with cuDNN), starting from the Maxwell architecture onwards (e.g., GTX 780 Ti, GTX 900 series and above), on both Linux and Windows systems.
- **AMD GPUs** are supported via ROCm, starting from the Vega architecture onwards (e.g., RX Vega, RX 5000 series and newer), on Linux only.

<img src="speed.png"/>
</details>




## Getting started
agar3000 requires **python** (>v3.6) with **opencv** and **onnxruntime** packages. We recommend to use environment management system such as conda for installation: https://www.anaconda.com/download/

### CPU-mode:
The CPU-mode is the simplest way to install, test and use agar3000, provided that slower inference is acceptable.

```bash
pip install opencv-python onnxruntime
```

### GPU-mode:

To set up a GPU-mode can be tricky and we don't recommend unless the inference speed is necessary.

GPU inference requires matching of GPU, [CUDA+cuDNN](https://developer.nvidia.com/cuda/) (for Nvidia GPUs) or [ROCm](https://www.amd.com/en/products/software/rocm.html) (for AMD GPUs), OS, python and onnxruntime versions. 

For Nvidia GPU inference with CUDA 12.x + cuDNN 9.x, try:

    pip install opencv-python onnxruntime-gpu 

For AMD GPU inference with ROCm 7.0, try:

    pip install opencv-python onnxruntime-rocm 

Take a note, that _onnxruntime-gpu_ and _onnxruntime-rocm_ will fall back to CPU-mode if GPU initialisation failed. 

For compatibility with other versions of CUDA and ROCm see corresponding tables:

[CUDA](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)

[ROCm](https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html#requirements)

## Workflow

    python agar3000.py input_path output_path [-t] [-b] [-h] [--no-crop] [--extra] 

#### Main arguments:

* `input_path`: Path to the folder with images of plates. Non-recursive: files in nested subdirectories are not processed. Input_path can be a single image file.
* `output_path`: Path to the folder where results will be saved. It will be created automatically if not exist.

#### Optinal flags:

* `-t`: Trans-illumination mode. Implements a different model trained for detecting colonies on trans-illuminated plates (see image3 and 4 in examples below)
* `-b`: Improves inference speed with marginal change in counting precision. 
* `--no-crop`: Skips cropping of images. Useful if cropping fails or plates are not circular
* `--extra`: Saves representation of intermediate stages of image processing as images and tables: cropped images, tiles before deduplication, tiles after deduplication. Significantly slowdown inference. Recommended only for testing

#### Results include:
- images of each plate with drown boxes
- csvs with annotation for each plate
- RESULTS.csv with the number of colonies for each file/plate
- agar3000_[timestamp].log
  

For the first run we recomend to test the tool on the single image. Or you can try our demo:

    python agar3000.py demo demo/results
    
for trans-illuminated plates:

    python agar3000.py demo_t demo_t/results


<img src="samples.png"/>

## Pipeline (TBD)

## QnA

*Which side of the plate has to be photographed?*

The lid side. That's how the model was trained and tested.

*There is condensate under the lid. What do I have to do?*

Open the lid. The condensate may partially or completely  obscure the surface of agar.  Unfortunately we haven't found better solution then simply open the lid. Use laminar flow if required.

*Which illumination of the plate is an adequate one?*

The light has to be homogenous to prevent reflection artefacts. We used a LED-panel with diffuser.

*What is trans-illumination and is it better?* 

Trans-illumination is simply illumination of a transparent plate from behind,  


*How small has to be a colony to be too small for detection?*

The size of the colony has to at least 8px in diameter, but the real performance may depend on multiple factors. So we recommend to check results if there is some suspictions


*Which media considered as unusually looking media*
agar3000 was tested on different media and we noticed drop of precision with some plates on media with not trivial colours: blood agar and chocolate agar. The good performance on such media may depend on colonies morphology and not guaranteed  

*What is the optimal number of colonies per plate?*
agar3000 was tested on plates with up to 300 colonies. However the maximum theoretical limitation is 1600 colonies per plate. We recommend to use plates with 10-300 colonies for good precision. 

*Is hyper-trading beneficial for inference speed?*

No. In our tests it actually  significantly slowed down inference.






## License

- **Code:** MIT (see LICENSE file)
- **Model weights:** CC BY-NC 4.0 — non-commercial use only

## References
This project was trained using the AGAR (Annotated Germs for Automated Recognition) dataset introduced in:

Majchrowska, S., Pawłowski, J., Guła, G., Bonus, T., Hanas, A., Loch, A., Pawlak, A., Roszkowiak, J., Golan, T., and Drulis-Kawa, Z.
“AGAR: A Microbial Colony Dataset for Deep Learning Detection” (2021).

--CHANGE--

Dataset source: https://agar.neurosys.com/

We gratefully acknowledge the authors and contributors for making this dataset publicly available.
 
Special thanks to @dedovskaya for sharing code and model that was used during the early development stage: https://github.com/dedovskaya/CFUCounter

If agar3000  was usefull for you, don't forget to recommend it your collegues and to mention it in your papers. Thanks for testing!
