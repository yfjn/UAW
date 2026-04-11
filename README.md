# Universal Adversarial Watermarking for Text Image Protection via Template-based Underpainting

Universal Adversarial Watermarking (UAW) is a unified framework that simultaneously embeds adversarial perturbations and robust watermarks into text images. It provides dual protection by resisting OCR-based text extraction while ensuring reliable traceability through watermark retrieval.



## Project Structure

```
UAW
├── AllConfig/                 # Configuration files and model checkpoints
├── AllData/
├── CSR/
├── EAST/
├── EasyOCR/
├── model_CRAFT/
├── model_DBnet/
├── model_PAN/
├── results/                   # Log files
├── TCM/
├── TextBoxesPP/
├── Tools/
├── UAWUP/                     # Our code
├── UDUP/
├── watermarklab/
├── README.md
└── watermarklab
```



## Environment Setup

Follow the steps below to configure and run the project:

1. **Create and activate a conda virtual environment**:

   ```bash
   conda create -n uaw python=3.8
   conda activate uaw
   ```

2. **Install project dependencies**:
   Use the following command to install the required packages:

   ```bash
   pip install -r requirements.txt --ignore-installed
   ```



## Project Workflow

Please ensure that all file paths are modified as needed before running the steps below.

### Stage 1: Bit Templates Generation

```bash
python UAWUP/train_uawup.py  # One-time offline cost
```

### Stage 2: Watermark Modulation

```bash
python UAWUP/eval_watermark/embed_dir.py
```

### Stage 3: Watermark Extraction

```bash
python UAWUP/eval_watermark/extract_dir.py
```

### Stage 4: Evaluate Adversarial Properties

```bash
python UAWUP/eval_adversarial/eval_dir.py
```


## Experimental Code

### IV.C. Quantitative Comparison in Text Image Protection

Experiment for CSR

```bash
python CSR/csr_watermark_test_batch.py --input-dir results/eval_watermark/ti_a×b/ti --output-dir results/eval_watermark/ti_a×b/ti_csr
```

Experiment for UDUP

```bash
python UAWUP/eval_watermark/embed_dir.py --use_wm False --root_eval results/AllData_results/results_udup/size=30_step=3_eps=120_lambdaw=0.1 --source_dir results/eval_watermark/ti_a×b/ti --save_dir results/eval_watermark/ti_a×b/ti_udup

python UAWUP/eval_adversarial/eval_dir.py --img_dir results/eval_watermark/ti_a×b/ti_udup --gt_img_dir results/eval_watermark/ti_a×b/ti --save_root results/eval_adversarial/ti_a×b/ti_udup
```

Experiment for FAWA

```bash
python UAWUP/fawa_pytorch.py --output_dir results/eval_adversarial/FAWA/attacked --wm_text TDSC

python UAWUP/eval_adversarial/eval_dir.py --img_dir results/eval_watermark/ti_a×b/ti_udup --gt_img_dir results/eval_watermark/ti_a×b/ti --save_root results/eval_adversarial/ti_a×b/ti_udup
```

Experiment for UAW(Ours)
```bash
python UAWUP/eval_watermark/embed_dir.py --use_wm True --patch_dir results/AllData_results/results_uawup_eps100/size=30_step=3_eps=100_lambdaw=0.01_lambday=1e-05 --source_dir results/eval_watermark/ti_a×b/ti --save_dir results/eval_watermark/ti_a×b/awti

python UAWUP/eval_adversarial/eval_dir.py --img_dir results/eval_watermark/ti_a×b/awti --gt_img_dir results/eval_watermark/ti_a×b/ti --save_root results/eval_adversarial/ti_a×b/awti

python UAWUP/eval_watermark/extract_dir.py --patch_dir results/AllData_results/results_uawup_eps100/size=30_step=3_eps=100_lambdaw=0.01_lambday=1e-05 --awti_dir results/eval_watermark/ti_a×b/awti
```

<img src="results\plots\image-20260410164654932.png" alt="image-20260410164654932" style="zoom: 25%;" />

### IV.D. Universality Evaluation of the Proposed Method

Character Size
```bash
python UAWUP/eval_watermark/embed_fontsize.py
python UAWUP/eval_watermark/extract_fontsize.py
python UAWUP/eval_adversarial/eval_character_size.py
```

<img src="results\plots\image-20260410200524208.png" alt="image-20260410200524208" style="zoom: 33%;" />

Character Color

```bash
python UAWUP/eval_watermark/embed_color.py
python UAWUP/eval_watermark/extract_color.py
python UAWUP/eval_adversarial/eval_color.py
```

<img src="results\plots\image-20260410164935522.png" alt="image-20260410164935522" style="zoom: 50%;" />

Language

```bash
python UAWUP/eval_watermark/embed_dir.py --patch_dir results/AllData_results/results_uawup_eps100/size=30_step=3_eps=100_lambdaw=0.01_lambday=1e-05 --source_dir AllData/web_page --save_dir results/eval_watermark/real/web_page

python UAWUP/eval_watermark/extract_dir.py --awti_dir results/eval_watermark/real/web_page --patch_dir results/AllData_results/results_uawup_eps100/size=30_step=3_eps=100_lambdaw=0.01_lambday=1e-05

python UAWUP/eval_adversarial/eval_dir.py --img_dir results/eval_watermark/real/web_page --gt_img_dir AllData/web_page --save_root results/eval_adversarial/real/web_page
```

<img src="results\plots\image-20260410195006043.png" alt="image-20260410195006043" style="zoom:25%;" />

<img src="results\plots\image-20260410213128176.png" alt="image-20260410213128176" style="zoom:25%;" />

Complexity of Background

```bash
python UAWUP/eval_watermark/embed_dir.py --patch_dir results/AllData_results/results_uawup_eps100/size=30_step=3_eps=100_lambdaw=0.01_lambday=1e-05 --patch_iter 27 --source_dir results/eval_watermark/real/bg_overlay --save_dir results/eval_watermark/real/bg_overlay

python UAWUP/eval_watermark/extract_dir.py --awti_dir results/eval_watermark/real/bg_overlay  --patch_dir results/AllData_results/results_uawup_eps100/size=30_step=3_eps=100_lambdaw=0.01_lambday=1e-05

python UAWUP/eval_adversarial/eval_dir.py --img_dir results/eval_watermark/real/bg_overlay --gt_img_dir results/eval_watermark/real/bg_overlay --save_root results/eval_adversarial/real/bg_overlay
```

<img src="results\plots\image-20260410195040183.png" alt="image-20260410195040183" style="zoom:20%;" />

### I.V. Robustness Evaluation of the Proposed Method

Adversarial Robustness & 
Watermarking Robustness
```bash
python UAWUP/eval_watermark/embed_dir.py --patch_dir results/AllData_results/results_uawup_eps100/size=30_step=3_eps=100_lambdaw=0.01_lambday=1e-05 --source_dir results/eval_watermark/ti_a×b/ti --save_dir results/eval_watermark/noise

python UAWUP/test/noise_batch.py
python UAWUP/eval_adversarial/eval_distortion.py
python UAWUP/eval_watermark/extract_noise.py
```

### IV.F. Evaluation on Real-world Applications

Privacy Protection for Sensitive Certificate Images
```bash
python UAWUP/eval_watermark/embed_dir.py --patch_dir results/AllData_results/results_uawup_eps100/size=30_step=3_eps=100_lambdaw=0.01_lambday=1e-05 --source_dir AllData/certificate --save_dir results/eval_watermark/real/certificate

python UAWUP/eval_watermark/extract_dir.py --awti_dir results/eval_watermark/real/certificate  --patch_dir results/AllData_results/results_uawup_eps100/size=30_step=3_eps=100_lambdaw=0.01_lambday=1e-05

python UAWUP/eval_adversarial/eval_dir.py --img_dir results/eval_watermark/real/certificate --gt_img_dir results/eval_watermark/real/certificate --save_root results/eval_adversarial/real/certificate
```

<img src="results\plots\image-20260410195137412.png" alt="image-20260410195137412" style="zoom: 33%;" />

Transferability to Unknown OCR Systems

```bash
python UAWUP/eval_watermark/embed_dir.py --patch_dir results/AllData_results/results_uawup_eps100/size=30_step=3_eps=100_lambdaw=0.01_lambday=1e-05 --source_dir results/eval_watermark/ti_a×b/ti --save_dir results/eval_watermark/ti_a×b/awti

python UAWUP/eval_adversarial/eval_transfer.py
```

<img src="results\plots\image-20260410195241379.png" alt="image-20260410195241379" style="zoom:33%;" />

Robustness Tests under Shooting Scenarios

```bash
python UAWUP/eval_watermark/embed_dir.py --patch_dir results/AllData_results/gray/s30_eps100_step3_w0.01_x0 --patch_iter 41 --source_dir results/eval_watermark/ti_a×b/ti --save_dir results/eval_watermark/gray/awti

python UAWUP/eval_watermark/extract_dir.py  --awti_dir results/eval_watermark/gray/print_shoot/100cm  --patch_dir results/AllData_results/gray/s30_eps100_step3_w0.01_x0 --patch_iter 41

python UAWUP/eval_adversarial/eval_dir.py --img_dir results/eval_watermark/gray/print_shoot/100cm --gt_img_dir results/eval_watermark/ti_a×b/ti --save_root results/eval_adversarial/gray/print_shoot/100cm
```

<img src="results\plots\image-20260410195330018.png" alt="image-20260410195330018" style="zoom:15%;" />

<img src="results\plots\image-20260410195346794.png" alt="image-20260410195346794" style="zoom:25%;" />

<img src="results\plots\image-20260410195407983.png" alt="image-20260410195407983" style="zoom:25%;" />

<img src="results\plots\image-20260410195649894.png" alt="image-20260410195649894" style="zoom: 33%;" />

Time Consumption for Real-time Requirements

```bash
python UAWUP/train_uawup.py  # One-time offline cost
python UAWUP/eval_watermark/embed_dir.py
python UAWUP/eval_watermark/extract_dir.py
```


### IV.G. Discussion on the Hyperparameter Settings

MUI and Size of Bit Template for Adversarial Effectiveness & 
MUI and Size of Bit Template for Watermark Reliability & 
Hyperparameter $\lambda_m$ & 
Ablation Study of Loss Functions

```bash
python UAWUP/train_uawup.py  # One-time offline cost
python UAWUP/eval_watermark/embed_size_it.py
python UAWUP/eval_watermark/extract_size_it.py
python UAWUP/eval_adversarial/eval_size_eps.py
```

Impact of Adversarial and Watermarking Interaction
```bash
python UAWUP/eval_watermark/embed_random.py
python UAWUP/eval_watermark/extract_dir.py --patch_dir results/eval_watermark/size_it/awu --awti_dir results/eval_watermark/size_it/awti_random
python UAWUP/eval_adversarial/eval_random.py
```

Watermark Matrix Size
```bash
python UAWUP/eval_watermark/embed_dir.py --patch_dir results/AllData_results/results_uawup_eps100/size=30_step=3_eps=100_lambdaw=0.01_lambday=1e-05 --source_dir results/eval_watermark/ti_a×b/ti --save_dir results/eval_watermark/ti_a×b/awti_9×9 --wm_h 9 --wm_w 9
python UAWUP/eval_watermark/extract_dir.py --patch_dir results/AllData_results/results_uawup_eps100/size=30_step=3_eps=100_lambdaw=0.01_lambday=1e-05 --awti_dir results/eval_watermark/ti_a×b/awti_9×9
python UAWUP/eval_adversarial/eval_dir.py --img_dir results/eval_watermark/ti_a×b/awti_9×9 --gt_img_dir results/eval_watermark/ti_a×b/ti --save_root results/eval_adversarial/ti_a×b/awti_9×9
```

<img src="results\plots\image-20260410201001387.png" alt="image-20260410201001387" style="zoom: 33%;" />

### References:

1. [UDUP](https://github.com/QRICKDD/UDUP)
2. [FAWA](https://github.com/strongman1995/Fast-Adversarial-Watermark-Attack-on-OCR)
3. [CRAFT](https://github.com/clovaai/CRAFT-pytorch)
4. [DBNet](https://github.com/MhLiao/DB)
5. [EasyOCR](https://github.com/JaidedAI/EasyOCR)
6. [EAST](https://github.com/songdejia/EAST)
7. [TextBoxes++](https://github.com/qjadud1994/Text_Detector)
8. [TCM](https://github.com/wenwenyu/TCM)
9. [watermarklab](https://github.com/chenoly/watermarklab)
