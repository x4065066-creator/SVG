# Singing-Voice-Guardian: Robust and Imperceptible Singing Voice Protection via Latent Diffusion Models

Singing-Voice-Guardian (SVG) is a framework designed to safeguard singer identity against unauthorized AI-driven Singing Voice Conversion. By leveraging Latent Diffusion Models (LDMs), SVG generates adversarial perturbations within the latent space of a generative model. This ensures that the protected audio remains perceptually indistinguishable from the original to human ears while causing significant performance degradation in unauthorized voice cloning systems.

## Environment Setup

Ensure your environment meets the CUDA requirements for high-performance diffusion sampling.

### Create and activate the environment
```shell
conda env create -f environment.yml -n SVG
conda activate SVG
```

### Optional: Install xformers for optimized memory usage during LDM training
pip install xformers


## Pre-trained Checkpoints

To ensure the pipeline functions correctly, download and place the following checkpoints into their respective directories:

1. -Timbre Encoder: [Speaker-Encoder](https://drive.google.com/drive/folders/15oeBYf6Qn1edONkVLXe82MzdIi3O_9m3)
  - put `best_model.pth.tar`  into `speaker_pretrain/`.

2.- whisper: [`G_0.pth` `D_0.pth`](https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt)
  - put `large-v2.pt` into `whisper_pretrain`.

3.- hubert_soft: [hubert_soft model](https://github.com/bshall/hubert/releases/tag/v0.1)
  - Put `hubert-soft-0d54a1f4.pt` into `hubert_pretrain/`.

4.- pitch extractor: [crepe full](https://github.com/maxrmorrison/torchcrepe/tree/master/torchcrepe/assets)
  - Put `full.pth` into `crepe/assets`.

5.- pretrain model: [sovits5.0.pretrain.pth](https://github.com/PlayVoice/so-vits-svc-5.0/releases/tag/5.0/)
  - Put it into `vits_pretrain/`.

## Dataset Preparation

Place the dataset in the `dataset` directory with the following file strcuture:
```
dataset
├───singer0
│   ├───000.wav
│   ├───...
│   └───xxx.wav
└───singer1
    ├───000.wav
    ├───...
    └───xxx.wav
```

### Audio Pre-processing
For optimal training stability and protection robustness:
 Sample Rate: High-fidelity audio ($\ge 32\text{kHz}$) is recommended.
 Duration: Audio should be sliced into segments of 5s – 15s. Excessive length may trigger Out-of-Memory (OOM) errors.
 Quality: Use clean vocals (dry source) without background instrumentation or heavy reverb.

```shell
cd posterior_encoder
python svc_preprocessing.py -t 2
```
After preprocessing you will get an output with following structure.
```
data_svc/
└── waves-16k
│    └── speaker0
│    │      ├── 000001.wav
│    │      └── 000xxx.wav
│    └── speaker1
│           ├── 000001.wav
│           └── 000xxx.wav
└── waves-32k
│    └── speaker0
│    │      ├── 000001.wav
│    │      └── 000xxx.wav
│    └── speaker1
│           ├── 000001.wav
│           └── 000xxx.wav
└── pitch
│    └── speaker0
│    │      ├── 000001.pit.npy
│    │      └── 000xxx.pit.npy
│    └── speaker1
│           ├── 000001.pit.npy
│           └── 000xxx.pit.npy
└── hubert
│    └── speaker0
│    │      ├── 000001.vec.npy
│    │      └── 000xxx.vec.npy
│    └── speaker1
│           ├── 000001.vec.npy
│           └── 000xxx.vec.npy
└── whisper
│    └── speaker0
│    │      ├── 000001.ppg.npy
│    │      └── 000xxx.ppg.npy
│    └── speaker1
│           ├── 000001.ppg.npy
│           └── 000xxx.ppg.npy
└── speaker
│    └── speaker0
│    │      ├── 000001.spk.npy
│    │      └── 000xxx.spk.npy
│    └── speaker1
│           ├── 000001.spk.npy
│           └── 000xxx.spk.npy
└── singer
│   ├── speaker0.spk.npy
│   └── speaker1.spk.npy
|
└── indexes
    ├── speaker0
    │   ├── some_prefix_hubert.index
    │   └── some_prefix_whisper.index
    └── speaker1
        ├── hubert.index
        └── whisper.index
```


## Training Pipeline

The SVG training process is divided into two distinct phases:

### Phase I: Posterior Encoder & Decoder 

1. Start training
```shell
python svc_trainer.py -c configs/base.yaml -n sovits5.0
```

2. Resume training
```shell
python svc_trainer.py -c configs/base.yaml -n sovits5.0 -p chkpt/sovits5.0/sovits5.0_***.pt
``` 

### Phase II: Latent Diffusion Protection (LDM Training)


CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py --base configs/latent-diffusion/SVG.yaml -t --gpus 0


## Inference & Protection Generation

To protect a specific vocal track, run the inference script. This produces a "protected" version of the audio that resists unauthorized cloning.

```shell
python ./svc-ldm.py \
    --source_feature "./data_audio/singer/sample.spk.npy" \
    --config "./configs/latent-diffusion/SVG.yaml" \
    --ckpt "./logs/svg_final.ckpt" \
    --ddim_steps 100 \
    --chunk_size 128 \
    --outdir "./OUT"
```
Key Parameters:
--ddim_steps: Number of sampling steps. Higher values (50–100) yield better audio quality.

--chunk_size: Latent block size. Reduce this if you encounter VRAM limitations.

## License

This project is intended for academic and ethical research purposes only. Users are strictly prohibited from using this technology to infringe upon intellectual property or create unauthorized deepfakes of public figures.