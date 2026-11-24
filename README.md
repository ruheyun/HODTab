
```
conda create -n hodtab python=3.12

conda activate hodtab

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

pip install "git+https://github.com/ruheyun/tabcodec"

pip install tqdm

pip install ucimlrepo
```