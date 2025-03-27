# devel needed for bitsandbytes requirement of libcudart.so, otherwise runtime sufficient
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

ENV PATH="/h2ogpt_conda/envs/h2ogpt/bin:${PATH}"
ARG PATH="/h2ogpt_conda/envs/h2ogpt/bin:${PATH}"

ENV HOME=/workspace
ENV CUDA_HOME=/usr/local/cuda-12.1
ENV VLLM_CACHE=/workspace/.vllm_cache
ENV TIKTOKEN_CACHE_DIR=/workspace/tiktoken_cache
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121 https://huggingface.github.io/autogptq-index/whl/cu121"
ENV GGML_CUDA=1
ENV CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=all"
ENV FORCE_CMAKE=1
ENV GPLOK=1
ENV NLTK_DATA=/usr/local/share/nltk_data


# Install APT Dependencies
RUN apt-get update && apt-get install -y \
git \
curl \
wget \
software-properties-common \
pandoc \
vim \
libmagic-dev \
poppler-utils \
tesseract-ocr \
libtesseract-dev \
libreoffice \
autoconf \
libtool \
docker.io \
zip \
unzip \
htop \
tree \
tmux \
jq \
net-tools \
nmap \
ncdu \
mtr \
rsync \
build-essential \
parallel \
bc \
pv \
expect \
cron \
at \
screen \
inotify-tools \
jq \
xmlstarlet \
dos2unix \
libtinfo5 \
ssh \
libmagic-dev poppler-utils tesseract-ocr libtesseract-dev libreoffice \
rubberband-cli \
ffmpeg \
unzip xvfb libxi6 libgconf-2-4 libu2f-udev \
default-jdk \
git-lfs \
libgomp1 \
libomp-dev \
libc6-dev \
ccache


RUN apt-get upgrade -y

WORKDIR /workspace

COPY . /workspace/

RUN touch /workspace/build_info.txt

# Get Natural Language Toolkit Data
RUN bash git-nltk.sh

### Begin of the build script
# Upgrade Chrome
RUN bash chrome-install.sh
# Conda Install
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    echo "Downloaded Miniconda installer" && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    mkdir -p /h2ogpt_conda && \
    ./Miniconda3-latest-Linux-x86_64.sh -b -u -p /h2ogpt_conda && \
    echo "Installed Miniconda to /h2ogpt_conda" && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    ls -la /h2ogpt_conda/bin && \
    /h2ogpt_conda/bin/conda config --set always_yes yes && \
    /h2ogpt_conda/bin/conda update -n base -c defaults conda && \
    /h2ogpt_conda/bin/conda init bash && \
    /h2ogpt_conda/bin/conda create -n h2ogpt python=3.10 -y && \
    /h2ogpt_conda/bin/conda install -n h2ogpt -c conda-forge pygobject weasyprint -y && \
    echo "Conda environment setup complete"


# Linux install script
RUN pip uninstall -y pandoc pypandoc pypandoc-binary flash-attn

# upgrade pip
RUN pip install --upgrade pip wheel

# broad support, but no training-time or data creation dependencies
RUN pip install -r requirements.txt -c reqs_optional/reqs_constraints.txt

# Required for Doc Q/A: LangChain:
RUN pip install -r reqs_optional/requirements_optional_langchain.txt -c reqs_optional/reqs_constraints.txt

# LLaMa/GPT4All
# RUN pip install -r reqs_optional/requirements_optional_llamacpp_gpt4all.txt -c reqs_optional/reqs_constraints.txt --no-cache-dir

# Optional: PyMuPDF/ArXiv:
RUN pip install -r reqs_optional/requirements_optional_langchain.gpllike.txt -c reqs_optional/reqs_constraints.txt

# Optional: FAISS
RUN pip install -r reqs_optional/requirements_optional_gpu_only.txt -c reqs_optional/reqs_constraints.txt
# Optional: Selenium/PlayWright:
RUN pip install -r reqs_optional/requirements_optional_langchain.urls.txt -c reqs_optional/reqs_constraints.txt

# Optional: For DocTR
RUN pip install -r reqs_optional/requirements_optional_doctr.txt -c reqs_optional/reqs_constraints.txt
# For DocTR: go back to older onnx so Tesseract OCR still works
RUN pip install onnxruntime==1.15.0 -c reqs_optional/reqs_constraints.txt
# GPU only:
RUN pip install onnxruntime-gpu==1.15.0 -c reqs_optional/reqs_constraints.txt

# Optional: Playwright
RUN playwright install --with-deps

# Audio speed-up and slowdown (best quality), if not installed can only speed-up with lower quality
RUN pip install pyrubberband==0.3.0 -c reqs_optional/reqs_constraints.txt
RUN pip uninstall -y pysoundfile soundfile

# install TTS separately to avoid conflicts
RUN pip install TTS deepspeed -c reqs_optional/reqs_constraints.txt

# install rest of deps
RUN pip install -r reqs_optional/requirements_optional_audio.txt -c reqs_optional/reqs_constraints.txt

# needed for librosa/soundfile to work, but violates TTS, but that's probably just too strict as we have seen before)
RUN pip install numpy==1.23.0 --no-deps --upgrade -c reqs_optional/reqs_constraints.txt
# TTS or other deps load old librosa, fix:
RUN pip install librosa==0.10.1 --no-deps --upgrade -c reqs_optional/reqs_constraints.txt

# Vision/Image packages
RUN pip install -r reqs_optional/requirements_optional_image.txt -c reqs_optional/reqs_constraints.txt

# In some cases old chroma migration package will install old hnswlib and that may cause issues when making a database, then do:
RUN pip uninstall -y hnswlib chroma-hnswlib
# restore correct version
RUN pip install chroma-hnswlib==0.7.3 --upgrade -c reqs_optional/reqs_constraints.txt

#* GPU Optional: For AutoAWQ support on x86_64 linux
RUN pip uninstall -y autoawq ; pip install autoawq -c reqs_optional/reqs_constraints.txt
# fix version since don't need lm-eval to have its version of 1.5.0
RUN pip install sacrebleu==2.3.1 --upgrade -c reqs_optional/reqs_constraints.txt

# ensure not installed if remade env on top of old env
RUN pip uninstall llama_cpp_python_cuda -y

#* GPU Optional: For exllama support on x86_64 linux
RUN echo "cuda121"
RUN pip install autoawq-kernels -c reqs_optional/reqs_constraints.txt
RUN pip install auto-gptq==0.7.1 exllamav2==0.0.16

#* GPU Optional: Support amazon/MistralLite with flash attention 2
RUN pip install --upgrade pip
RUN pip install flash-attn==2.6.3 --no-build-isolation --no-cache-dir -c reqs_optional/reqs_constraints.txt

# Duckdb used by Chroma < 0.4 uses DuckDB 0.8.1 that has no control over number of threads per database, `import duckdb` leads to all virtual cores as threads and each db consumes another number of threads equal to virtual cores.  To prevent this, one can rebuild duckdb using [this modification](https://github.com/h2oai/duckdb/commit/dcd8c1ffc53dd020623630efb99ba6a3a4cbc5ad) or one can try to use the prebuild wheel for x86_64 built on Ubuntu 20.
RUN pip uninstall -y pyduckdb duckdb
RUN pip install https://h2o-release.s3.amazonaws.com/h2ogpt/duckdb-0.8.2.dev4025%2Bg9698e9e6a8.d20230907-cp310-cp310-linux_x86_64.whl --no-cache-dir --force-reinstall --no-deps -c reqs_optional/reqs_constraints.txt

#* SERP for search:
RUN pip install -r reqs_optional/requirements_optional_agents.txt -c reqs_optional/reqs_constraints.txt
#  For more info see [SERP Docs](README_SerpAPI.md).
RUN pip install aider-chat
# now fix
RUN pip install transformers -U -c reqs_optional/reqs_constraints.txt

RUN pip uninstall flash_attn autoawq autoawq-kernels -y
RUN pip install flash_attn autoawq autoawq-kernels --no-cache-dir -c reqs_optional/reqs_constraints.txt

# work-around issue with tenacity 8.4.0
RUN pip install tenacity==8.3.0 -c reqs_optional/reqs_constraints.txt

# work-around for some package downgrading jinja2 but >3.1.0 needed for transformers
RUN pip install jinja2==3.1.4 -c reqs_optional/reqs_constraints.txt

# Install Unsloth
RUN pip install unsloth -c reqs_optional/reqs_constraints.txt

# Run patches
RUN bash ./docs/run_patches.sh

# Install latest Node.js and npm
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    npm install -n -g npm@latest

# NPM based - with version specification
RUN npm install -g @mermaid-js/mermaid-cli@10.6.1
RUN npm install -g puppeteer-core

# fifty one doesn't install db right for wolfi, so improve
# https://github.com/voxel51/fiftyone/issues/3975
RUN wget https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-ubuntu2204-7.0.4.tgz
RUN tar xvzf mongodb-linux-x86_64-ubuntu2204-7.0.4.tgz
RUN mkdir -p /usr/lib/python3.10/site-packages/fiftyone/db/
RUN cp -r mongodb-linux-x86_64-ubuntu2204-7.0.4/bin /usr/lib/python3.10/site-packages/fiftyone/db/
RUN chmod -R a+rwx /usr/lib/python3.10/site-packages/fiftyone/db

# Remainder of Build
RUN python3.10 tiktoken_cache.py

# Track build info
RUN cp /workspace/build_info.txt /build_info.txt

RUN mkdir -p /workspace/save
RUN chmod -R a+rwx /workspace/save
# Cleanup
RUN rm -rf /workspace/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh
RUN rm -rf /workspace/.cache/pip
RUN rm -rf /h2ogpt_conda/pkgs
RUN rm -rf /workspace/spaces
RUN rm -rf /workspace/benchmarks
RUN rm -rf /workspace/data
RUN rm -rf /workspace/cloud
RUN rm -rf /workspace/docs
RUN rm -rf /workspace/helm
RUN rm -rf /workspace/notebooks
RUN rm -rf /workspace/papers
RUN rm -rf /tmp/*
### End of the build script
RUN chmod -R a+rwx /workspace

ARG user=h2ogpt
ARG group=h2ogpt
ARG uid=1000
ARG gid=1000

RUN groupadd -g ${gid} ${group} && useradd -u ${uid} -g ${group} -s /bin/bash ${user}
# already exists in base image
# RUN groupadd -g ${gid} docker && useradd -u ${uid} -g ${group} -m ${user}

# Add the user to the docker group
RUN usermod -aG docker ${user}

# Switch to the new user
USER ${user}

EXPOSE 8888
EXPOSE 7860
EXPOSE 5000
EXPOSE 5002
EXPOSE 5004

ENTRYPOINT ["python3.10"]
