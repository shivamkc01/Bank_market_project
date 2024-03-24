FROM python:3.12-slim
WORKDIR .

RUN apt update \
    && apt install -y htop wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  -O miniconda.sh \
    && sh miniconda.sh -b -p ./miniconda3 \
    && rm miniconda.sh

# Add Miniconda to PATH
ENV PATH="./miniconda3/bin:${PATH}"
# RUN conda init 
# RUN conda activate bank_proj

COPY src/requirements.txt .
RUN pip install -r requirements.txt

# COPY /src/train.py .
# CMD ["python", "train.py", "--fold", "10", "--model", "rf"]d