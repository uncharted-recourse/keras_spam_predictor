FROM ubuntu:bionic

ARG BRANCH_NAME=__UNSET__
ENV BRANCH_NAME=${BRANCH_NAME}

# RUN mkdir -p /usr/src/app
# WORKDIR /usr/src/app

# ADD . /usr/src/app/

# Primitive system requirements
RUN apt-get -qq update -y
RUN apt-get -qq install -y \
  build-essential \
  python-pip \
  python3.6 \
  python3-pip \
  python3-dev \
  zip \
  unzip \
  && rm -rf /var/lib/apt/lists/*

# Configure python3.6 at `python`
RUN ln -s -f /usr/bin/python3.6 /usr/bin/python \
    && ln -s -f /usr/bin/pip3 /usr/bin/pip

# Upgrade pip and install dumb-init
# RUN pip install --upgrade pip && \
#    pip install dumb-init
RUN pip install dumb-init

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt
RUN cp -r /root/nltk_data /nltk_data
