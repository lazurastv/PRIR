FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt upgrade -y && \
    apt install -y python3 python3-pip libopenmpi-dev openssh-client openssh-server

COPY requirements.txt .
RUN pip3 install --no-cache --upgrade pip
RUN pip3 install --no-cache -r requirements.txt

RUN useradd -ms /bin/bash mpiuser
RUN echo "mpiuser:pass" | chpasswd
USER mpiuser
WORKDIR /home/mpiuser

RUN mkdir ./.ssh
RUN ssh-keygen -q -t rsa -N '' -f ./.ssh/id_rsa
RUN cat ./.ssh/id_rsa.pub > ./.ssh/authorized_keys
COPY --chown=mpiuser:mpiuser ./utils/config ./.ssh/config

RUN mkdir ./app
COPY --chown=mpiuser:mpiuser ./utils/hostfiles ./app/hostfiles
COPY --chown=mpiuser:mpiuser ./data ./app/data
RUN mkdir ./app/results
COPY --chown=mpiuser:mpiuser *.py ./app/

USER root
COPY ./utils/commands.sh ./commands.sh
RUN chmod +x ./commands.sh
CMD ./commands.sh