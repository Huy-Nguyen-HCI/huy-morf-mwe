FROM ubuntu:20.04

# install Python 
RUN apt update
RUN apt install -y python3-pip
RUN apt update
RUN apt install -y iputils-ping
RUN apt install -y net-tools
RUN apt install -y curl
RUN apt install -y wget
RUN apt install -y nano
RUN pip install requests==2.28.1 pandas==1.5.1 numpy==1.23.4 scikit-learn==1.1.3

# add scripts
ADD mwe.py mwe.py
ADD workflow workflow

# define entrypoint
ENTRYPOINT ["python3", "mwe.py"]