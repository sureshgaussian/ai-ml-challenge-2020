#FROM heroku/miniconda
FROM continuumio/miniconda

ADD ./conda_tars.txt /tmp/conda_tars.txt
ADD ./pip_reqs.txt /tmp/pip_reqs.txt

RUN conda create -n nlp --file /tmp/conda_tars.txt
RUN echo "source activate nlp" > ~/.bashrc
ENV PATH /opt/conda/envs/nlp/bin:$PATH
RUN conda --version

#SHELL ["conda", "activate", "nlp", "/bin/bash", "-c"]
#SHELL ["sudo apt install build-essential libpoppler-cpp-dev pkg-config python3-dev
RUN apt-get update
#&& \
#      apt-get -y install sudo

#RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo

#USER docker
#CMD /bin/bash      
RUN apt-get -y  install build-essential libpoppler-cpp-dev pkg-config python3-dev

#RUN conda activate nlp

ADD ./flask_app /opt/flask_app
ADD ./common /opt/common
ADD ./Inference /opt/Inference

WORKDIR /opt/flask_app

COPY run.py /opt/flask_app

CMD ["pwd"]

#ENTRYPOINT ["conda", "run", "-n", "nlp", "python", "run.py"]


RUN pip install gunicorn
RUN pip install -r /tmp/pip_reqs.txt

#ENTRYPOINT ["python"]
RUN python -m nltk.downloader averaged_perceptron_tagger
RUN python -m nltk.downloader wordnet

CMD gunicorn --bind 0.0.0.0:5000 wsgi

