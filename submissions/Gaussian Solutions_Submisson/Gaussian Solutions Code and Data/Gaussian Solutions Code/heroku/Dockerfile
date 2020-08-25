FROM heroku/miniconda

ADD ./conda_reqs.txt /tmp/conda_reqs.txt

RUN conda create -n nlp --file conda_reqs.txt

RUN conda activate nlp

ADD ../flask_app /opt/flask_app

WORKDIR /opt/flask_app

RUN pip install gunicorn

CMD gunicorn --bind 0.0.0.0:$PORT wsgi

