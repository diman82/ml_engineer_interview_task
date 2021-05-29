FROM tiangolo/uwsgi-nginx-flask:python3.8

COPY requirements.txt /
RUN pip3 install -r /requirements.txt

COPY ./main.py /main.py
WORKDIR /

CMD ["python", "main.py"]