FROM python:3.12-bullseye
RUN mkdir /src
WORKDIR /src
ADD . /src
RUN pip install -r requirements.txt
CMD ["python", "rgg_app_model.py"]
EXPOSE 5000