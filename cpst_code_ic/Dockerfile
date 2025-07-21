# pull python base image
FROM python:3.12-slim

# copy application files
ADD /ic_api /ic_api/

# specify working directory
WORKDIR /ic_api

# update pip
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

# expose port for application
EXPOSE 8001

# start fastapi application
CMD ["python", "app/main.py"]
