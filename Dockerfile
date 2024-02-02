FROM tensorflow/tensorflow:2.15.0

WORKDIR /ai

COPY ./AI/requirements.txt .

RUN pip install --upgrade pip

RUN apt-get update && apt-get install libgl1 -y

# Install or upgrade 'blinker' inside the virtual environment
RUN pip install blinker

RUN pip install --ignore-installed -r requirements.txt


COPY ./AI .

EXPOSE 5000

CMD [ "python", "Model.py" ]
