FROM python:3
WORKDIR /src/app
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . .
CMD [ "python", "./src/model/rate_model.py" ]
