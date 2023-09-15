FROM python:3.8-alpine
WORKDIR /src/app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD [ "python", "./src/model/rate_model.py" ]
