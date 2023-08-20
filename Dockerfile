FROM python:3.10.2

RUN mkdir banodoco

WORKDIR /banodoco

COPY requirements.txt .

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN apt-get update && apt-get install -y ffmpeg
RUN echo "SERVER=production" > .env

COPY . .

EXPOSE 5500

CMD ["sh", "entrypoint.sh"]