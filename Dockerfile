From python:3.7
ADD . /usr/flask_app
WORKDIR /usr/flask_app
EXPOSE 3000
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get install -y gnupg
RUN apt-get install -y wget
RUN wget -qO - https://www.mongodb.org/static/pgp/server-4.2.asc | apt-key add -
RUN echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu bionic/mongodb-org/4.2 multiverse" |  tee /etc/apt/sources.list.d/mongodb-org-4.2.list
RUN apt-get update
RUN apt-get install -y mongodb-org
CMD systemctl start mongod
ENV FLASK_APP=app.py

ENTRYPOINT ["flask", "run"]