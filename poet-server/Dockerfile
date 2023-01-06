# TODO: currently the build context uses the entire POET repo since it installs
# the poet-ai package locally; it should be isolated in the future

FROM python:3.10-slim

# Install server dependencies
WORKDIR /app/poet-server
ADD poet-server/requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Install the poet-ai package
WORKDIR /app
ADD setup.py setup.py
ADD poet poet
RUN pip3 install -e .

WORKDIR /app/poet-server
ADD poet-server/server.py server.py

CMD ["python", "server.py"]
