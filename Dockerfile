# Base image.
FROM ubuntu

RUN apt-get update \
    && apt-get upgrade -y \
    && apt install python3-pip -y \
    && apt install python3.7 -y \
    && python3.7 -m pip install pip

# Copy app files.
COPY requirements.txt /usr/src/
COPY indeed_scraper.py /usr/src/
COPY Resume.txt /usr/src/
COPY .env /usr/src/

# Expose gmail port.
EXPOSE 587

# Import developer tools.
# Install Python modules from requirements.txt.
# Load spacy vocabulary.
RUN apt install g++ musl-dev python3-dev gfortran -y \
    && python3.7 -m pip install -r /usr/src/requirements.txt \
    && python3.7 -m spacy download en_core_web_lg

# Run the app.
ENTRYPOINT ["python3.7", "/usr/src/indeed_scraper.py"]
