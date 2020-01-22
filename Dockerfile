# Base image.
FROM python:3.7-alpine

# Import developer tools.
# Install Python modules from requirements.txt.
COPY requirements.txt /usr/src/
RUN apk add --virtual .build-deps g++ musl-dev python3-dev blas lapack gfortran \
    && pip install -r /usr/src/requirements.txt

# Load spacy vocabulary.
RUN python -m spacy download en_core_web_lg

# Remove dev tools.
RUN apk del .build-deps

# Copy app files.
COPY indeed_scraper.py /usr/src/
COPY Resume.txt /usr/src/

# Expose gmail port.
EXPOSE 587

# Run the app.
CMD ["python", "usr/src/indeed_scraper.py"]
