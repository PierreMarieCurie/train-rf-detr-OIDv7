FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY csv_manifest.txt .

# Use shell to download CSVs listed in the manifest
RUN while IFS=, read -r url filename; do \
      curl -L "$url" -o "$filename"; \
    done < csv_manifest.txt

COPY . .

CMD ["python", "train.py"]
