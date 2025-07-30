FROM ubuntu:latest
LABEL authors="wfxwy"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY rsu_simulator.py .
CMD ["python", "rsu_simulator.py"]
ENTRYPOINT ["top", "-b"]