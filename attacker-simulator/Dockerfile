FROM ubuntu:latest
LABEL authors="wfxwy"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY attacker_simulator.py .
CMD ["python", "attacker_simulator.py"]
ENTRYPOINT ["top", "-b"]