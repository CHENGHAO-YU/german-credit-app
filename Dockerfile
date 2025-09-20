FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir streamlit pandas numpy scikit-learn

COPY app/main.py ./

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
