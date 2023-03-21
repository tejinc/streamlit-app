FROM python:3.10-slim

WORKDIR /run

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY ./ ./

EXPOSE 4545

ENTRYPOINT [ "streamlit", "run"]
CMD [ "Info.py" ]
