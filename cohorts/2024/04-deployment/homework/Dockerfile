FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

#COPY requirements.txt /app

#RUN pip install --upgrade pip
#RUN pip install -r requirements.txt

RUN pip install pandas
RUN pip install scikit-learn
RUN pip install pyarrow
RUN pip install fastparquet

COPY starter.py /app

CMD [ "python3", "starter.py", "2023", "5" ]