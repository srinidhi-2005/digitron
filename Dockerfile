# using lightweight python image
FROM python:3.10-slim

# setting up the working directory
WORKDIR /app

# copy all the files to container
COPY . /app

# installing the dependencies from requirements text file
RUN pip install --no-cache-dir -r requirements.txt

# exposing port number
EXPOSE 8000

# run FastAPI app as
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

