FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
ADD ./ /app
RUN pip install /app