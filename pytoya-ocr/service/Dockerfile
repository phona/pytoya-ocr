FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0t64 libgomp1 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir onnxruntime fastapi uvicorn python-multipart pyyaml opencv-python-headless shapely

COPY server.py infer_onnx.py ./

COPY models/det_v4/onnx/model.onnx models/det_v4/onnx/model.onnx
COPY models/rec_v8/onnx/model.onnx models/rec_v8/onnx/model.onnx

EXPOSE 8090

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8090"]
