import argparse, os, tempfile

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))
DET_ONNX = os.path.join(MODEL_DIR, "det_v4", "onnx", "model.onnx")
REC_ONNX = os.path.join(MODEL_DIR, "rec_v8", "onnx", "model.onnx")

for p in [DET_ONNX, REC_ONNX]:
    if not os.path.exists(p):
        raise RuntimeError(f"Model not found: {p}")

from infer_onnx import OcrOnnxEngine

app = FastAPI(title="OCR Inference Service", version="1.0.0")
engine = OcrOnnxEngine(lazy=True)


@app.get("/health")
async def health():
    from infer_onnx import check_models
    return {"status": "ok", "engine": "onnx", **check_models()}


@app.post("/infer")
async def infer(image: UploadFile = File(...)):
    if engine is None:
        return JSONResponse({"error": "engine not loaded"}, status_code=503)
    buffer = await image.read()
    if not buffer:
        return JSONResponse({"error": "empty image"}, status_code=400)
    suffix = ".png"
    if image.filename:
        ext = os.path.splitext(image.filename)[1].lower()
        if ext in (".jpg", ".jpeg", ".png", ".bmp"):
            suffix = ext
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(buffer)
        tmp.close()
        results = engine.infer(tmp.name)
    finally:
        os.unlink(tmp.name)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
