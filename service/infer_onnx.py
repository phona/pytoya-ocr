"""
ONNX Runtime 推理引擎：det_v4 + rec_v8。
"""
import logging, os, math

logging.disable(logging.CRITICAL)
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import cv2
import numpy as np
import yaml

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))
DET_ONNX = os.path.join(MODEL_DIR, "det_v4", "onnx", "model.onnx")
REC_ONNX = os.path.join(MODEL_DIR, "rec_v8", "onnx", "model.onnx")
REC_YML  = os.path.join(MODEL_DIR, "rec_v8", "infer", "inference.yml")


class OcrOnnxEngine:
    _instance = None

    def __init__(self, lazy=True):
        if OcrOnnxEngine._instance is not None:
            return
        OcrOnnxEngine._instance = self
        self.det = None
        self.rec = None
        self.char_list = None
        if not lazy:
            self._load()

    def _load(self):
        if self.det is not None:
            return
        import onnxruntime
        opts = onnxruntime.SessionOptions()
        opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.det = onnxruntime.InferenceSession(DET_ONNX, sess_options=opts, providers=["CPUExecutionProvider"])
        self.rec = onnxruntime.InferenceSession(REC_ONNX, sess_options=opts, providers=["CPUExecutionProvider"])
        self._char_list = self._load_char_list()

    def _load_char_list(self) -> list[str]:
        if not os.path.exists(REC_YML):
            return list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        with open(REC_YML) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("PostProcess", {}).get("character_dict", [])

    @staticmethod
    def _det_preprocess(img):
        h, w = img.shape[:2]
        resize_long = 960
        ratio = resize_long / max(h, w)
        if ratio < 1:
            new_h, new_w = int(round(h * ratio)), int(round(w * ratio))
        else:
            new_h, new_w = h, w
        new_h = ((new_h + 31) // 32) * 32
        new_w = ((new_w + 31) // 32) * 32
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        h2, w2 = img.shape[:2]
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = img.transpose((2, 0, 1))[np.newaxis, :, :, :]
        return img, (h2, w2)

    @staticmethod
    def _det_postprocess(prob_map, orig_shape, scaled_shape):
        from shapely.geometry import Polygon

        thresh, box_thresh, max_candidates, unclip_ratio = 0.3, 0.6, 1000, 1.5
        prob_map = prob_map[0, 0, :, :]
        segmentation = (prob_map > thresh).astype(np.uint8)
        n_labels, labels, _, _ = cv2.connectedComponentsWithStats(segmentation, connectivity=4)
        boxes = []
        oh, ow = orig_shape
        for i in range(1, min(n_labels, max_candidates + 1)):
            mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) < 3:
                continue
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) < 4:
                continue
            poly = Polygon(approx.reshape(-1, 2))
            if poly.area < 1:
                continue
            dist = unclip_ratio * np.sqrt(poly.area / (1 + len(approx)))
            expanded = poly.buffer(dist, resolution=2)
            if expanded.is_empty or expanded.geom_type not in ("Polygon", "MultiPolygon"):
                continue
            if expanded.geom_type == "MultiPolygon":
                expanded = max(expanded.geoms, key=lambda p: p.area)
            pts = np.array(expanded.exterior.coords[:-1], dtype=np.float32)
            pts[:, 0] *= ow / scaled_shape[1]
            pts[:, 1] *= oh / scaled_shape[0]
            pts = np.clip(pts, [0, 0], [ow - 1, oh - 1])
            rect = cv2.minAreaRect(pts.astype(np.int32))
            box = cv2.boxPoints(rect)
            box = np.clip(box, [0, 0], [ow - 1, oh - 1]).astype(np.float32)
            boxes.append(box)
        return boxes

    @staticmethod
    def _rec_preprocess(img):
        h, w = img.shape[:2]
        imgH, imgW = 48, 320
        ratio = w / float(h)
        resized_w = imgW if math.ceil(imgH * ratio) > imgW else int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH), interpolation=cv2.INTER_LINEAR)
        resized_image = resized_image.astype(np.float32)
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image = (resized_image - 0.5) / 0.5
        padding_im = np.zeros((3, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im[np.newaxis, :, :, :]

    def _ctc_decode(self, pred: np.ndarray) -> tuple[str, float]:
        pred = pred[0]
        probs = np.max(pred, axis=1)
        pred_idx = np.argmax(pred, axis=1)
        blank = 0
        res, prev = [], -1
        for idx in pred_idx:
            if idx == blank or idx > len(self._char_list):
                prev = -1
                continue
            if idx != prev:
                res.append(self._char_list[idx - 1])
            prev = idx
        text = "".join(res)
        conf = float(np.mean(probs))
        return text, conf

    def infer(self, image_path: str) -> list[dict]:
        self._load()
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            return []
        orig_h, orig_w = img.shape[:2]
        det_input, scaled_shape = self._det_preprocess(img)
        det_output = self.det.run(None, {self.det.get_inputs()[0].name: det_input})
        boxes = self._det_postprocess(det_output[0], (orig_h, orig_w), scaled_shape)
        results = []
        for box in boxes:
            x, y, w, h = cv2.boundingRect(box.astype(np.int32))
            crop = img[y:y+h, x:x+w]
            if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
                continue
            rec_input = self._rec_preprocess(crop)
            rec_output = self.rec.run(None, {self.rec.get_inputs()[0].name: rec_input})
            text, conf = self._ctc_decode(rec_output[0])
            if not text.strip():
                continue
            results.append({
                "text": text,
                "confidence": round(conf, 4),
                "bbox": [int(x), int(y), int(w), int(h)],
            })
        results.sort(key=lambda r: (r["bbox"][1] // 20, r["bbox"][0]))
        return results


def check_models() -> dict:
    return {
        "det_model": os.path.exists(DET_ONNX),
        "rec_model": os.path.exists(REC_ONNX),
    }
