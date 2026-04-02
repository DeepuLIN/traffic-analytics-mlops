from fastapi import FastAPI, UploadFile, File

import tempfile
import os
import time
import mlflow

# ============================================================
# 🔥 MLFLOW SETUP (CI-SAFE)
# ============================================================

def setup_mlflow():
    # Disable MLflow in CI
    if os.getenv("CI") == "true":
        print("⚠️ MLflow disabled in CI")
        return False

    try:
        # Use local tracking (no server needed)
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("traffic-analytics")
        print("✅ MLflow initialized")
        return True
    except Exception as e:
        print("⚠️ MLflow setup failed:", e)
        return False


MLFLOW_ENABLED = setup_mlflow()

# ============================================================
# 🚀 FASTAPI APP
# ============================================================

app = FastAPI()

# Lazy globals
detector = None
tracker = None
analytics = None


@app.get("/")
def home():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):

    global detector, tracker, analytics

    print("🔥 ENTERED /analyze")
    start_time = time.time()

    # ============================================================
    # 🔥 Lazy import + initialization
    # ============================================================
    if detector is None:
        from src.detection.yolo_detector import YOLODetector
        from src.tracking.deepsort_tracker import Tracker
        from src.analytics.counter import Analytics

        detector = YOLODetector()
        tracker = Tracker()
        analytics = Analytics(line_position=200)

    # ============================================================
    # 📥 SAVE VIDEO
    # ============================================================
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(await file.read())
    temp_path = temp_file.name
    import cv2
    cap = cv2.VideoCapture(temp_path)

    # Reset analytics
    analytics.__init__(line_position=300)

    frame_count = 0
    stats = {"total": 0, "in": 0, "out": 0}

    # ============================================================
    # 🔥 PROCESS VIDEO
    # ============================================================
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect_frame(frame)
        tracks = tracker.update(detections, frame)
        stats = analytics.update(tracks)

        frame_count += 1

    cap.release()

    processing_time = time.time() - start_time

    # ============================================================
    # 🔥 MLFLOW LOGGING (SAFE)
    # ============================================================
    if MLFLOW_ENABLED:
        try:
            experiment = mlflow.get_experiment_by_name("traffic-analytics")

            with mlflow.start_run(
                experiment_id=experiment.experiment_id,
                run_name=f"run_{file.filename}"
            ) as run:

                print("🔥 MLFLOW STARTED | RUN ID:", run.info.run_id)

                mlflow.set_tag("project", "traffic-analytics")
                mlflow.set_tag("pipeline", "yolo+deepsort")

                mlflow.log_param("video_name", file.filename)
                mlflow.log_metric("total_count", stats.get("total", 0))
                mlflow.log_metric("processing_time_sec", processing_time)

                mlflow.log_dict(stats, "results.json")

                print("🔥 MLFLOW RUN ENDED")

        except Exception as e:
            print("⚠️ MLflow logging failed:", e)

    # ============================================================
    # 🧹 CLEANUP
    # ============================================================
    os.remove(temp_path)

    return {
        "status": "completed",
        "results": stats,
        "processing_time_sec": processing_time
    }