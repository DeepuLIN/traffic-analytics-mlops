from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

class Tracker:
    def __init__(self):
        self.tracker = DeepSort(
            max_age=30,
            n_init=2,
            max_cosine_distance=0.3
        )

    def update(self, detections, frame):
        """
        detections: YOLO output
        frame: current frame
        """

        ds_detections = []

        # 🔹 Convert YOLO → DeepSORT format
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]

            w = x2 - x1
            h = y2 - y1

            ds_detections.append(([x1, y1, w, h], conf, "object"))

        # 🔹 Update tracker
        tracks = self.tracker.update_tracks(ds_detections, frame=frame)

        results = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id

            # ✅ FIXED: correct bbox interpretation
            l, t, r, b = track.to_ltrb()

            x1 = int(l)
            y1 = int(t)
            x2 = int(r)
            y2 = int(b)

            results.append({
                "id": track_id,
                "bbox": [x1, y1, x2, y2]
            })

        return results

    def draw_tracks(self, frame, tracks):
        for track in tracks:
            x1, y1, x2, y2 = track["bbox"]
            track_id = track["id"]

            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # ID text
            cv2.putText(
                frame,
                f"ID: {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

        return frame