import cv2

class Analytics:
    def __init__(self, line_position=300):
        """
        line_position: y-coordinate of horizontal line
        """
        self.line_y = line_position

        self.counted_ids = set()      # to avoid double counting
        self.total_count = 0

        self.in_count = 0
        self.out_count = 0

        self.prev_positions = {}      # track previous positions

    def _get_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        return cx, cy

    def update(self, tracks):
        """
        tracks = [
            { "id": int, "bbox": [x1,y1,x2,y2] }
        ]
        """

        for track in tracks:
            track_id = track["id"]
            bbox = track["bbox"]

            cx, cy = self._get_centroid(bbox)

            # 🔹 Get previous position
            prev_cy = self.prev_positions.get(track_id, None)

            # 🔹 Store current position
            self.prev_positions[track_id] = cy

            if prev_cy is None:
                continue

            # 🔥 LINE CROSSING LOGIC
            # crossing downward
            if prev_cy < self.line_y and cy >= self.line_y:
                if track_id not in self.counted_ids:
                    self.counted_ids.add(track_id)
                    self.total_count += 1
                    self.in_count += 1

            # crossing upward
            elif prev_cy > self.line_y and cy <= self.line_y:
                if track_id not in self.counted_ids:
                    self.counted_ids.add(track_id)
                    self.total_count += 1
                    self.out_count += 1

        return {
            "total": self.total_count,
            "in": self.in_count,
            "out": self.out_count
        }

    def draw_analytics(self, frame):
        h, w, _ = frame.shape

        # 🔹 Draw line
        cv2.line(frame, (0, self.line_y), (w, self.line_y), (0, 255, 255), 2)

        # 🔹 Draw counts
        cv2.putText(frame, f"Total: {self.total_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.putText(frame, f"In: {self.in_count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

        cv2.putText(frame, f"Out: {self.out_count}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        return frame