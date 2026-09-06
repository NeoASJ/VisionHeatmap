import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8l.pt")

GRID_ROWS, GRID_COLS = 100, 100
grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)

cap = cv2.VideoCapture("test_3.mp4")
frame_count = 0
PROCESS_EVERY_N = 5
INFER_W, INFER_H = 960, 720

SPLAT_RADIUS = 2
SPLAT_SIGMA = 1.2

def gaussian_splat(grid, row, col, radius=SPLAT_RADIUS, sigma=SPLAT_SIGMA, weight=1.0):
    r_min, r_max = max(0, row - radius), min(grid.shape[0], row + radius + 1)
    c_min, c_max = max(0, col - radius), min(grid.shape[1], col + radius + 1)
    for r in range(r_min, r_max):
        for c in range(c_min, c_max):
            dist_sq = (r - row) ** 2 + (c - col) ** 2
            grid[r, c] += weight * np.exp(-dist_sq / (2 * sigma ** 2))

running_max = 1.0

# --- Output video writer: side-by-side [Live Feed | Heatmap] ---
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_w, out_h = INFER_W * 2, INFER_H   # two frames side by side
writer = cv2.VideoWriter("customer_demo_output.mp4", fourcc, 20.0, (out_w, out_h))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % PROCESS_EVERY_N != 0:
        continue

    frame_infer = cv2.resize(frame, (INFER_W, INFER_H))

    results = model.track(frame_infer, persist=True, classes=[0], conf=0.15, imgsz=1536, verbose=False)

    person_count = 0
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box
            foot_x, foot_y = int((x1 + x2) / 2), int(y2)
            col = np.clip(int(foot_x / INFER_W * GRID_COLS), 0, GRID_COLS - 1)
            row = np.clip(int(foot_y / INFER_H * GRID_ROWS), 0, GRID_ROWS - 1)
            gaussian_splat(grid, row, col)
            person_count += 1
            cv2.circle(frame_infer, (foot_x, foot_y), 5, (0, 255, 0), -1)

    grid_blurred = cv2.GaussianBlur(grid, (0, 0), sigmaX=1.0)
    current_max = grid_blurred.max()
    running_max = max(current_max, 0.98 * running_max + 0.02 * current_max)
    safe_max = max(running_max, 1e-5)
    heat_norm = np.clip((grid_blurred / safe_max) * 255, 0, 255).astype(np.uint8)
    heat_resized = cv2.resize(heat_norm, (INFER_W, INFER_H), interpolation=cv2.INTER_LINEAR)
    heat_color = cv2.applyColorMap(heat_resized, cv2.COLORMAP_JET)

    mask = heat_resized > 10
    overlay = frame_infer.copy()
    overlay[mask] = cv2.addWeighted(frame_infer, 0.5, heat_color, 0.5, 0)[mask]

    cv2.putText(overlay, f"People detected: {person_count}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame_infer, "LIVE FEED", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(overlay, "HEATMAP", (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    combined = np.hstack([frame_infer, overlay])
    writer.write(combined)

    cv2.imshow("Preview (this is just for you while recording)", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
print("Saved: customer_demo_output.mp4")