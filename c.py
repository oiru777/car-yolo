import numpy as np
import cv2
from ultralytics import YOLO
import csv
import datetime
import pytz

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)  # カメラ番号が違う場合は0を調整

if not cap.isOpened():
    print("カメラを開けません")
    exit()

# タイムスタンプをファイル名に入れる
japan = pytz.timezone('Asia/Tokyo')
now = datetime.datetime.now(japan)
timestamp_str = now.strftime('%Y%m%d_%H%M%S')

csv_filename = f'vehicle_log_{timestamp_str}.csv'
video_filename = f'output_{timestamp_str}.mp4'

# 動画保存設定（必要なければコメントアトしてもOK）
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS) or 30  # カメラによっては0が返るのでデフォルト30fps
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

csv_filename = csv_filename
with open(csv_filename, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['timestamp', 'object_id', 'class_name', 'confidence', 'direction', 'travel_time_sec', 'speed_kmh'])

# カウントライン（X座標） — 画面幅に応じて調整してください
line_x1 = int(width * 0.4)  # 例: 画面の40%地点
line_x2 = int(width * 0.65)  # 例: 画面の70%地点

in_count = 0
out_count = 0
tracked_positions = {}
line1_pass_times = {}

frame_skip = 1
frame_count = 0

distance_m = 5.5  # ★実際の距離をここで指定！

while True:
    ret, frame = cap.read()
    if not ret:
        print("フレーム取得失敗")
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    results = model.track(frame, device='mps', persist=True, classes=[2,5,7])
    #results = model.track(frame, device='mps', persist=True, classes=[0,])
    annotated_frame = frame.copy()
    boxes = results[0].boxes

    if boxes.id is not None:
        ids = boxes.id.int().cpu().tolist()
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.int().cpu().tolist()

        for box, tid, conf, cls in zip(xyxy, ids, confs, clss):
            x1, y1, x2, y2 = box
            center_x = x1
            #center_x = (x1 + x2) // 2
            bottom_y = y2
            class_name = model.names[cls]
            confidence = conf

            japan = pytz.timezone('Asia/Tokyo')
            now = datetime.datetime.now(japan)
            timestamp = now.strftime('%Y-%m-%d %H:%M:%S')

            if tid in tracked_positions:
                prev_x = tracked_positions[tid]

                # 線1通過（左→右方向）
                if prev_x < line_x1 and center_x >= line_x1:
                    line1_pass_times[tid] = now
                    in_count += 1

                # 線2通過（左→右方向）
                elif prev_x < line_x2 and center_x >= line_x2:
                    if tid in line1_pass_times:
                        travel_time = (now - line1_pass_times[tid]).total_seconds()
                        # 通過フレーム記録

                        speed_kmh = (distance_m / travel_time) * 3.6 if travel_time > 0 else 0

                        with open(csv_filename, mode='a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([timestamp, tid, class_name, confidence, 'R', f'{travel_time:.4f}', f'{speed_kmh:.4f}'])
                        del line1_pass_times[tid]

                # 逆方向（右→左）通過
                elif prev_x > line_x2 and center_x <= line_x2:
                    line1_pass_times[tid] = now
                    out_count += 1

                elif prev_x > line_x1 and center_x <= line_x1:
                    if tid in line1_pass_times:
                        travel_time = (now - line1_pass_times[tid]).total_seconds()
                        speed_kmh = (distance_m / travel_time) * 3.6 if travel_time > 0 else 0
                        with open(csv_filename, mode='a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([timestamp, tid, class_name, confidence, 'L', f'{travel_time:.4f}', f'{speed_kmh:.4f}'])
                        del line1_pass_times[tid]

            tracked_positions[tid] = center_x

            # バウンディングボックスとラベル表示
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'ID:{tid} {class_name} {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
                    # 下辺の中心を赤丸で表示
            cv2.circle(annotated_frame, (center_x, bottom_y), 4, (0,0,255), -1)
            cv2.putText(annotated_frame, f"cx:{center_x}", (center_x+5, bottom_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    # カウントライン表示
    cv2.line(annotated_frame, (line_x1, 0), (line_x1, height), (0, 0, 255), 2)
    cv2.line(annotated_frame, (line_x2, 0), (line_x2, height), (255, 0, 0), 2)

    # カウントバー表示
    cv2.rectangle(annotated_frame, (0, 0), (width, 40), (0, 0, 0), -1)
    cv2.putText(annotated_frame, f'R: {in_count}  L: {out_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    # ★現在時刻を右上に表示（ミリ秒まで）
    now = datetime.datetime.now(japan)
    time_str = now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # 末尾3桁カット → ミリ秒表示
    text_size, _ = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    text_x = width - text_size[0] - 10
    text_y = 30
    cv2.putText(annotated_frame, time_str, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    out.write(annotated_frame)

    cv2.imshow('Vehicle Tracking', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("終了します")
        break

cap.release()
out.release()
cv2.destroyAllWindows()