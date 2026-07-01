import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import argparse
from pathlib import Path
import time


class VehicleCounter:

    def __init__(self, model_path='yolov8n.pt', line_position=0.65, confidence_threshold=0.5):
        """
            Inicijalizacija na VehicleCounter
        
        Args:
            model_path:
            line_position: pozicija na linijata za broenje
            confidence_threshold: minimalen confidence
        """
        self.model = YOLO(model_path)
        self.line_position = line_position
        self.confidence_threshold = confidence_threshold

        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.class_names = {2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}
        
        # Sledenje vozila
        self.tracked_vehicles = {}  # {track_id: {'prev_centroid': (x, y), 'curr_centroid': (x, y), 'class': cls_id, 'crossed': bool}}
        self.counted_ids = set()  # Set of track IDs that have been counted (prevents double counting)
        self.offset = 8
        self.max_history = 15  # Keep more history for better tracking
        self.vehicle_count = {
            'total': 0,
            'car': 0,
            'truck': 0,
            'bus': 0,
            'motorcycle': 0
        }
        
    def _increment_vehicle_type_counter(self, cls_id):
        if cls_id == 2:  # Car
            self.vehicle_count['car'] += 1
        elif cls_id == 3:  # Motorcycle
            self.vehicle_count['motorcycle'] += 1
        elif cls_id == 5:  # Bus
            self.vehicle_count['bus'] += 1
        elif cls_id == 7:  # Truck
            self.vehicle_count['truck'] += 1
    
    def draw_counting_line(self, frame):
        height, width = frame.shape[:2]
        y = int(height * self.line_position)
        cv2.line(frame, (0, y), (width, y), (255, 0, 0), 3)
        return y
    
    def _has_crossed_line_direction(self, prev_centroid, curr_centroid, line_y):
        """
        proverka za dali centroidite preminale na druga lenta
        """
        prev_y = prev_centroid[1]
        curr_y = curr_centroid[1]

        # voziloto ja preminalo linijata od gore nadolu
        crossed_down = prev_y < line_y and curr_y >= line_y
        
        # od dolu nagore
        crossed_up = prev_y > line_y and curr_y <= line_y
        
        return crossed_down or crossed_up
    
    def _has_crossed_in_history(self, positions, line_y):
        """
        Check position history for crossing evidence
        Useful when vehicle crosses between frames or detection is missed
        More lenient: just checks if positions span both sides of the line
        """
        if len(positions) < 2:
            return False

        y_coords = [pos[1] for pos in positions]
        
        # Check if we have positions on both sides of the line
        # More lenient: just check if any position is above and any is below
        min_y = min(y_coords)
        max_y = max(y_coords)
        
        # If the range of Y positions spans the line, vehicle likely crossed
        if min_y < line_y and max_y > line_y:
            return True
        
        return False
    
    def update_tracking(self, detections, line_y, frame_height):
        """
        azuriranje i broenje
        Args:
            detections: YOLO tracking results (so IDs)
            line_y: Y koordinata na linijata za broenje
            frame_height:
        """
        # YOLO vrakjat lista so results, ni trebit prviot
        if len(detections) == 0:
            self._cleanup_old_tracks(line_y)
            return
        
        results = detections[0]
        current_frame_tracks = set()  # ids so se vo toj frame
        
        # Procesiranje na site detektirani vehicles
        if results.boxes is not None and len(results.boxes) > 0:
            has_tracking = results.boxes.id is not None
            
            if not has_tracking:
                print("Warning: Tracking IDs not available. Make sure to use model.track() instead of model()")
            
            for i in range(len(results.boxes)):
                cls_id = int(results.boxes.cls[i].item())
                conf = results.boxes.conf[i].item()
                
                # Check dali e vozilo
                if cls_id not in self.vehicle_classes:
                    continue
                
                # Check na confidence
                if conf < self.confidence_threshold:
                    continue
                
                # Get tracking ID
                if has_tracking:
                    track_id = int(results.boxes.id[i].item())
                else:
                    continue
                
                # centorids
                box = results.boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = box.astype(int)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                curr_centroid = (center_x, center_y)
                
                current_frame_tracks.add(track_id)
                
                # Get or create vehicle track
                if track_id not in self.tracked_vehicles:
                    # novo vozilo - initialize tracking
                    self.tracked_vehicles[track_id] = {
                        'prev_centroid': curr_centroid,
                        'curr_centroid': curr_centroid,
                        'positions': [curr_centroid],  # cuvame history za backup
                        'class': cls_id,
                        'crossed': False,
                        'frames_seen': 1,
                        'first_seen_y': center_y  # kade voziloto bilo prvpat videno
                    }
                else:
                    # Update existing vehicle
                    vehicle = self.tracked_vehicles[track_id]
                    
                    # Update centroids: previous becomes current, new becomes current
                    vehicle['prev_centroid'] = vehicle.get('curr_centroid', curr_centroid)
                    vehicle['curr_centroid'] = curr_centroid
                    
                    # Add to position history
                    vehicle['positions'].append(curr_centroid)
                    if len(vehicle['positions']) > self.max_history:
                        vehicle['positions'] = vehicle['positions'][-self.max_history:]
                    
                    vehicle['frames_seen'] = vehicle.get('frames_seen', 0) + 1
                    
                    # Check if vehicle crossed the line (only if not already counted)
                    if track_id not in self.counted_ids:
                        # Primary method: direction-based crossing detection
                        # Check if centroid moved across the line
                        prev_centroid = vehicle.get('prev_centroid', curr_centroid)
                        
                        # Only check if we have a valid previous position (not the same as current)
                        if prev_centroid != curr_centroid:
                            if self._has_crossed_line_direction(prev_centroid, curr_centroid, line_y):
                                # Vehicle crossed the line - count it once
                                vehicle['crossed'] = True
                                self.counted_ids.add(track_id)
                                self.vehicle_count['total'] += 1
                                self._increment_vehicle_type_counter(cls_id)
                                print(f"Vehicle {track_id} counted! Total: {self.vehicle_count['total']}")  # Debug output
                        
                        # Backup method 1: Check if vehicle appeared on opposite side from where it started
                        # This catches vehicles that were first detected after crossing
                        if track_id not in self.counted_ids and vehicle['frames_seen'] >= 2:
                            first_y = vehicle.get('first_seen_y', center_y)
                            # If vehicle started on one side and is now clearly on the other
                            if first_y < line_y and center_y > line_y + 20:  # Started above, now well below
                                vehicle['crossed'] = True
                                self.counted_ids.add(track_id)
                                self.vehicle_count['total'] += 1
                                self._increment_vehicle_type_counter(cls_id)
                                print(f"Vehicle {track_id} counted (backup 1)! Total: {self.vehicle_count['total']}")  # Debug
                            elif first_y > line_y and center_y < line_y - 20:  # Started below, now well above
                                vehicle['crossed'] = True
                                self.counted_ids.add(track_id)
                                self.vehicle_count['total'] += 1
                                self._increment_vehicle_type_counter(cls_id)
                                print(f"Vehicle {track_id} counted (backup 1)! Total: {self.vehicle_count['total']}")  # Debug
                        
                        # Backup method 2: Check position history for crossing evidence
                        # Useful if crossing happened between frames or detection was missed
                        if track_id not in self.counted_ids and vehicle['frames_seen'] >= 3 and len(vehicle['positions']) >= 3:
                            if self._has_crossed_in_history(vehicle['positions'], line_y):
                                # Vehicle crossed based on history - count it
                                vehicle['crossed'] = True
                                self.counted_ids.add(track_id)
                                self.vehicle_count['total'] += 1
                                self._increment_vehicle_type_counter(cls_id)
                                print(f"Vehicle {track_id} counted (backup 2)! Total: {self.vehicle_count['total']}")  # Debug
        
        # Clean up old tracks (but keep counted_ids to prevent re-counting)
        self._cleanup_old_tracks(line_y, current_frame_tracks)
    
    def _cleanup_old_tracks(self, line_y, active_tracks=None):
        """
        Remove old tracks that are no longer needed
        Important: Keep counted_ids even after removing tracks to prevent re-counting
        """
        if active_tracks is None:
            active_tracks = set()
        
        vehicles_to_remove = []
        for track_id, vehicle in self.tracked_vehicles.items():
            # Keep active tracks (seen in current frame)
            if track_id in active_tracks:
                continue
            
            # For counted vehicles: remove if far from line
            if vehicle.get('crossed', False) or track_id in self.counted_ids:
                curr_y = vehicle.get('curr_centroid', (0, line_y))[1]
                if abs(curr_y - line_y) > 300:  # Far from line
                    vehicles_to_remove.append(track_id)
            else:
                # For uncounted vehicles: be more lenient - keep longer
                # Only remove if very far from line and haven't been seen
                if vehicle.get('positions'):
                    last_y = vehicle['positions'][-1][1]
                    # Keep vehicles near the line area longer
                    if abs(last_y - line_y) > 400:  # Very far from line
                        vehicles_to_remove.append(track_id)
        
        for track_id in vehicles_to_remove:
            if track_id in self.tracked_vehicles:
                del self.tracked_vehicles[track_id]
            # CRITICAL: Keep track_id in counted_ids even after removing track
            # This prevents the same vehicle from being counted again if ID is reused
            # counted_ids persists for the entire video processing session
    
    def draw_detections(self, frame, detections):
        """Iscrtuvanje na pravoagolnici i centroids za broenje na vozilata"""
        # YOLO returns a list with one Results object, get the first one
        if len(detections) == 0:
            return
        
        results = detections[0]
        has_tracking = results.boxes.id is not None
        
        # Iteriranje niz site boxes
        if results.boxes is not None and len(results.boxes) > 0:
            for i in range(len(results.boxes)):
                cls_id = int(results.boxes.cls[i].item())
                conf = results.boxes.conf[i].item()
                
                if cls_id not in self.vehicle_classes:
                    continue
                
                if conf < self.confidence_threshold:
                    continue
                
                box = results.boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = box.astype(int)
                
                # Presmetka na centroid
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                centroid = (center_x, center_y)

                track_id = None
                if has_tracking:
                    track_id = int(results.boxes.id[i].item())
                
                # Box color vo zavisnost dali se counted ili ne
                if track_id and track_id in self.counted_ids:
                    #zeleno za tie so se counted
                    box_color = (0, 255, 0)
                    dot_color = (0, 255, 0)
                else:
                    #blue color za tie so ne se counter
                    box_color = (255, 0, 0)
                    dot_color = (255, 0, 0)
                
                # Iscrtuvanje na pravoagolnik
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                
                #Centroid koj so ni ovozmozvit tocno broenje na site vehicles
                cv2.circle(frame, centroid, 8, dot_color, -1)
                cv2.circle(frame, centroid, 10, (255, 255, 255), 2)

                if cls_id in self.class_names:
                    vehicle_name = self.class_names[cls_id]
                else:
                    vehicle_name = "Vehicle"
                
                if track_id is not None:
                    label = f"ID:{track_id} {vehicle_name}: {conf:.2f}"
                else:
                    label = f"{vehicle_name}: {conf:.2f}"
                
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.1, box_color, 4)


    """ZA COUNTERS (gore desno)"""
    def draw_counters(self, frame):

        y_offset = 40

        frame_height, frame_width = frame.shape[:2]

        font = cv2.FONT_HERSHEY_COMPLEX
        total_font_scale = 1.3
        vehicle_font_scale = 0.9
        total_thickness = 3
        vehicle_thickness = 2
        box_padding = 10
        spacing_between_boxes = 5
        alpha = 0.75

        def draw_text_with_box(text, y_pos, font_scale, text_thickness, bg_color=(60, 60, 60), text_color=(255, 255, 255)):
            text_size_result = cv2.getTextSize(text, font, font_scale, text_thickness)
            if text_size_result is None or len(text_size_result) != 2:
                text_width, text_height = 200, 30
                baseline = 10
            else:
                (text_width, text_height), baseline = text_size_result
                text_width = int(text_width) if text_width is not None else 200
                text_height = int(text_height) if text_height is not None else 30
                baseline = int(baseline) if baseline is not None else 10
            
            # Kalkulacija na box coordinates
            box_x1 = 5
            box_y1 = max(0, int(y_pos - text_height - box_padding))
            box_x2 = min(frame_width, int(box_x1 + text_width + box_padding * 2))
            box_y2 = min(frame_height, int(y_pos + baseline + box_padding))
            
            # Kalkulacija na box height
            box_height = box_y2 - box_y1

            if box_x1 < box_x2 and box_y1 < box_y2:
                overlay = frame.copy()
                cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), bg_color, -1)
                frame[box_y1:box_y2, box_x1:box_x2] = cv2.addWeighted(
                    overlay[box_y1:box_y2, box_x1:box_x2], alpha,
                    frame[box_y1:box_y2, box_x1:box_x2], 1 - alpha, 0
                )

            for dx in [-2, -1, 0, 1, 2]:
                for dy in [-2, -1, 0, 1, 2]:
                    if dx != 0 or dy != 0:
                        cv2.putText(frame, text,
                                   (10 + dx, y_pos + dy), font, font_scale,
                                   (0, 0, 0), text_thickness + 1)

            cv2.putText(frame, text, (10, y_pos), font, font_scale, text_color, text_thickness)
            
            return box_height

        total_text = f"Total: {self.vehicle_count['total']}"
        box_height = draw_text_with_box(total_text, y_offset, total_font_scale, total_thickness, 
                                       bg_color=(40, 40, 40), text_color=(255, 255, 255))
        y_offset += box_height + spacing_between_boxes

        vehicle_texts = [
            f"Car: {self.vehicle_count['car']}",
            f"Truck: {self.vehicle_count['truck']}",
            f"Bus: {self.vehicle_count['bus']}",
            f"Motorcycle: {self.vehicle_count['motorcycle']}"
        ]
        # Syle za counters
        for text in vehicle_texts:
            box_height = draw_text_with_box(text, y_offset, vehicle_font_scale, vehicle_thickness,
                                           bg_color=(50, 50, 50), text_color=(255, 255, 255))
            y_offset += box_height + spacing_between_boxes
    
    def process_video(self, video_path, output_path=None, show_video=True):
        """
        Obrabotka na video file
        
        Args:
            video_path:
            output_path:
            show_video:
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Не може да се отвори видео фајлот: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Видео параметри: {width}x{height}, {fps} FPS, {total_frames} фрејмови")

        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        display_skip = 1

        if show_video:
            cv2.namedWindow('Vehicle Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Vehicle Detection', width, height)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1

                results = self.model.track(
                    frame, 
                    persist=True, 
                    verbose=False,
                    imgsz=320,
                    conf=self.confidence_threshold,
                    device='cpu',
                    max_det=20,
                    agnostic_nms=True,
                    stream=False
                )

                line_y = self.draw_counting_line(frame)

                self.update_tracking(results, line_y, height)

                self.draw_detections(frame, results)

                self.draw_counters(frame)

                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Обработено: {frame_count}/{total_frames} фрејмови ({progress:.1f}%)")

                if out:
                    out.write(frame)
                
                # Prikaz na videoto
                if show_video:
                    if frame_count % display_skip == 0:
                        cv2.imshow('Vehicle Detection', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Прекинато од корисник")
                        break
        
        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
        
        print(f"\n=== Results ===")
        print(f"Total vehicles: {self.vehicle_count['total']}")
        print(f"\nBy vehicle type:")
        print(f"  Cars: {self.vehicle_count['car']}")
        print(f"  Trucks: {self.vehicle_count['truck']}")
        print(f"  Buses: {self.vehicle_count['bus']}")
        print(f"  Motorcycles: {self.vehicle_count['motorcycle']}")
        
        return self.vehicle_count


def main():
    parser = argparse.ArgumentParser(description='Детекција и броење на возила во видео')
    parser.add_argument('--video', type=str, required=True, 
                       help='Патека до видео фајлот')
    parser.add_argument('--output', type=str, default=None,
                       help='Патека за зачувување на излезното видео (опционално)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='Патека до YOLO моделот (default: yolov8n.pt)')
    parser.add_argument('--line', type=float, default=0.65,
                       help='Позиција на линијата за броење (0.0-1.0, default: 0.65)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Минимален confidence за детекција (default: 0.5)')
    parser.add_argument('--no-display', action='store_true',
                       help='Не прикажувај видео во реално време')
    
    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"Грешка: Видео фајлот не постои: {args.video}")
        return

    counter = VehicleCounter(
        model_path=args.model,
        line_position=args.line,
        confidence_threshold=args.confidence
    )
    
    # video obrabotka
    try:
        counter.process_video(
            video_path=args.video,
            output_path=args.output,
            show_video=not args.no_display
        )
    except Exception as e:
        print(f"Грешка при обработка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
