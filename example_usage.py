"""
Пример за користење на VehicleCounter класата
"""

from vehicle_detection import VehicleCounter
import cv2

def example_basic():
    """Основен пример"""
    print("=== Основен пример ===")
    
    # Креирање на VehicleCounter
    counter = VehicleCounter(
        model_path='yolov8n.pt',
        line_position=0.5,  # Линија на средината
        confidence_threshold=0.5
    )
    
    # Обработка на видеото
    video_path = 'your_video.mp4'  # Заменете со вашата патека
    results = counter.process_video(video_path, show_video=True)
    
    print(f"Резултати: {results}")


def example_with_output():
    """Пример со зачувување на излезното видео"""
    print("=== Пример со зачувување ===")
    
    counter = VehicleCounter(
        model_path='yolov8n.pt',
        line_position=0.6,
        confidence_threshold=0.6
    )
    
    video_path = 'input_video.mp4'
    output_path = 'output_with_detections.mp4'
    
    results = counter.process_video(
        video_path=video_path,
        output_path=output_path,
        show_video=False
    )
    
    print(f"Видеото е зачувано во: {output_path}")
    print(f"Резултати: {results}")


def example_camera():
    """Пример со камера во реално време"""
    print("=== Пример со камера ===")
    
    counter = VehicleCounter(
        model_path='yolov8n.pt',
        line_position=0.5,
        confidence_threshold=0.5
    )
    
    # Користење на камера (0 = default камера)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Не може да се отвори камерата")
        return
    
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_y = int(height * counter.line_position)
    
    print("Притиснете 'q' за излез")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Детекција
            results = counter.model(frame, verbose=False)
            
            # Исцртување
            counter.draw_counting_line(frame)
            counter.update_tracking(results, line_y, height)
            counter.draw_detections(frame, results)
            counter.draw_counters(frame)
            
            cv2.imshow('Vehicle Detection - Camera', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print(f"Резултати: {counter.vehicle_count}")


if __name__ == '__main__':
    # Раскоментирајте го примерот што го сакате:
    
    # example_basic()
    # example_with_output()
    # example_camera()
    
    print("Раскоментирајте го примерот што го сакате во example_usage.py")








