from ultralytics import YOLO
from bullmq import Worker, Job
import asyncio
import signal
import os
import json
import time
import cv2
from pathlib import Path
import numpy as np

# Define the prediction function that will be used by BullMQ worker
async def predict(job: Job, token=None):
    try:
        # Extract parameters from job data
        video_path = "media/input.mp4"
        confidence = 0.25
        model_path = 'src/best.pt'
        
        # Create output directory for temporary JSON files
        job_id = job.id
        output_dir = Path(f"./temp_results/{job_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Final JSON result path
        final_json_path = output_dir / "results.json"
        
        # Validate input
        if not video_path or not os.path.exists(video_path):
            await job.updateProgress(100)  # Mark job as complete even though it failed
            save_error_json(final_json_path, f"Video file not found: {video_path}")
            return {"status": "error", "message": f"Video file not found: {video_path}"}

        # Load YOLO model
        model = YOLO(model_path)
        
        # Calculate total frames for progress tracking
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Store metadata in JSON
        metadata = {
            "video_info": {
                "path": video_path,
                "total_frames": total_frames,
                "fps": fps,
                "width": width,
                "height": height
            },
            "job_info": {
                "id": job_id,
                "model": model_path,
                "confidence": confidence
            },
            "frames": []
        }
        
        # Save initial metadata
        with open(final_json_path, 'w') as f:
            json.dump(metadata, f)
        
        # Function to extract detection data from a frame result
        def extract_frame_data(result, frame_number):
            # Get bounding boxes, classes, and confidence scores
            boxes = result.boxes.cpu().numpy()
            
            detections = []
            if len(boxes) > 0:
                for box in boxes:
                    # Convert box coordinates to JSON-serializable format
                    xyxy = box.xyxy[0].tolist() if hasattr(box, 'xyxy') else []
                    conf = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
                    cls = int(box.cls[0]) if hasattr(box, 'cls') else -1
                    
                    # Get class name if available
                    cls_name = result.names[cls] if hasattr(result, 'names') and cls in result.names else "unknown"
                    
                    detections.append({
                        "bbox": xyxy,  # [x1, y1, x2, y2]
                        "confidence": conf,
                        "class_id": cls,
                        "class_name": cls_name
                    })
            
            return {
                "frame_number": frame_number,
                "timestamp": frame_number / fps if fps > 0 else 0,
                "detections": detections
            }
        
        # Function to update JSON file with new frame data
        def update_json_results(json_path, frame_data):
            try:
                # Load current JSON
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # Append new frame data
                data["frames"].append(frame_data)
                
                # Save updated JSON
                with open(json_path, 'w') as f:
                    json.dump(data, f)
            except Exception as e:
                print(f"Error updating JSON: {e}")
        
        # Function to save error information to JSON
        def save_error_json(json_path, error_message):
            error_data = {
                "status": "error",
                "message": error_message,
                "timestamp": time.time()
            }
            
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, 'w') as f:
                json.dump(error_data, f)
        
        # Initialize progress
        await job.updateProgress(0)
        
        # Run prediction with streaming enabled
        results = model.predict(
            source=video_path, 
            conf=confidence,
            stream=True  # Enable streaming mode
        )
        
        # Process results in streaming mode directly (no callbacks)
        processed_frames = 0
        
        # Iterate through each frame result
        for result in results:
            processed_frames += 1
            
            # Calculate progress percentage (0-100)
            progress = min(int((processed_frames / total_frames) * 100), 100)
            
            # Update BullMQ job progress
            await job.updateProgress(progress)
            
            # Log progress
            if processed_frames % 10 == 0 or progress == 100:
                print(f'Processing frame {processed_frames}/{total_frames} - {progress}% complete')
            
            # Extract detection data for current frame
            frame_data = extract_frame_data(result, processed_frames)
            
            # Append to JSON file
            update_json_results(final_json_path, frame_data)
            
            # Optional: Add a small delay to prevent CPU overuse if needed
            # await asyncio.sleep(0.001)
        
        # Ensure progress is at 100% when complete
        await job.updateProgress(100)
        
        # Return success result with path to the JSON file
        return {
            "status": "success",
            "processed_frames": processed_frames,
            "total_frames": total_frames,
            "results_json": str(final_json_path),
        }
        
    except Exception as e:
        # Update progress to 100% even on error
        await job.updateProgress(100)
        
        # Save error to JSON
        try:
            error_json_path = Path(f"./temp_results/{job.id}/error.json")
            os.makedirs(os.path.dirname(error_json_path), exist_ok=True)
            with open(error_json_path, 'w') as f:
                json.dump({"status": "error", "message": str(e)}, f)
        except:
            pass
            
        return {"status": "error", "message": str(e)}

async def main():
    # Create an event that will be triggered for shutdown
    shutdown_event = asyncio.Event()

    def signal_handler(signal, frame):
        print("Signal received, shutting down.")
        shutdown_event.set()

    # Assign signal handlers to SIGTERM and SIGINT
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Redis connection settings
    redis_connection = {
        "connection": "redis://default:password@/localhost:6379"
        # Replace with your actual Redis credentials
    }

    # Create worker with prediction function
    worker = Worker("videoProcessingQueue", predict, redis_connection)
    print("Worker started and waiting for jobs...")

    # Wait until the shutdown event is set
    await shutdown_event.wait()

    # Close the worker
    print("Cleaning up worker...")
    await worker.close()
    print("Worker shut down successfully.")

if __name__ == "__main__":
    asyncio.run(main())