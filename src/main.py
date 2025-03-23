from ultralytics import YOLO
from bullmq import Worker, Job
import asyncio
import signal
import os

# Define the prediction function that will be used by BullMQ worker
async def predict(job: Job, token=None):
    try:
        # Extract parameters from job data
        video_path = "media/input.mp4"
        confidence = 0.25
        model_path = 'src/best.pt'
        
        # Validate input
        if not video_path or not os.path.exists(video_path):
            await job.updateProgress(100)  # Mark job as complete even though it failed
            return {"status": "error", "message": f"Video file not found: {video_path}"}

        # Load model
        model = YOLO(model_path)
        
        # Track total batches for progress calculation
        # We need to calculate total frames first
        import cv2
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Store job reference for callback use
        job_ref = job
        
        # Define progress callback
        def on_predict_batch_end(predictor):
            # Get current batch number (add 1 because it's zero-indexed)
            current_batch = predictor.batch_i + 1
            
            # Get total batches
            total_batches = len(predictor.dataset)
            
            # Calculate progress percentage (0-100)
            progress = min(int((current_batch / total_batches) * 100), 100)
            
            # Update BullMQ job progress
            asyncio.create_task(job_ref.updateProgress(progress))
            
            # Print progress for logging
            print(f'Processing batch {current_batch}/{total_batches} - {progress}% complete')
            
            
            # You can add additional processing here if needed

        # Register callback
        model.add_callback("on_predict_batch_end", on_predict_batch_end)
        
        # Initialize progress
        await job.updateProgress(0)
        
        # Run prediction
        results = model.predict(source=video_path, conf=confidence)
        
        # Ensure progress is at 100% when complete
        await job.updateProgress(100)
        
        # Return success result
        return {
            "status": "success",
            "processed_frames": total_frames,
        }
        
    except Exception as e:
        # Update progress to 100% even on error (job is complete, but failed)
        await job.updateProgress(100)
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