from ultralytics import YOLO

# Load model
model = YOLO('best.pt')
test_video_path = "input.mp4"

def on_predict_batch_end(predictor):
    # Get current batch number
    current_batch = predictor.batch_i + 1  # Adding 1 because it's zero-indexed
    
    # Get total batches
    total_batches = len(predictor.dataset)
    
    # Print progress
    print(f'Processing batch {current_batch}/{total_batches}')
    
    # Access the current batch of frames
    frames = predictor.batch[1]  # batch[1] contains the input images
    
    # You can also access the current results
    results = predictor.results
    
    # Do something with the results and frames here
    # For example, you could save them with batch number in the filename

model.add_callback("on_predict_batch_end", on_predict_batch_end)

# Run prediction on video
results = model.predict(source=test_video_path, conf=0.25, save=True)