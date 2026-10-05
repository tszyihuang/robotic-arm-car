# Stamp of the captured frame, not the publication heartbeat.
std_msgs/Header header
uint64 frame_index
string state
string error
float32 inference_ms
# Array of {class_id, name, confidence, box: [x1, y1, x2, y2]} in original image pixels.
string detections_json
