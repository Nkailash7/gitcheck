'''Kailash
created:2/11/23
last modified:2/11/23'''
import cv2
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
import numpy
# Initialize GStreamer run time environment
Gst.init(None)

# Create a GStreamer pipeline with the v4l2src element for the camera
pipeline_str = (
    "v4l2src device=/dev/video1 ! videoconvert ! video/x-raw,format=BGR ! appsink"
)
pipeline = Gst.parse_launch(pipeline_str)

# Create a callback function to process frames
def on_new_sample(appsink):
    sample = appsink.emit("pull-sample")
    buffer = sample.get_buffer()
    caps = sample.get_caps()
    width = caps.get_structure(0).get_value("width")
    height = caps.get_structure(0).get_value("height")

    # Convert the GStreamer buffer to a NumPy array
    data = buffer.extract_dup(0, buffer.get_size())
    image = numpy.frombuffer(data, dtype=numpy.uint8).reshape((height, width, 3))

    # Perform face detection using OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the processed image
    cv2.imshow("Face Detection", image)
    cv2.waitKey(1)

appsink = pipeline.get_by_name("appsink0")
appsink.set_property("emit-signals", True)
appsink.connect("new-sample", on_new_sample)

# Start the GStreamer pipeline
pipeline.set_state(Gst.State.PLAYING)

try:
    # Run the main loop
    while True:
        pass
except KeyboardInterrupt:
    # Cleanup and exit on Ctrl+C
    pipeline.set_state(Gst.State.NULL)
    cv2.destroyAllWindows()
