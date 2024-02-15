import esim_py
import numpy as np
import sys

p= {
    "CONTRAST_THRESH_POS": 0.8,
    "CONTRAST_THRESH_NEG": 0.8,
    "REFRACTORY_PERIOD": 0.0,
    "LOG_EPS": 1e-3,
    "USE_LOG": True,
    "TIMESTAMP_PATH": "timestamps.txt",
    "FRAME_DIR": "rec_frames/",
    "OUT_FILE": "events",
}

p["TIMESTAMP_PATH"] = sys.argv[1]
p["FRAME_DIR"] = sys.argv[2]

if len(sys.argv) == 4:
    p["OUT_FILE"]= sys.argv[3]

# constructor
esim = esim_py.EventSimulator(
    p["CONTRAST_THRESH_POS"],  # contrast thesholds for positive 
    p["CONTRAST_THRESH_NEG"],  # and negative events
    p["REFRACTORY_PERIOD"], # minimum waiting period (in sec) before a pixel can trigger a new event
    p["LOG_EPS"], # epsilon that is used to numerical stability within the logarithm
    p["USE_LOG"],  # wether or not to use log intensity
    )

events_from_images = esim.generateFromFolder(
    p["FRAME_DIR"], # absolute path to folder that stores images in numbered order
    p["TIMESTAMP_PATH"]    # absolute path to timestamps file containing one timestamp (in secs) for each 
)
print(f"writing to {p['OUT_FILE']}")
np.save(p["OUT_FILE"], events_from_images)
