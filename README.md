# Event-Based Automotive Collision Detection
Event-based looming detection using GeNN.

# Data Formats
Each sequence consists of an event file and a sim_data / metadata file

## Event File
The event file is a .npy file that holds a structured array, with each element corresponding to an event.
The data types of the structured array are

`[("t", "<u4"), ("x", "<u2"), ("y", "<u2"), ("p", "<u2")]`.

- t: event time in milliseconds
- x, y: pixel coordinate
- p: polarity, where 0 is negative event, 1 is positive.

## Metadata File
The metadata is stored in a sim_data.npz file, which consists of

`["coll_type", "t_end", "dt", "vel", "diameter_object"]`.

- coll_type: string describing the type of collision. Event types are: `["pedestrians", "cars", "none", "none_with_traffic", "none_with_crossing"]`.
- t_end: end time of the sequence in milliseconds. If the sequence has a collision, it ends with it, so this also marks the collision time.
- dt: time step in milliseconds.
- vel: the average velocity of the collision object relative to the camera (forward, i.e. z-axis projected) in meters / sec. If no collision occurs it is `np.nan`.
- diameter object: The geometric mean of the collision object bounding box (orthografic projection along the camera z-axis, *not* perspective projection. So, basically a representation of the "cross section" of the collision object in the direction of collision) height and width, averaged over the time of the sequence, in meters. If no collision object occurs, it is `np.nan`.