import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import json
import os
from tqdm import tqdm

"""
Tool for generating images of looming objects that
can then be used with the Prophesee event based
camera simulator "vid2e" to generate input event
streams for the LGMD simulation
"""

def clip(pt):
    npt= [ int(min(max(pt[0], 0), p["INPUT_WIDTH"]-1)),
           int(min(max(pt[1], 0), p["INPUT_HEIGHT"]-1))]
    return np.array(npt)


"""
Main script
"""

if len(sys.argv) != 2:
    print("usage: python generate_looming_images.py <name.json>")
    exit(1)

"""
Standard settings for generating images of looming objects
"""

p = {
    "INPUT_WIDTH": 304,
    "INPUT_HEIGHT": 240,
    "v": 3,      # speed in m/s
    "WD": 1.9,   # width of object in m
    "HT": 1.2,   # height of object in m (if image is used, WD and aspect ratio of the image are used)
    "X0": 0.0,   # x position of bottom left corner
    "Y0": 0.0,   # y position of bottom left corner
    "Z0": -30.0, # z position of bottom left corner
    "FOCAL": -1, # focal length in m
    "DT_MS": 10,
    "TRIAL_MS": 10000,
    "BACKG_COLOR": [ 255, 255, 255],
    "FOREG_COLOR": [ 0, 0, 0 ],
    "BACKGROUND": "random",
    "OUT_DIR": "rec_frames/",
    "OUT_VIDEO": "frames_video.avi",
    "OUT_TIMESTAMPS": "timestamps.txt",
    "IMAGE_FILE": "car_frontal_cut.jpg"
}

with open(sys.argv[1],"r") as f:
    p0 = json.load(f)

for (name,value) in p0.items():
    p[name] = value

out_json_name= sys.argv[1][:-5]+"_out.json"
jfile= open(out_json_name,'w')
json.dump(p,jfile)

if p["IMAGE_FILE"] != "":
    img= cv2.imread(p["IMAGE_FILE"])
    imght= img.shape[0]
    imgwd= img.shape[1]
else:
    imght= int(p["HT"]*100)
    imgwd= int(p["WD"]*100)
    img= np.ones((imght,imgwd,3), dtype= np.uint8)
    img[:,:,:]*= np.array(p["FOREG_COLOR"],dtype= np.uint8)

# transform into image coordinates on "far image"
if imgwd > 0:
    fac= imgwd/p["WD"]    # multiplying with fac turns world meters into far image pixels
else:
    fac= 1
in_wd= p["INPUT_WIDTH"]
in_ht= p["INPUT_HEIGHT"]
fwd= int(p["Z0"]/p["FOCAL"]*in_wd) # far canvas width in pixels
fht= int(p["Z0"]/p["FOCAL"]*in_ht) # far canvas height in pixels
obj_off= np.array([fwd/2-imgwd/2+p["X0"]*fac, fht/2-imght/2+p["Y0"]*fac]).astype(int)

# open file for timestamps
timestamp_f= open(p["OUT_TIMESTAMPS"],"w")

# empty the directory to contain the frames
if os.path.isdir(p["OUT_DIR"]):
    # frame directory exists, empty it
    filelist = os.listdir(p["OUT_DIR"])
    for f in filelist:
        os.remove(os.path.join(p["OUT_DIR"], f))
else:
    # drame directory does not yet exist, create it
    os.mkdir(p["OUT_DIR"])

t= 0.0
if p["v"] > 0:
    z0= p["Z0"]
    z1= 0
else:
    z0= 0
    z1= p["Z0"]

# set the background either to speckled noise or a uniform colour
if p["BACKGROUND"] == "noise":
    fframe= np.random.uniform(70,255,(fht,fwd,3)).astype(np.uint8)
else:
    fframe= np.ones((fht,fwd,3), dtype= np.uint8)
    fframe[:,:,:]*= np.array(p["BACKG_COLOR"],dtype= np.uint8)

fframe[obj_off[1]:obj_off[1]+imght, obj_off[0]:obj_off[0]+imgwd,:]=img[:,:,:]

if p["OUT_VIDEO"] != "":
    video= True
    vwrite= cv2.VideoWriter(p["OUT_VIDEO"],cv2.VideoWriter_fourcc('M','J','P','G'), 100, (p["INPUT_WIDTH"],p["INPUT_HEIGHT"]))

# Main loop to generate images
for int_t in tqdm(range(0,p["TRIAL_MS"],p["DT_MS"])):
    t= int_t/1000.0
    z= z0+p["v"]*t
    if z < 0:
        dfac= z/p["FOCAL"]
        dwd= int(in_wd*dfac)
        dht= int(in_ht*dfac)
        p1= np.array([ fwd/2-dwd/2, fht/2-dht/2]).astype(int)
        # rescale the appropriate part of the "far screen" to generate the correct image frame
        frame= cv2.resize(fframe[p1[1]:p1[1]+dht, p1[0]:p1[0]+dwd,:], (in_wd, in_ht),
                          interpolation=cv2.INTER_CUBIC)
        cv2.imshow("frame",frame)
        cv2.waitKey(1)
        fname= "rec"+str(int_t).zfill(5)
        cv2.imwrite(p["OUT_DIR"]+fname+".png",frame)
        timestamp_f.write(f"{t}\n")
        if video:
            vwrite.write(frame)

timestamp_f.close()
if video:
    vwrite.release()

