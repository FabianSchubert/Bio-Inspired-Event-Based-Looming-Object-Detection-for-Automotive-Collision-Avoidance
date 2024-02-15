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
    "DT_MS": 10,
    "TRIAL_MS": 10000,
    "OUT_DIR": "rec_frames/",
    "OUT_VIDEO": "frames_video.avi",
    "OUT_TIMESTAMPS": "timestamps.txt",
    "IMAGE_FILE": "pattern.png",
    "OFFSET_Y": 0,
    "OFFSET_X": 0,
    "SPACE_X": 0
}


with open(sys.argv[1],"r") as f:
    p0 = json.load(f)

for (name,value) in p0.items():
    p[name] = value

WIDTH = p["INPUT_WIDTH"]
HEIGHT = p["INPUT_HEIGHT"]

V = p["v"]

print(V)

out_json_name= sys.argv[1][:-5]+"_out.json"
jfile= open(out_json_name,'w')
json.dump(p,jfile)

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

img = cv2.imread(p["IMAGE_FILE"])
IMAGE_HEIGHT, IMAGE_WIDTH = img.shape[:2]

RESCALE = HEIGHT/IMAGE_HEIGHT

M = np.float32([[RESCALE,0,0],[0,RESCALE,0]])
img_rescale = cv2.warpAffine(img, M, (int(IMAGE_WIDTH*RESCALE), HEIGHT), borderMode=cv2.BORDER_TRANSPARENT)

t = 0.0

fframe = np.ones((HEIGHT, WIDTH, 3)).astype(np.uint8) * 255

dst = np.zeros((HEIGHT, WIDTH, 3)).astype(np.uint8)

N_REPS = int(2. * p["TRIAL_MS"] * V / (2.*np.pi * (img_rescale.shape[1]+p["SPACE_X"])))+1

if p["OUT_VIDEO"] != "":
    video= True
    vwrite= cv2.VideoWriter(p["OUT_VIDEO"],cv2.VideoWriter_fourcc('M','J','P','G'), 100, (WIDTH, HEIGHT))

# Main loop to generate images
for int_t in tqdm(range(0,p["TRIAL_MS"], p["DT_MS"])):
    t = int_t/1000.0

    shift = p["OFFSET_X"] + (np.cos(2.*np.pi * int_t / p["TRIAL_MS"] - np.pi) + 1.) * p["TRIAL_MS"] * V / (2.*np.pi)

    dst[:,:] = fframe

    for k in range(N_REPS+1):
        #M = np.float32([[1,0,shift+(img_rescale.shape[1]-1)*k],[0,1,0]])
        #cv2.warpAffine(img_rescale, M, (WIDTH, HEIGHT), dst, borderMode=cv2.BORDER_TRANSPARENT)
        M = np.float32([[1,0,shift-(img_rescale.shape[1]+p["SPACE_X"]-1)*k],[0,1,p["OFFSET_Y"]]])
        cv2.warpAffine(img_rescale, M, (WIDTH, HEIGHT), dst, borderMode=cv2.BORDER_TRANSPARENT)
    M = np.float32([[1,0,shift+img_rescale.shape[1]+p["SPACE_X"]-1],[0,1,0]])
    cv2.warpAffine(img_rescale, M, (WIDTH, HEIGHT), dst, borderMode=cv2.BORDER_TRANSPARENT)

    #fframe[:,:] = dst[:HEIGHT,:WIDTH]

    cv2.imshow("frame", dst)
    cv2.waitKey(1)

    fname= "rec"+str(int_t).zfill(5)
    cv2.imwrite(p["OUT_DIR"]+fname+".png", dst)
    timestamp_f.write(f"{t}\n")
    if video:
        vwrite.write(dst)

    '''
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
        '''
timestamp_f.close()
if video:
    vwrite.release()

