#!/usr/bin/env python3
"""
CARLA template: constant forward speed + steering from student model.

Students: put your own logic inside predict_steering(img).
The function must return a value between -1 and 1.

Dependencies
------------
Only CARLA is required.  If your model needs numpy, torch, etc.,
import them at the top and add them to requirements as needed.
"""

import carla
import random
import time
import sys
import math
import joblib
import numpy as np
import os
import cv2

# ------------------------ CONFIGURATION --------------------------------------
HOST            = "localhost"
PORT            = 2000
TIMEOUT_S       = 5.0          # seconds

THROTTLE        = 0.5          # constant forward throttle (0..1)
DEFAULT_STEER   = 0.0          # fallback if no camera frame yet
PRINT_EVERY_N   = 30           # console frames between logs
# -----------------------------------------------------------------------------

def extract_features(image):
    h, w, _ = image.shape
    bottom_half = image[h//2:, :]
    hsv = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask, bottom_half, None
    largest = max(contours, key=cv2.contourArea)
    [vx, vy, x, y] = cv2.fitLine(largest, cv2.DIST_L2, 0, 0.01, 0.01)
    angle = -np.degrees(np.arctan2(vx, vy))

    vis_img = bottom_half.copy()
    lefty = int((-x * vy / vx) + y)
    righty = int(((vis_img.shape[1] - x) * vy / vx) + y)
    cv2.line(vis_img, (vis_img.shape[1] - 1, righty), (0, lefty), (0, 0, 255), 3)
    # Draw center vertical blue line
    center_x = vis_img.shape[1] // 2
    cv2.line(vis_img, (center_x, 0), (center_x, vis_img.shape[0]-1), (255, 0, 0), 2)
    # Find green lane center (mean x of green pixels)
    green_coords = np.column_stack(np.where(mask > 0))
    if green_coords.size == 0:
        lane_center = None
    else:
        # green_coords[:,1] is the x axis
        lane_center = int(np.mean(green_coords[:,1]))
    return angle, mask, vis_img, lane_center
# ------------------------------------------------------------------ STUDENTS --
def predict_steering(img):
    """
    Returns random steering for the vehicle should be replaced by the prediciton of the model

    Parameters
    ----------
    img : carla.Image
        The latest RGB camera frame (BGRA byte-buffer).

    Returns
    -------
    float
        Random value in [-1, 1] – the car still drives randomly.
    """
    # -------------- load the model only once ---------------------------
    if not hasattr(predict_steering, "_model"):
        model_path = "svm_model.joblib"
        if not os.path.isfile(model_path):
            print(f"SVM file '{model_path}' not found – "
                    f"only random steering will be used.")
            predict_steering._model = None
        else:
            predict_steering._model = joblib.load(model_path)
            print(f"[INFO] Loaded SVM from '{model_path}'")

    model = predict_steering._model
    if model is not None:
        features = extract_features(img)  # You must define extract_features for your use-case
    try:
        pred = float(model.predict(features)[0])
    except Exception as e:
        pred = 0.0
        print(f" SVM predict failed: {e}")

    if pred > 0.0:
        return 0.1
    elif pred < 0:
        return -1
    else:
        return 0
    
# -----------------------------------------------------------------------------


# ---------------------------- UTILITIES --------------------------------------
def parent_of(actor):
    if hasattr(actor, "get_parent"):
        return actor.get_parent()
    return getattr(actor, "parent", None)

def ang_diff_deg(a, b):
    return (a - b + 180.0) % 360.0 - 180.0

def pick_center_camera(world, vehicle):
    v_yaw = vehicle.get_transform().rotation.yaw
    best = None
    for s in world.get_actors().filter("sensor.camera.rgb"):
        p = parent_of(s)
        if p and p.id == vehicle.id:
            delta = abs(ang_diff_deg(s.get_transform().rotation.yaw, v_yaw))
            if best is None or delta < best[0]:
                best = (delta, s)
    return best[1] if best else None
# -----------------------------------------------------------------------------


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(TIMEOUT_S)
    world  = client.get_world()

    vehicles = world.get_actors().filter("vehicle.*")
    if not vehicles:
        print("No vehicles found. Start a scene first.")
        return
    vehicle = vehicles[0]
    print("Controlling vehicle id=%d type=%s" % (vehicle.id, vehicle.type_id))
    vehicle.set_autopilot(False)

    camera = pick_center_camera(world, vehicle)
    if camera is None:
        print("No center RGB camera attached to the vehicle.")
        return
    print("Using camera id=%d for live feed" % camera.id)

    state = {"frames": 0, "first_ts": None, "latest_img": None}

    def cam_cb(img):
        state["latest_img"] = img
        state["frames"] += 1
        if state["frames"] % PRINT_EVERY_N == 0:
            if state["first_ts"] is None:
                state["first_ts"] = img.timestamp
            elapsed = img.timestamp - state["first_ts"]
            fps = state["frames"] / elapsed if elapsed else 0.0
            print("camera frames: %d   %.1f FPS" % (state["frames"], fps))

    camera.listen(cam_cb)

    try:
        while True:
            img = state["latest_img"]
            if img is not None:
                steer = float(max(-1.0, min(1.0, predict_steering(img))))
            else:
                steer = DEFAULT_STEER  # if no frame yet
            vehicle.apply_control(carla.VehicleControl(throttle=THROTTLE,
                                                       steer=steer))
            time.sleep(0.01)  # ~100 Hz loop

    except KeyboardInterrupt:
        print("\nStopping.")

    finally:
        camera.stop()
        vehicle.apply_control(carla.VehicleControl(brake=1.0))

if __name__ == "__main__":
    try:
        main()
    except RuntimeError as err:
        sys.stderr.write("[ERROR] " + str(err) + "\n"
                         "Is the CARLA server running on this host/port?\n")
