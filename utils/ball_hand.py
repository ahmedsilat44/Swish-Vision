from drawers.utils import get_center

def valid_point(p):
    return (
        p is not None and
        len(p) == 2 and
        p[0] is not None and
        p[1] is not None
    )

def valid_bbox(b):
    """Return False for None, 0, empty, or invalid bbox."""
    if b == 0:
        return False
    else:
        return True
    
def distance(p1, p2):
    if p1 is None or p2 is None:
        return None
    return ((p1**2 + p2**2)**0.5)

def ball_hand(ball_loco, points, frames):
    leave_frames = []
    in_hand_prev = False
    is_dribble = False
    ball_is_head = False
    prev_ball_valid = False


    dist_thresh=40
    for i, frame_id in enumerate(frames):

        # ---- Safe Ball Center ----
        ball_bbox = ball_loco[i] if i < len(ball_loco) else None

        if valid_bbox(ball_bbox):
            ball_center = get_center(ball_bbox)
        else:
            ball_center = None  # No ball detected
            prev_ball_valid = False
            continue

        # ---- Right wrist ----
        joints = points[i] if i < len(points) else None
        right_wrist = joints[10] if joints is not None else None
        right_soulder = joints[6] if joints is not None else None
        nose = joints[0] if joints is not None else None
        l_eye = joints[1] if joints is not None else None
        r_eye = joints[2] if joints is not None else None
        l_ear = joints[3] if joints is not None else None
        r_ear = joints[4] if joints is not None else None



        # ---- Validity check ----
        if not (valid_point(ball_center) and valid_point(right_wrist)):
            in_hand = False
        else:
            dx = ball_center[0] - right_wrist[0]
            dy = ball_center[1] - right_wrist[1]
            dist = distance(dx, dy)

            if (dist < dist_thresh):
                in_hand = True
            else:
                in_hand = False
       
        if (ball_center[1] > right_soulder[1]): 
            is_dribble = True

        #check if the head is wrongly detected as the ball
        # dist_thresh_head = (ball_bbox[2] - ball_bbox[0])/2 * 1.5
        if not (valid_point(nose) and valid_point(r_eye) and valid_point(l_eye) and
                valid_point(r_ear) and valid_point(l_ear)):
            ball_is_head = False
        else:
            dist_thresh_head_1 = (ball_bbox[2] - ball_bbox[0])/2 * 1.5
            dist_thresh_head = 65

            dx_nose = ball_center[0] - nose[0]
            dy_nose = ball_center[1] - nose[1]

            distance_nose = distance(dx_nose, dy_nose)

            dx_r_eye = ball_center[0] - r_eye[0]
            dy_r_eye = ball_center[1] - r_eye[1]

            distance_r_eye = distance(dx_r_eye, dy_r_eye)

            dx_l_eye = ball_center[0] - l_eye[0]
            dy_l_eye = ball_center[1] - l_eye[1]

            distance_l_eye = distance(dx_l_eye, dy_l_eye)

            dx_r_ear = ball_center[0] - r_ear[0]
            dy_r_ear = ball_center[1] - r_ear[1]

            distance_r_ear = distance(dx_r_ear, dy_r_ear)

            dx_l_ear = ball_center[0] - l_ear[0]
            dy_l_ear = ball_center[1] - l_ear[1]

            distance_l_ear = distance(dx_l_ear, dy_l_ear)



            if (distance_nose <= dist_thresh_head or distance_r_eye <= dist_thresh_head or 
                distance_l_eye <= dist_thresh_head or distance_r_ear <= dist_thresh_head or 
                distance_l_ear <= dist_thresh_head):
                ball_is_head = True


        # ---- Detect transition: ball was in hand → now not ----
        if prev_ball_valid and in_hand_prev  and not in_hand  and not is_dribble  and not ball_is_head :
            leave_frames.append(i)
            print("\n" + "-"*60)
            print(f"[Frame {i}]")
            print(f"ball_bbox       = {ball_bbox}")
            print(f"ball_center     = {ball_center}")
            print(f"right_wrist     = {right_wrist}")
            print(f"right_shoulder  = {right_soulder}")
            print(f"nose            = {nose}")
            print(f"l_eye, r_eye    = {l_eye}, {r_eye}")
            print(f"l_ear, r_ear    = {l_ear}, {r_ear}")
            print()
            print(f"dist_thresh  = {dist_thresh}")
            print(f"dist_thresh_head  = {dist_thresh_head}")
            print(f"dist_thresh_head_1  = {dist_thresh_head_1}")
            print(f"dist(ball, wrist)  = {dist}")
            print(f"dist(ball, nose)   = {distance_nose}")
            print(f"dist(ball, r_eye)  = {distance_r_eye}")
            print(f"dist(ball, l_eye)  = {distance_l_eye}")
            print(f"dist(ball, r_ear)  = {distance_r_ear}")
            print(f"dist(ball, l_ear)  = {distance_l_ear}")
            print()
            print(f"in_hand_prev     = {in_hand_prev}")
            print(f"in_hand          = {in_hand}")
            print(f"is_dribble       = {is_dribble}")
            print(f"ball_is_head     = {ball_is_head}")
            print("will_append_leave_frame = "
                f"{in_hand_prev and (not in_hand) and (not is_dribble) and (not ball_is_head)}")
            print("-"*60)
        prev_ball_valid = True
        in_hand_prev = in_hand
        is_dribble = False
        ball_is_head = False

    return leave_frames


