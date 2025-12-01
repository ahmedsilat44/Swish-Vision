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

def ball_hand(ball_loco, points, frames):
    leave_frames = []
    in_hand_prev = False


    dist_thresh=40
    for i, frame_id in enumerate(frames):

        # ---- Safe Ball Center ----
        ball_bbox = ball_loco[i] if i < len(ball_loco) else None

        if valid_bbox(ball_bbox):
            ball_center = get_center(ball_bbox)
        else:
            ball_center = None  # No ball detected
            continue

        # ---- Right wrist ----
        joints = points[i] if i < len(points) else None
        right_wrist = joints[10] if joints is not None else None
        right_soulder = joints[6] if joints is not None else None

        # ---- Validity check ----
        if not (valid_point(ball_center) and valid_point(right_wrist)):
            in_hand = False
        else:
            dx = ball_center[0] - right_wrist[0]
            dy = ball_center[1] - right_wrist[1]
            dist = (dx**2 + dy**2)**0.5

            if (dist < dist_thresh):
                in_hand = True
            else:
                in_hand = False
       
        if (ball_center[1] > right_soulder[1]): 
            is_dribble = True

        # ---- Detect transition: ball was in hand → now not ----
        if in_hand_prev and not in_hand and not is_dribble:
            leave_frames.append(i)

        in_hand_prev = in_hand
        is_dribble = False

    return leave_frames
