import cv2

def get_center(box):
    x1, y1, x2, y2 = box
    
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_box_width(box):
    return box[2] - box[0]



def draw_elipse(frame, box, color, trackId=None, label=None):
    x1,y1 = int(box[0]), int(box[1])
    x2,y2 = int(box[2]), int(box[3])
    x_center, y_center = get_center(box)
    width = get_box_width(box)

    cv2.rectangle(frame, (x1,y1),(x2,y2),color=color, thickness=2, lineType=cv2.LINE_4)
    if label is not None:
            cv2.putText(frame, f"{label}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame

def draw_trajectory(frame, centers, color):
    if len(centers) < 2:
        return frame
    for i in range(1, len(centers)):
        cv2.line(frame, centers[i-1], centers[i], color, thickness=2)
    return frame