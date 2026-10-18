"""Conservative, untrained court-view selection and short-term association."""
from dataclasses import dataclass

import numpy as np


class TennisSelector:
    """Fit a broadcast court trapezoid from white Hough lines.

    This intentionally rejects uncertain views. It is a heuristic, not a
    replay detector: wide-court replays are visually indistinguishable.
    """
    def __init__(self):
        self.polygon = None
        self.stable = 0

    def reset(self):
        self.polygon = None
        self.stable = 0

    def select(self, image, boxes, tracks=None):
        import cv2
        h, w = image.shape[:2]
        small = cv2.resize(image, (320, 180))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([179, 85, 255]))
        lines = cv2.HoughLinesP(white, 1, np.pi / 180, 35, minLineLength=40, maxLineGap=12)
        horizontal, left, right = [], [], []
        for x1, y1, x2, y2 in ([] if lines is None else np.asarray(lines).reshape(-1, 4)):
            dx, dy = float(x2 - x1), float(y2 - y1)
            if abs(dx) > 45 and abs(dy) < abs(dx) * .12 and 25 < (y1 + y2) / 2 < 175:
                horizontal.append((y1 + y2) / 2)
            if abs(dy) > 30 and abs(dx) > 3:
                slope = dx / dy
                # x = slope*y + intercept
                intercept = x1 - slope * y1
                (left if slope < 0 else right).append((slope, intercept))
        polygon = None
        if len(horizontal) >= 2 and left and right:
            top, bottom = min(horizontal), max(horizontal)
            if bottom - top > 35:
                l = np.median(left, axis=0)
                r = np.median(right, axis=0)
                polygon = np.array([[l[0]*top+l[1], top], [r[0]*top+r[1], top],
                                    [r[0]*bottom+r[1], bottom], [l[0]*bottom+l[1], bottom]], np.float32)
                widths = polygon[[1, 2], 0] - polygon[[0, 3], 0]
                if not (widths[0] > 25 and widths[1] > widths[0] and
                        np.all(polygon[:, 0] > -80) and np.all(polygon[:, 0] < 400)):
                    polygon = None
        if polygon is None:
            self.reset()
            return [], False
        if self.polygon is not None and np.max(np.abs(polygon - self.polygon)) < 18:
            self.stable += 1
        else:
            self.stable = 1
        self.polygon = polygon
        if self.stable < 2:
            return [], False
        # Fit the detected broadcast trapezoid to a canonical court plane.
        target = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
        transform = cv2.getPerspectiveTransform(polygon, target)
        candidates = {0: [], 1: []}
        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            if not (.025 < (y2-y1)/h < .65 and x2 > x1):
                continue
            foot = np.array([[[(x1+x2)/2/w*320, y2/h*180]]], np.float32)
            x, y = cv2.perspectiveTransform(foot, transform)[0, 0]
            if -.2 <= x <= 1.2 and -.4 <= y <= 1.5:
                role = int(y > .5)
                continuity = max((iou(box,t.box) for t in (tracks or {}).values()
                                  if t.role == role),default=0.)
                candidates[role].append((abs(x-.5)-.15*continuity, index, role))
        selected = []
        for group in candidates.values():
            group.sort()
            # Ambiguous similarly placed people are masked, never force top two.
            if group and (len(group) == 1 or group[1][0] - group[0][0] > .12):
                selected.append(group[0][1:])
        return selected, True


@dataclass
class Track:
    identity: int
    box: np.ndarray
    joints: np.ndarray
    valid: np.ndarray
    last_time: float
    velocity: np.ndarray
    role: int


def iou(a, b):
    intersection = np.prod(np.maximum(np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0))
    union = np.prod(np.maximum(a[2:] - a[:2], 0)) + np.prod(np.maximum(b[2:] - b[:2], 0)) - intersection
    return float(intersection / max(union, 1))


class PlayerAssociation:
    def __init__(self, players=2, retention=.5):
        self.players, self.retention = players, retention
        self.next_id = 0
        self.tracks = {}

    def reset(self):
        self.tracks.clear()  # IDs never repeat inside a video, including cuts.

    def update(self, boxes, joints, valid, selected, timestamp, image_size):
        from scipy.optimize import linear_sum_assignment
        self.tracks = {slot: t for slot, t in self.tracks.items()
                       if timestamp - t.last_time <= self.retention}
        slots = sorted(self.tracks)
        assignments, used = [], set()
        if slots and selected:
            costs = np.full((len(slots), len(selected)), 1e6)
            diagonal = max(float(np.linalg.norm(image_size)), 1)
            for row, slot in enumerate(slots):
                t = self.tracks[slot]
                predicted = t.box + t.velocity * (timestamp - t.last_time)
                for col, (index, role) in enumerate(selected):
                    distance = np.linalg.norm((predicted[:2]+predicted[2:]-boxes[index,:2]-boxes[index,2:])/2) / diagonal
                    common = t.valid & valid[index]
                    joint_distance = np.linalg.norm(t.joints[common] - joints[index, common], axis=1).mean()/diagonal if common.any() else distance
                    overlap = iou(predicted, boxes[index])
                    if (t.role == role or role < 0 or t.role < 0) and distance < .12 and (overlap > .05 or distance < .04):
                        costs[row, col] = distance + .1 * (1-overlap) + joint_distance
            rows, cols = linear_sum_assignment(costs)
            for row, col in zip(rows, cols):
                if costs[row, col] < 1e6:
                    assignments.append((slots[row], *selected[col]))
                    used.add(col)
        free = [slot for slot in range(self.players) if slot not in self.tracks]
        for col, (index, role) in enumerate(selected):
            if col not in used and free:
                assignments.append((free.pop(0), index, role))
        result = []
        for slot, index, role in assignments:
            previous = self.tracks.get(slot)
            delta = timestamp - previous.last_time if previous else 0
            velocity = (boxes[index] - previous.box)/delta if delta > 0 else np.zeros(4)
            identity = previous.identity if previous else self.next_id
            if previous is None:
                self.next_id += 1
            self.tracks[slot] = Track(identity, boxes[index].copy(), joints[index].copy(),
                                      valid[index].copy(), timestamp, velocity, role)
            result.append((slot, index, identity, role))
        return result
