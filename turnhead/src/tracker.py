"""用相邻帧框的重合率分配临时编号，不是身份识别。"""


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.0


class PersonTracker:
    def __init__(self, iou_threshold=.3, max_missing_seconds=.7):
        self.threshold, self.ttl = iou_threshold, max_missing_seconds
        self.tracks, self.next_id = {}, 1

    def update(self, detections, time_seconds):
        self.tracks = {k: v for k, v in self.tracks.items() if time_seconds-v[1] <= self.ttl}
        pairs = sorted(((iou(d['body_box'], b), i, ident)
            for i, d in enumerate(detections) for ident, (b, _) in self.tracks.items()), reverse=True)
        assigned, used = {}, set()
        for score, index, ident in pairs:
            if score >= self.threshold and index not in assigned and ident not in used:
                assigned[index] = ident
                used.add(ident)
        for index, detection in enumerate(detections):
            ident = assigned.get(index)
            if ident is None:
                ident = self.next_id
                self.next_id += 1
            detection['person_id'] = ident
            self.tracks[ident] = (detection['body_box'], time_seconds)
        return detections
