"""ROS header timestamps refer to acquisition rather than publication."""


def set_stamp(header, seconds, frame_id):
    header.stamp.sec, header.stamp.nanosec = divmod(max(0, int(seconds * 1e9)), 1_000_000_000)
    header.frame_id = frame_id


def stamp_seconds(header):
    return header.stamp.sec + header.stamp.nanosec / 1e9
