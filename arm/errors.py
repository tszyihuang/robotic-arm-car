class ArmError(RuntimeError):
    """通信或机械臂运动错误。"""


class ProtocolError(ArmError):
    """串口应答缺失或格式错误。"""


class MotorFault(ArmError):
    """电机报告故障。"""


class MotionTimeout(ArmError):
    """关节未在规定时间内到位。"""
