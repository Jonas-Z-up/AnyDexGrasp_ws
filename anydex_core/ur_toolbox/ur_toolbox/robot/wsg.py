import socket
from time import time, sleep
from enum import Enum
import atexit

class WSG:
    def __init__(self, TCP_IP = "192.168.2.20", TCP_PORT = 1000):
        self.TCP_IP = TCP_IP
        self.TCP_PORT = TCP_PORT 
        self.BUFFER_SIZE = 1024
        self.tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_sock.connect((TCP_IP, TCP_PORT))
        self.timeout = 10
        atexit.register(self.__del__)
        # Acknowledge fast stop from failure if any
        self.ack_fast_stop()
        self.home()

    def wait_for_msg(self, msg):
        since = time()
        while True:
            data = self.tcp_sock.recv(self.BUFFER_SIZE)
            if data == msg:
                ret = True
                break
            elif data.decode("utf-8").startswith("ERR"):
                ret = False
                print(f"[WSG] Error: {data}")
                break
            if time() - since >= self.timeout:
                print(f"[WSG] Timeout ({self.timeout} s) occurred.")
                break
            sleep(0.1)
        return ret

    def ack_fast_stop(self):
        MESSAGE = str.encode("FSACK()\n")
        self.tcp_sock.send(MESSAGE)
        return self.wait_for_msg(b"ACK FSACK\n")

    def set_verbose(self, verbose=True):
        """
        Set verbose True for detailed error messages
        """
        MESSAGE = str.encode(f"VERBOSE={1 if verbose else 0}\n")
        self.tcp_sock.send(MESSAGE)
        return self.wait_for_msg(MESSAGE)

    def home(self):
        """
        Fully open the gripper
        """
        MESSAGE = str.encode("HOME()\n")
        self.tcp_sock.send(MESSAGE)
        return self.wait_for_msg(b"FIN HOME\n")

    def gripper_action(self, position, speed=55, force=0):
        """
        Move fingers to specific position
        * position 0 :- fully close
        * position 110 :- fully open
        """
        MESSAGE = str.encode(f"MOVE({position}, {speed})\n")
        self.tcp_sock.send(MESSAGE)
        return self.wait_for_msg(b"FIN MOVE\n")

    def close_gripper(self, force=60, width=5, speed=50, sleep_time = 0):
        MESSAGE = str.encode(f"GRIP({force}, {width}, {speed})\n")
        self.tcp_sock.send(MESSAGE)
        return self.wait_for_msg(b"FIN GRIP\n")

    def open_gripper(self, position=110, speed=300, force=0, sleep_time = 0):
            """
            Move fingers to specific position
            * position 0 :- fully close
            * position 110 :- fully open
            """
            MESSAGE = str.encode(f"MOVE({position}, {speed})\n")
            self.tcp_sock.send(MESSAGE)
            return self.wait_for_msg(b"FIN MOVE\n")

    def get_gripper_position(self, distance):
        if distance > 110:
            distance = 110
        elif distance < 0:
            distance = 0
        return distance

    def bye(self):
        MESSAGE = str.encode("BYE()\n")
        self.tcp_sock.send(MESSAGE)
        return

    def __del__(self):
        self.bye()
