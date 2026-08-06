import serial
import binascii
import time
import struct

class Robotiq():
    def __init__(self, port = '/dev/ttyUSB0'):
        
        self.port = port
        self.ser = serial.Serial(port=self.port,baudrate=115200,timeout=1, parity=serial.PARITY_NONE,stopbits=serial.STOPBITS_ONE,bytesize=serial.EIGHTBITS)
        self._max_distance = 90
        self._min_distance = -18
        self.ser.write(b"\x09\x10\x03\xE8\x00\x03\x06\x00\x00\x00\x00\x00\x00\x73\x30")
        self.activate()
        self.close_gripper()
        self.open_gripper()
        print('-----------Gripper Activated--------------')
    
    def activate(self):
        self.ser.write(b"\x09\x10\x03\xE8\x00\x03\x06\x01\x00\x00\x00\x00\x00\x72\xE1")
        # self.ser.write(b"\x09\x10\x03\xE8\x00\x03\x06\x00\x00\x00\x00\x00\x00\x73\x30")
        data_raw = self.ser.readline()
        # print(data_raw)
        data = binascii.hexlify(data_raw)
        print("Response 1 ", data)
        time.sleep(0.01)
        
        self.ser.write(b"\x09\x03\x07\xD0\x00\x01\x85\xCF")
        data_raw = self.ser.readline()
        # print(data_raw)
        data = binascii.hexlify(data_raw)
        print("Response 2 ", data)
        time.sleep(0.2)

    def open_gripper(self, sleep_time = 0.5):
        self.ser.write(b"\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00\x00\xFF\xFF\x72\x19")
        # data_raw = self.ser.readline()
        # print(data_raw)
        # data = binascii.hexlify(data_raw)
        # print("Response 3 ", data)
        time.sleep(sleep_time)

    def close_gripper(self, sleep_time = 0.5):
        # print("Close gripper")
        t1 = time.time()
        self.ser.write(b"\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00\xFF\xFF\x64\x03\x82")
        # self.gripper_action(position=255, speed=255, force=120)
        t2 = time.time()
        # data_raw = self.ser.readline()
        t3 = time.time()
        # print(data_raw)
        # data = binascii.hexlify(data_raw)
        t4 = time.time()
        # print("Response 3 ", data)
        time.sleep(sleep_time)
        t5 = time.time()
        print(f'write time:{t2 -t1}, read time:{t3 -t2}, convert time:{t4 - t3}, sleep time:{t5 - t4}')


    def gripper_action(self, position, speed, force):
        # position: 0x00...minimum, 0xff...maximum
        # speed: 0x00...minimum, 0xff...maximum
        # force: 0x00...minimum, 0xff...maximum
        #print('move hand')
        command = bytearray(b'\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00\x00\x00\x00')
        command[10] = position
        command[11] = speed
        command[12] = force
        self.send_command(command)

    def get_gripper_position(self, distance):
        if distance > self._max_distance:
            distance = self._max_distance
        elif distance < self._min_distance:
            distance = self._min_distance
        position = int( 255 * (self._max_distance - distance) / (self._max_distance - self._min_distance) )
        #print 'max=%d, min=%d, pos=%d pos_mm=%.1f' % (self._max_position, self._min_position, position, distance)
        return position

    def adjust(self):
        self.gripper_action(255, 100, 1)
        (status, position, force) = self.wait_move_complete()
        self._max_distance = self.get_distance(position)
        self.gripper_action(0, 100, 11)
        (status, position, force) = self.wait_move_complete()
        self._min_distance = self.get_distance(position)

    def get_distance(self, position):
        if position > self._max_position:
            position = self._max_position
        elif position < self._min_position:
            position = self._min_position
        distance = 90.0 * (self._max_position - position) / (self._max_position - self._min_position)
        #print 'max=%d, min=%d, pos=%d pos_mm=%.1f' % (self._max_position, self._min_position, position, distance)
        return distance

    def send_command(self, command):
        crc = self._calc_crc(command)
        data = command + crc
        self.ser.write((data))

    def wait_move_complete(self):
    # result: (status, position, force)
        while True:
            data = self.status()
            if data[5] != 0x00:
                return (-1, data[7], data[8])
            if data[3] == 0x79:
                return (2, data[7], data[8])
            if data[3] == 0xb9:
                return (1, data[7], data[8])
            if data[3] == 0xf9:
                return (0, data[7], data[8])

    def _calc_crc(self, command):
        crc_registor = 0xFFFF
        for data_byte in command:
            tmp = crc_registor ^ data_byte
            for _ in range(8):
                if(tmp & 1 == 1):
                    tmp = tmp >> 1
                    tmp = 0xA001 ^ tmp
                else:
                    tmp = tmp >> 1
            crc_registor = tmp
        crc = bytearray(struct.pack('<H', crc_registor))
        return crc