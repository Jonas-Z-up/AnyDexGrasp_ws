import numpy as np
import socket
import time

class dh_client_socket(object):
    def connect_device(self, host_add, port):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ret = -1
        ret = self.client_socket.connect_ex((host_add,port))
        print('Connect recv: ',ret)
        if(ret < 0) :
            print('Connect Error')
            ret = -1
        else :
            print('Connect Success')
            ret = 0
        return ret

    def disconnect_device(self):
        self.client_socket.close()
        return True

    def device_wrire(self, nDate):
        length = 0
        date = bytes(nDate)
        length = self.client_socket.send(date)
        return length

    def device_read(self, length):
        date = self.client_socket.recv(length)
        return date



class dh_socket_gripper(object):
    def __init__(self,):
        self.m_device = dh_client_socket()
        self.gripper_ID = 0x01

    def open(self, host_add, port):
        ret = 0
        ret = self.m_device.connect_device(host_add, port)
        if(ret < 0) :
            print('open failed')
            return ret
        else :
            print('open successful')
            return ret

    def close(self, ):
        self.m_device.disconnect_device()

    def WriteRegisterFunc(self,index, value):
        send_buf = [0 for i in range(14)]
        send_buf[0] = 0xFF
        send_buf[1] = 0xFE
        send_buf[2] = 0xFD
        send_buf[3] = 0xFC
        send_buf[4] = self.gripper_ID
        send_buf[5] = (index>>8)&0xFF
        send_buf[6] = index&0xFF
        send_buf[7] = 0x01
        send_buf[8] = 0x00

        send_buf[9] = value&0xFF
        send_buf[10] = (value>>8)&0xFF
        send_buf[11] = 0x00
        send_buf[12] = 0x00

        send_buf[13] = 0xFB

        send_temp = send_buf
        ret = False
        retrycount = 3

        while (ret == False):
            ret = False

            if(retrycount < 0) :
                break
            retrycount = retrycount - 1

            wdlen = self.m_device.device_wrire(send_temp)
            if(len(send_temp) != wdlen) :
                print('write error ! write : ', send_temp)
                continue

            rev_buf = self.m_device.device_read(wdlen)
            if(len(rev_buf) == wdlen):
                ret = True
        return ret

    def ReadRegisterFunc(self,index):
        send_buf = [0 for i in range(14)]
        send_buf[0] = 0xFF
        send_buf[1] = 0xFE
        send_buf[2] = 0xFD
        send_buf[3] = 0xFC

        send_buf[4] = self.gripper_ID

        send_buf[5] = (index>>8)&0xFF
        send_buf[6] = index&0xFF
        send_buf[7] = 0x00
        send_buf[8] = 0x00

        send_buf[9] = 0x00
        send_buf[10] = 0x00
        send_buf[11] = 0x00
        send_buf[12] = 0x00

        send_buf[13] = 0xFB

        send_temp = send_buf
        ret = False
        retrycount = 3
        value = -1

        while (ret == False ):
            ret = False

            if(retrycount < 0):
                break
            retrycount = retrycount - 1

            wdlen = self.m_device.device_wrire(send_temp)
            if(len(send_temp) != wdlen):
                print('write error ! write : ', send_temp)
                continue

            rev_buf = self.m_device.device_read(wdlen)
            if(len(rev_buf) == wdlen) :
                value = ((rev_buf[9]&0xFF)|(rev_buf[10] << 8))
                ret = True
            #print('read value : ', value)
        return value

    def Initialization(self):
        self.WriteRegisterFunc(0x0802,0x00)
        
    def SetTargetPosition(self,refpos):
        self.WriteRegisterFunc(0x0602,refpos)

    def SetTargetRotation(self,refpos):
        self.WriteRegisterFunc(0x0702,refpos)

    def SetTargetForce(self,force):
        self.WriteRegisterFunc(0x0502,force)
        
    def SetTargetSpeed(self,speed):
       self.WriteRegisterFunc(0x0104,speed)

    def GetCurrentPosition(self):
        return self.ReadRegisterFunc(0x0602)

    def GetCurrentTargetForce(self):
        return self.ReadRegisterFunc(0x0502)

    #def GetCurrentTargetSpeed(self):
     #   return self.ReadRegisterFunc(0x0104);

    def GetInitState(self):
        return self.ReadRegisterFunc(0x0802)

    def GetGripState(self):
        return self.ReadRegisterFunc(0x0F01)


class DH3:
    def __init__(self, ip = '192.168.1.29', port = 8888):
        self.m_gripper = dh_socket_gripper()
        initstate = 0
        g_state = 0
        force = 70
        speed = 100

        self.m_gripper.open(ip, port)

        # initstate = self.m_gripper.GetInitState()
        # while(initstate != 1) :
        #     print('Send grip init')
        #     self.m_gripper.Initialization()
        #     initstate = self.m_gripper.GetInitState()
        #     time.sleep(0.2)
            
        self.m_gripper.SetTargetForce(force)
        self.m_gripper.SetTargetSpeed(speed)
        self.m_gripper.SetTargetRotation(100)
        self.m_gripper.SetTargetPosition(0)
        


    def set_ready_pose(self, position=0, rotation=100):
        self.m_gripper.SetTargetPosition(50)
        self.m_gripper.SetTargetRotation(rotation)
        self.m_gripper.SetTargetPosition(position)

    def set_pose(self, position, rotation):
        self.m_gripper.SetTargetPosition(position)
        self.m_gripper.SetTargetRotation(rotation)
        
        

    def open_gripper(self, angle=np.array([0, 100]), sleep_time=0.2):
        print("Open gripper")
        position, rotation = angle
        self.set_pose(int(position), int(rotation))
        time.sleep(sleep_time)

    def close_gripper(self, position=0, sleep_time=0.2):
        print("Close gripper")
        self.m_gripper.SetTargetPosition(int(position))
        time.sleep(sleep_time)

if __name__ == '__main__':
    a = DH3()
    a.open_gripper(np.array([100, 100]))