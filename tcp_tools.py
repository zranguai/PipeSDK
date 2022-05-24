import struct  # 防止tcp粘包
import traceback


def send_msg(sock, msg, timeout=30):
    try:
        sock.settimeout(timeout)
        # struct将任意大小转换成四字节
        msg = struct.pack("i", len(msg)) + msg
        sock.sendall(msg)
        return True
    except:
        traceback.print_exc()
        return None


def recvall(sock, n, timeout):
    try:
        sock.settimeout(timeout)
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data
    except:
        traceback.print_exc()
        return None


def recv_msg(sock, timeout=30):
    """
    1. 为了防止粘包，先接受该数据的长度
    2. 根据该数据的长度来接受该数据
    :param sock:
    :param timeout:
    :return: type: bytearray
    """
    try:
        # raw_msglen是该数据的长度，通过struct变为4个字节
        raw_msglen = recvall(sock, 4, timeout)
        if raw_msglen is None:
            return None
        msglen = struct.unpack("i", raw_msglen)[0]
        data = recvall(sock, msglen, timeout)
        if data is None:
            return None
        return data
    except:
        traceback.print_exc()
        return None
