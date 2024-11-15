import numpy as np
import socket
import json


class KalmanFilter:
    def __init__(self):
        self.q = np.array([9.99999890e-01 , 1.83339006e-04, 2.62600908e-04, -3.42677467e-04])  # Initial quaternion
        self.p = np.eye(4)  # Initial covariance matrix
        self.R = np.eye(4) * 0.01  # Measurement noise
        self.Q = np.eye(4) * 0.001  # Process noise

    def update(self, gyro, accel, mag, dt):
        F = np.eye(4) + 0.5 * dt * np.array([
            [0, -gyro[0], -gyro[1], -gyro[2]],
            [gyro[0], 0, gyro[2], -gyro[1]],
            [gyro[1], -gyro[2], 0, gyro[0]],
            [gyro[2], gyro[1], -gyro[0], 0]
        ])

        # Predict
        self.q = np.dot(F, self.q)
        self.p = np.dot(F, np.dot(self.p, F.T)) + self.Q

        # Update
        if np.linalg.norm(accel) == 0 or np.linalg.norm(mag) == 0:
            return

        accel = accel / np.linalg.norm(accel)
        mag = mag / np.linalg.norm(mag)

        z = self.q  # Measurement
        H = np.eye(4)
        y = z - np.dot(H, self.q)
        S = np.dot(H, np.dot(self.p, H.T)) + self.R
        K = np.dot(self.p, np.dot(H.T, np.linalg.inv(S)))
        self.q = self.q + np.dot(K, y)
        self.p = self.p - np.dot(K, np.dot(H, self.p))

        self.q = self.q / np.linalg.norm(self.q)  # Normalize quaternion


def send_data(data):
    server_address = ('localhost', 6969)
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(server_address)

    try:
        data_list = data.tolist() if isinstance(data, np.ndarray) else data
        json_data = json.dumps({"array": data_list})
        client_socket.sendall(json_data.encode('utf-8'))
    finally:
        client_socket.close()


def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        s.connect(('10.254.254.254', 1))
        ip_address = s.getsockname()[0]
    except Exception:
        ip_address = '127.0.0.1'
    finally:
        s.close()
    return ip_address


def parse_data_line(line):
    try:
        parts = line.strip().split(':')
        if len(parts) != 2:
            raise ValueError("Line does not have exactly one ':' character.")
        key, values = parts
        values = values.strip()
        return np.array([float(x) for x in values.split(',')])
    except ValueError as e:
        print(f"Error parsing line '{line}': {e}")
        return None


def start_server():
    ip_address = get_ip_address()
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((ip_address, 5000))
    server_socket.listen(1)
    print(f"Server started on {ip_address}. Waiting for connections...")

    kalman_filter = KalmanFilter()

    while True:
        client_socket, client_address = server_socket.accept()
        print(f"Connection from {client_address} has been established!")

        accel_data = None
        mag_data = None
        gyro_data = None
        dt = 1 / 256  # Time step

        while True:
            try:
                data = client_socket.recv(1024)
                if not data:
                    break

                data_str = data.decode('utf-8').strip()
                if not data_str:
                    continue

                lines = data_str.splitlines()

                for line in lines:
                    line = line.strip()
                    if line.startswith("Accelerometer:"):
                        accel_data = parse_data_line(line)
                    elif line.startswith("Magnetometer:"):
                        mag_data = parse_data_line(line)
                    elif line.startswith("Gyroscope:"):
                        gyro_data = parse_data_line(line)

                if gyro_data is not None and mag_data is not None and accel_data is not None:
                    print(f"Received data - Accel: {accel_data}, Mag: {mag_data}, Gyro: {gyro_data}")
                    #kalman_filter.update(gyro_data, accel_data, mag_data, dt)
                    # #quaternion = kalman_filter.q
                    # if len(quaternion) == 4:
                    #     send_data(quaternion)
                    #     print(quaternion)
                    # else:
                    #     print("Quaternion data is not the expected size.")
                else:
                    print("Received data does not contain all required lines.")

            except Exception as e:
                print(f"Unexpected error: {e}")

        client_socket.close()


if __name__ == "__main__":
    start_server()
