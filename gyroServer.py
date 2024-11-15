import socket
import json

def send_data(data):
    server_address = ('localhost', 6969)
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(server_address)

    try:
        # Serialize list to JSON
        json_data = json.dumps({"array": data})
        client_socket.sendall(json_data.encode('utf-8'))
    finally:
        client_socket.close()

def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        # Doesn't even need to be reachable
        s.connect(('10.254.254.254', 1))
        ip_address = s.getsockname()[0]
    except Exception:
        ip_address = '127.0.0.1'
    finally:
        s.close()
    return ip_address

def start_server():
    ip_address = get_ip_address()
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((ip_address, 5000))
    server_socket.listen(1)
    print(f"Server started on {ip_address}. Waiting for connections...")

    while True:
        client_socket, client_address = server_socket.accept()
        print(f"Connection from {client_address} has been established!")

        while True:
            data = client_socket.recv(1024)
            if not data:
                break

            decoded_data = data.decode('utf-8')
            print(f"Received data: {decoded_data}")

            # Split decoded data by comma and convert to numbers
            number_list = [float(x) for x in decoded_data.split(',')]

            # Group the numbers into sub-arrays of 3 elements each and send each group
            grouped_data = [number_list[i:i+3] for i in range(0, len(number_list), 3)]
            for group in grouped_data:
                send_data(group)
                print(f"Sent data: {group}")

            # Clear the list after sending the data
            grouped_data.clear()

        client_socket.close()

if __name__ == "__main__":
    start_server()
