import sys

def main():
    # 获取命令行参数
    if len(sys.argv) > 1:
        # 打印传递的参数
        received_value = sys.argv[1]
        print(f"Received value: {received_value}")
        if received_value.lower() == 'true':
            print("Received True value.")
        else:
            print("Received a non-True value.")
    else:
        print("No value received.")

# if __name__ == '__main__':
main()
