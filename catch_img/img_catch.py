import sys
import cv2
import os
import curses  # 用于处理命令行输入
sys.path.append("..")
import Socket
import queue

#捕获照片, 用于模型训练
def capture_images(stdscr):
    # 清除屏幕
    curses.curs_set(0)  # 隐藏光标
    stdscr.nodelay(1)  # 非阻塞输入
    stdscr.timeout(100)  # 设置输入超时

    # 创建目录保存图片
    if not os.path.exists('captured_images_to_train'):
        os.makedirs('captured_images_to_train')
    # 获取已有文件数量，避免覆盖
    existing_images = len([f for f in os.listdir('captured_images_to_train') if f.endswith('.jpg')])
    image_count = existing_images  # 从已有文件数量开始计数

    # 打开默认摄像头
    cap = cv2.VideoCapture(8)

    #启动Socket
    data_queue = queue.Queue()
    socket_test = Socket.SocketRecoCameraSender(data_queue)
    socket_test.start()

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 设置显示窗口的分辨率
    display_width = 640
    display_height = 480

    stdscr.addstr(0, 0, "按 'c' 键拍照，按 'q' 键退出")
    stdscr.refresh()

    while True:
        # 从摄像头读取帧
        ret, frame = cap.read()

        #将frame数据通过socket传输
        data_queue.put(frame)

        if not ret:
            print("无法读取帧")
            break

        # 调整帧的大小以匹配显示窗口的分辨率
        frame_resized = cv2.resize(frame, (display_width, display_height))

        # 获取按键输入
        key = stdscr.getch()

        if key == ord('c'):  # 按下 'c' 键时拍照并保存
            image_count += 1
            filename = f'captured_images_to_train/image_{image_count:03d}.jpg'  # 生成带编号的文件名
            cv2.imwrite(filename, frame)  # 保存原始分辨率的图片
            stdscr.addstr(1, 0, f"照片 {filename} 已保存", curses.A_NORMAL)
            stdscr.refresh()

        elif key == ord('q'):  # 按 'q' 键退出
            break

    # 释放摄像头
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    curses.wrapper(capture_images)
