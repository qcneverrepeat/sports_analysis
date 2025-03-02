import cv2, torch, joblib
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchreid import models as reidmodels
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont


class playerDetection(object):

    def __init__(self):
        self.clk_points = []
        self.wn_diy_get_court_coords = '获取底线坐标'
        self.wn_detect = '运动员追踪'

        self.box_color = (0, 255, 0)
        self.box_thickness = 1
        self.label_color = (255, 255, 255)
        self.label_bg_color = (0, 255, 0)
        self.player_dot_color = (255, 255, 255)

        self.tracker = DeepSort(max_age = 5)


    def load_video(self, dir = "./data/video_01__momota_lindan.mp4"):
        self.cap = cv2.VideoCapture(dir)
        # self.cap = cv2.VideoCapture(0) # 摄像头

    def load_yolo_model(self, dir = "./model/yolo11n.pt", is_gpu = True):
        self.yolo_model = YOLO(dir)
        if is_gpu and torch.backends.mps.is_available(): self.yolo_model.to('mps')

    def load_reid_model(self, is_gpu = True):
        self.reid_model = reidmodels.osnet_x1_0(pretrained=True)  # 加载预训练的 OSNet 模型
        self.reid_model.eval()  # 推理模式
        if is_gpu and torch.backends.mps.is_available(): self.reid_model.to('mps')
        self.reid_model_device = next(self.reid_model.parameters()).device # 模型所在的设备：cpu/mps, 后续推理向量需要据此移动设备

    def get_court_coords(self, type = 'diy'):
        """
        获取球场四个角在视频中的坐标
        1.智能检测模式 type = 'auto', TBD
        2.手动点击模式 type = 'diy'
        """
        if type == 'auto': 
            self.court_coords = self._auto_get_court_coords()
        elif type == 'diy':
            self.court_coords = self._diy_get_court_coords()
        
    def detect_and_track(self):
        """
        cv逐帧读取视频,yolo逐帧预测,获取运动员真实坐标并绘制在虚拟球场上
        预测完成后返回真实坐标的时间序列,用于后续数据分析

        @return player_track_dict {track_id1: [(ms, track),...], track_id2: [(ms, track),...], ...}
        """

        # 初始化运动员的轨迹信息
        player_track_dict = defaultdict(list)
        
        # 创建GUI显示窗口
        cv2.namedWindow(self.wn_detect, cv2.WINDOW_NORMAL)
        self.trail_layer = None # 初始化轨迹图层

        while True:
            # cv2逐帧读取视频
            ret, frame = self.cap.read()
            if not ret:
                print("detect: 无法读取视频帧")
                break

            # 1.使用 YOLO 对帧进行目标检测
            results = self.yolo_model.predict(frame, stream = False, conf = 0.5, batch = 3)[0]

            # 2.去掉场内运动员之外的检测对象
            boxes_filter = self._filter_boxes(results.boxes)

            # 3.使用 DeepSORT 进行目标追踪, 追踪结果写入运动轨迹信息中: player_track_dict
            tracks = self._track_boxes(boxes_filter, frame)
            for track in tracks: player_track_dict[track.track_id].append( (self.cap.get(cv2.CAP_PROP_POS_MSEC), track) )

            # 4.检测结果绘制到视频 frame 中
            frame = self._draw_track_res(tracks, frame)

            # 5.对 frame 做右侧扩展，显示球场虚拟正投影
            expand_frame, mapping = self._expand_court_proj_2_frame(frame)

            # 6.获取运动员在场地真实坐标, 并绘制投影
            expand_frame = self._draw_player_position(boxes_filter, expand_frame, mapping)

            # 7.GUI弹窗展示帧和追踪框等
            cv2.imshow(self.wn_detect, expand_frame)

            # 监听按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('detect: 主动退出视频')
                break 
                self.cap.release()
                cv2.destroyAllWindows()

        return player_track_dict
    
    def _track_boxes(self, boxes, frame):
        """
        使用 DeepSORT 进行目标追踪

        params:
        - boxes (list) yolo 返回帧的 result[0].boxes
        - frame 原始帧

        return:
        - tracks (list): 每个元素为一个追踪对象/轨迹, track_id, ...
        """
        bbs, embeds = [], []

        for box in boxes:
            x, y = map(int, box.xyxy[0][0:2])
            w, h = map(int, box.xywh[0][2:4])
            bbs.append( ([x, y, w, h], box.conf.item(), int(box.cls.item())) ) # list of tuples of ( [left,top,w,h], confidence, detection_class )

            obj_chip = self._chipper(frame, [x, y, w, h]) # 检测对象的图像切片
            obj_chip_trans = self._preprocess_4_embedding(obj_chip) # 转成 reid 模型需要的tensor
            embed = self.reid_model(obj_chip_trans.to(self.reid_model_device)) # 切片 -> 特征向量
            embeds.append(embed.squeeze().cpu().detach().numpy()) # list of feature vector, Deepsort中除了reid部分都需要CPU运行

        tracks = self.tracker.update_tracks(bbs, embeds = embeds)
        tracks = [track for track in tracks if track.is_confirmed()] # 筛掉未确认的轨迹

        return tracks

    def _chipper(self, frame, box):
        """
        根据边界安全地框裁剪目标区域

        params:
        - frame (np.ndarray): 原始帧，形状为 (H, W, C)
        - box (list): 边界框，[left, top, width, height]

        return:
        - obj_chip: 裁剪后的目标图像块 NumPy 数组
        """

        left, top, width, height = box

        # 确保边界框在图像范围内
        h, w, _ = frame.shape
        left = max(0, int(left))
        top = max(0, int(top))
        right = min(w, int(left + width))
        bottom = min(h, int(top + height))

        # 裁剪目标区域
        obj_chip = frame[top:bottom, left:right]

        return obj_chip

    def _preprocess_4_embedding(self, chips):
        """
        对 OpenCV 图像进行预处理，使其符合 reid model 的输入要求
        """
        # OpenCV 默认是 BGR 格式，需要转换为 RGB
        image = cv2.cvtColor(chips, cv2.COLOR_BGR2RGB)

        # 定义预处理步骤
        transform = transforms.Compose([
            transforms.ToPILImage(),               # 转换为 PIL 图像
            transforms.Resize((256, 128)),         # 调整大小到 (256, 128)
            transforms.ToTensor(),                 # 转换为 Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        ])

        return transform(image).unsqueeze(0)  # 增加 batch 维度

    def _draw_track_res(self, tracks, frame):
        """
        将追踪结果 tracks 添加到视频帧 frame 中
        """
        for track in tracks:
            track_id = track.track_id # 轨迹ID
            left, top, right, bottom = map(int, track.to_ltrb())  # 边界框
            cv2.rectangle(frame, (left, top), (right, bottom), self.box_color, self.box_thickness) # 绘制, 会直接修改 frame

        return frame

    def _filter_boxes(self, boxes):
        """
        过滤掉除了场内运动员之外的对象
        1.类别不是person的
        2.置信度小于50%的
        3.场地外的对象去掉 TBD: 是否有更合理的做法
        """
        boxes_filter = []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 边界框左上&右下角坐标
            x1_rc, y1_rc = self._pixel_2_real_coords(x1, y1) # 边界框中心对应的场上真实坐标
            x2_rc, y2_rc = self._pixel_2_real_coords(x2, y2)
            is_in_court = self._is_intersect([x1_rc, y1_rc, x2_rc, y2_rc], [0, 0, 6.1, 13.4]) # 边界框是否和场内有交集

            conf = box.conf.item()  # 置信度
            cls_id = int(box.cls.item())  # 类别 ID
            cls_name = self.yolo_model.names[cls_id]  # 类别名称

            if cls_name == 'person' and conf > 0.5 and is_in_court: boxes_filter.append(box)

        return boxes_filter

    def _is_intersect(self, box, region):
        """
        判断 box 是否与 region 存在交集

        参数:
            box: [x_min, y_min, x_max, y_max]，边界框的坐标。
            region: [x_min, y_min, x_max, y_max]，指定区域的坐标。

        返回:
            bool: 如果存在交集，返回 True；否则返回 False。
        """
        # 解包边界框和区域的坐标
        box_x_min, box_y_min, box_x_max, box_y_max = box
        region_x_min, region_y_min, region_x_max, region_y_max = region

        # 判断是否相交
        if (box_x_max > region_x_min and
            box_x_min < region_x_max and
            box_y_max > region_y_min and
            box_y_min < region_y_max):
            return True
        return False

    def _pixel_2_real_coords(self, x, y):
        """
        像素坐标(x,y) -> 场上的真实坐标, 单位m, 左上角(0,0)
        """
        player_pixel_coords = np.array([[x, y]], dtype=np.float32)
        player_real_coords = cv2.perspectiveTransform(player_pixel_coords[None, :, :], self.perspective_matrix)[0][0]
        return player_real_coords

    def _draw_player_position(self, boxes, frame, mapping):
        """
        @params 
        - mapping: virtual_x_offset, virtual_y_offset, scalar (virtual_court_height / 13.4)

        1.boxes 线性变换 得到球员在场上的真实坐标, 单位m, 左上角(0,0): player_real_coords
        2.在 frame 中显示一个虚拟球场正投影, 实时显示球员位置: player_mapped_coords

        @return
        - frame: 在虚拟球场中绘制了球员位置的帧
        """
        virtual_x_offset, virtual_y_offset, scalar = mapping

        # 对每一帧, 遍历识别到的对象
        for box in boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 边界框左上&右下角坐标

            # 1.获取球员位置
            player_real_coords = self._pixel_2_real_coords((x1+x2)/2, y2) # 球员的真实场上坐标, 单位m, 左上角(0,0)
            player_mapped_coords = player_real_coords * scalar + np.array([virtual_x_offset, virtual_y_offset]) # 映射球员坐标到虚拟球场
            x, y = player_mapped_coords.astype(int)[0], player_mapped_coords.astype(int)[1]

            # 2.绘制球员轨迹
            if self.trail_layer is None: self.trail_layer = np.zeros_like(frame, dtype=np.float32) # 初始化轨迹涂层
            cv2.circle(self.trail_layer, (x, y), 2, self.player_dot_color, -1) # 在轨迹图层上绘制点
            frame = cv2.addWeighted(frame.astype(np.float32), 1.0, self.trail_layer, 1.0, 0).astype(np.uint8) # 将轨迹图层叠加到扩展帧
            self.trail_layer *= 0.95  # 降低轨迹亮度，使其逐渐变淡

        return frame

    def _expand_court_proj_2_frame(self, frame):
        """
        处理每一帧：扩展空间、绘制虚拟球场正投影
        球场左上角: 0.2*帧高度,0.2*帧高度
        球场长度: 0.6*帧高度

        :param frame: 原始帧
        :return: 
        - expanded_frame 处理后的帧
        - mapping 放缩信息
        """
        # 1.圈定虚拟球场范围
        frame_height, frame_width = frame.shape[0], frame.shape[1]
        virtual_x_offset, virtual_y_offset = int(frame_height * 0.1) + frame_width, int(frame_height * 0.1) # 虚拟球场的左上角位置
        virtual_court_height = int(0.8 * frame_height) # 虚拟球场的长度(纵向排列): frame_width 的一半
        virtual_court_width = int(virtual_court_height / 13.4 * 6.1) # 虚拟球场的宽度(横向排列): 由长度等比例折算

        # 2.创建扩展帧
        expanded_width = virtual_court_width + 2 * virtual_y_offset
        expanded_frame = np.full((frame_height, expanded_width + frame_width, 3), 128, dtype=np.uint8)
        expanded_frame[:, :frame_width] = frame  # 将原始帧复制到扩展帧左侧

        # 3.绘制球场边界
        top_left = (virtual_x_offset, virtual_y_offset)
        bottom_right = (virtual_x_offset + virtual_court_width, virtual_court_height + virtual_y_offset)
        cv2.rectangle(expanded_frame, top_left, bottom_right, (255, 255, 255), 2)

        # 4.绘制网线
        mid_line_start = (virtual_x_offset, virtual_y_offset + virtual_court_height // 2)
        mid_line_end = (virtual_x_offset + virtual_court_width, virtual_y_offset + virtual_court_height // 2)
        cv2.line(expanded_frame, mid_line_start, mid_line_end, (255, 255, 255), 2)

        # 5.绘制半场线
        top_half_line_start = (virtual_x_offset + virtual_court_width // 2, virtual_y_offset)
        top_half_line_end = (virtual_x_offset + virtual_court_width // 2, virtual_y_offset + int(virtual_court_height * 4.72 / 13.4))
        bottom_half_line_start = (virtual_x_offset + virtual_court_width // 2, virtual_y_offset + int(virtual_court_height * 8.68 / 13.4))
        bottom_half_line_end = (virtual_x_offset + virtual_court_width // 2, virtual_y_offset + virtual_court_height)
        cv2.line(expanded_frame, top_half_line_start, top_half_line_end, (255, 255, 255), 1)
        cv2.line(expanded_frame, bottom_half_line_start, bottom_half_line_end, (255, 255, 255), 1)

        # 6.绘制双打边线
        double_line_offset = int(virtual_court_width * 0.46/6.1)  
        left_double_line_start = (virtual_x_offset + double_line_offset, virtual_y_offset)
        left_double_line_end = (virtual_x_offset + double_line_offset, virtual_y_offset + virtual_court_height)
        right_double_line_start = (virtual_x_offset + virtual_court_width - double_line_offset, virtual_y_offset)
        right_double_line_end = (virtual_x_offset + virtual_court_width - double_line_offset, virtual_y_offset + virtual_court_height)
        cv2.line(expanded_frame, left_double_line_start, left_double_line_end, (255, 255, 255), 1)
        cv2.line(expanded_frame, right_double_line_start, right_double_line_end, (255, 255, 255), 1)

        # 7. 绘制单打发球线
        front_service_line_y = virtual_y_offset + int(virtual_court_height * 4.72 / 13.4)
        back_service_line_y = virtual_y_offset + int(virtual_court_height * 8.68 / 13.4)
        cv2.line(expanded_frame, (top_left[0], front_service_line_y), (bottom_right[0], front_service_line_y), (255, 255, 255), 1)
        cv2.line(expanded_frame, (top_left[0], back_service_line_y), (bottom_right[0], back_service_line_y), (255, 255, 255), 1)

        # 8. 绘制双打发球界
        double_serve_offset = int(virtual_court_width * 0.76/6.1)  
        top_double_line_start = (virtual_x_offset, virtual_y_offset + double_serve_offset)
        top_double_line_end = (virtual_x_offset + virtual_court_width, virtual_y_offset + double_serve_offset)
        bottom_double_line_start = (virtual_x_offset, virtual_y_offset + virtual_court_height - double_serve_offset)
        bottom_double_line_end = (virtual_x_offset + virtual_court_width, virtual_y_offset + virtual_court_height - double_serve_offset)
        cv2.line(expanded_frame, top_double_line_start, top_double_line_end, (255, 255, 255), 1)
        cv2.line(expanded_frame, bottom_double_line_start, bottom_double_line_end, (255, 255, 255), 1)

        # 提供场地偏移和放缩信息
        mapping = (virtual_x_offset, virtual_y_offset, virtual_court_height / 13.4)

        return expanded_frame, mapping

    def _auto_get_court_coords(self):
        pass

    def _diy_get_court_coords(self):
        """
        手动点击两底线, 得到场地边角在图像中的坐标
        return np.array [c1_n, c2_n, c3_n, c4_n], 按照球场左上、右上、左下、右下的顺序排列

        TBD:
        暂不支持超出场地范围的边角
        暂不支持动态视角的场地
        """
        # 1.读取首帧用于展示
        ret, frame = self.cap.read()
        if not ret:
            print("_diy_get_court_coords: 无法读取视频帧")
            return 

        # 2.创建窗口并绑定鼠标回调函数
        cv2.namedWindow(self.wn_diy_get_court_coords)
        cv2.setMouseCallback(self.wn_diy_get_court_coords, self.__mouse_callback)

        # 3.点击绑定边界点
        while True:
            display_frame = frame.copy() # 确保每一帧绘制都是基于原始的干净帧

            # 显示提示信息
            display_frame = self._add_text(display_frame, "请手动标注场地两条双打底线", (10,30))
            display_frame = self._add_text(display_frame, "最多选择 4 个点, 退格键撤销上一次选择", (10,60))
            display_frame = self._add_text(display_frame, "回车键最终确认", (10,90))

            # 绘制已标记的点
            for i, point in enumerate(self.clk_points):
                cv2.circle(display_frame, point, 5, (0, 0, 255), -1)  # 红色圆点

            # 绘制第一条底线
            if len(self.clk_points) >= 2:
                cv2.line(display_frame, self.clk_points[0], self.clk_points[1], (0, 255, 0), 2)

            # 绘制第二条底线
            if len(self.clk_points) >= 4:
                cv2.line(display_frame, self.clk_points[2], self.clk_points[3], (0, 255, 0), 2)

            # 绘制4个点之后，按 左上/右上/左下/右下 显示 1/2/3/4
            if len(self.clk_points) == 4:
                court_coords = np.array(self.clk_points)
                court_coords_normed = self._court_coords_normalization(court_coords)
                for i, point in enumerate(court_coords_normed):
                    cv2.putText(display_frame, str(i + 1), (point[0] + 10, point[1] + 10), # 标记1/2/3/4
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # 回车键确认
                if len(self.clk_points) == 4:
                    msg = "用户确认标注完成！"
                    self._add_text(display_frame, msg, (300,450))
                    print(msg)
                    break # 只有回车确认才能进入下一步
                else:
                    msg = "请标注 4 个点后再确认！"
                    self._add_text(display_frame, msg, (300,450))
                    print(msg)
            elif key == 127:  # 退格键撤销
                if len(self.clk_points) > 0:
                    removed_point = self.clk_points.pop()
                    print(f"撤销点: {removed_point}")
            elif key == 27:  # ESC 键退出
                msg = "用户取消操作！"
                self._add_text(display_frame, msg, (300,450))
                print(msg)
                raise Exception

            # 显示图像
            cv2.imshow(self.wn_diy_get_court_coords, display_frame)

        # 4.释放首帧: 时间指针跳回0, 关闭窗口
        self.cap.set(cv2.CAP_PROP_POS_MSEC, 0)
        cv2.destroyAllWindows()

        # 5.返回标记点的坐标
        print("标记点的坐标, 左上、右上、左下、右下：\n", court_coords_normed)
        return court_coords_normed

    def _court_coords_normalization(self, court_coords):
        """
        np.array [c1, c2, | c3, c4], 按照用户标注顺序的两底线坐标, cx为某坐标(x,y)
        ->
        np.array [c1_n, c2_n, c3_n, c4_n], 按照球场左上、右上、左下、右下的顺序排列
        """

        # 1.Y求和更小的底线(图像上方)作为上底线，剩下一条作为下底线
        if court_coords[0:2].sum(axis=0)[1] < court_coords[2:4].sum(axis=0)[1]:
            top_line, bot_line = court_coords[0:2], court_coords[2:4]
        else:
            top_line, bot_line = court_coords[2:4], court_coords[0:2]

        # 2.找到上底线中靠左的点，作为凸包顺时针的起点
        start_point = top_line[np.argmin(top_line[:, 0])]
        hull = cv2.convexHull(court_coords, clockwise=False).reshape(-1,2)
        start_index = np.where((hull==start_point).all(axis=1))[0][0]
        sorted_points = np.roll(hull, - start_index, axis=0) # 沿第一个轴反向滑动 start_index 个元素, 让开始的元素到首位
        
        return sorted_points
    
    def get_perspective_matrix(self):
        """
        获取像素坐标->球场坐标的映射矩阵
        """
        # 球场坐标
        real_points = np.float32([
            [0, 0],       # 左上角
            [6.1, 0],    # 右上角
            [6.1, 13.4],    # 右下角
            [0, 13.4]     # 左下角
        ])
        self.perspective_matrix = cv2.getPerspectiveTransform(self.court_coords.astype(np.float32), real_points)
        
    def __mouse_callback(self, event, x, y, flags, param):
        """
        鼠标回调函数
        标注坐标写入self.clk_points
        """
        
        if event == cv2.EVENT_LBUTTONDOWN: # 鼠标左键点击事件
            if len(self.clk_points) < 4:  # 最多允许标记 4 个点
                self.clk_points.append((x, y))
                print(f"标记点: ({x}, {y})")
            else:
                print("最多标记 4 个点，重新选择请按退格键撤销上一次选择")

    def _add_text(self, 
                    image_cv, 
                    text, 
                    text_position = (50, 100), # 文字左上角位于 (50, 100)
                    text_color = (255, 255, 255), # RGB
                    font_dir = "./resource/字体圈欣意吉祥宋~可商用.ttf", 
                    font_size = 20):
        """
        cv2无法显示中文,用pillow中转
        """

        # 将 OpenCV 图像转换为 Pillow 图像
        image_pil = Image.fromarray(image_cv)

        # 绘图
        draw = ImageDraw.Draw(image_pil)
        font = ImageFont.truetype(font_dir, font_size)
        draw.text(text_position, text, font=font, fill=text_color)

        # 将 Pillow 图像转换回 OpenCV 图像
        image = np.array(image_pil)

        return image


if __name__ == "__main__":

    player_detection = playerDetection()

    # 1.载入视频 self.cap
    player_detection.load_video(dir = "./data/video_01__momota_lindan__1_shot.mp4")

    # 2.载入模型 self.yolo_model / self.reid_model
    player_detection.load_yolo_model(dir = "./model/yolo11n.pt", is_gpu = True)
    player_detection.load_reid_model()

    # 3.获取场地四角的图像坐标
    player_detection.get_court_coords(type = 'diy')

    # 4.像素坐标->球场坐标的映射矩阵
    player_detection.get_perspective_matrix()

    # 5.逐帧做yolo人物检测, 获取运动员轨迹
    player_track_dict = player_detection.detect_and_track()

    # joblib.dump(player_track_dict, './data/player_track_dict')

