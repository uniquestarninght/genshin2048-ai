import copy
import logging
import os
import time
from datetime import datetime
from typing import Optional
import cv2
import keyboard
import numpy as np
import pyautogui
from PIL import ImageGrab
from genshin2048 import Genshin2048
from resnet_infer import ResNetPredictor


class H52048Automation:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.resnet_predictor = ResNetPredictor("models/resnet34_filtered_max_2048.pth")
        self.save_train_data = False
        self.save_board = False
        self.game=Genshin2048()
        # 游戏区域坐标
        self.save_move_interval = 1
        self.game_region = None
        self.pic_recognize_mode = "ResNet"
        self.ocr_mode="all"
        # 数字模板字典 {数字: 模板图像}
        self.number_templates = {}
        # 格子位置缓存
        self.cell_positions = None
        # 上一次的棋盘状态和动作（用于缓存）
        self.last_board_state = None
        self.last_action = None
        # 游戏文件夹
        self.game_folder = None
        self.game_folder_created = False
        # 初始化模板
        self._init_templates()

    def _init_templates(self):
        """初始化数字模板"""
        # 加载数字模板
        #模板匹配作者写的很烂，所以这里仅此写了加载模板的代码，匹配逻辑没写
        template_paths = {
            0: "templates/empty.png",  # 空格子
            2: "templates/2.png",
            4: "templates/4.png",
            8: "templates/8.png",
            16: "templates/16.png",
            32: "templates/32.png",
            64: "templates/64.png",
            128: "templates/128.png",
            256: "templates/256.png",
            512: "templates/512.png",
            1024: "templates/1024.png",
            2048: "templates/2048.png"
        }
        for number, path in template_paths.items():
            template = cv2.imread(path)
            if template is not None:
                self.number_templates[number] = template
                self.logger.info(f"成功加载数字模板: {number}")
            else:
                self.logger.warning(f"无法加载模板: {path}")

    def create_game_folder(self):
        """创建游戏文件夹"""
        # 找到最大的game编号
        existing_games = []
        for item in os.listdir('.'):
            if item.startswith('game') and os.path.isdir(item):
                game_num = int(item[4:])  # 去掉'game'前缀
                existing_games.append(game_num)
        new_game_num = max(existing_games) + 1 if existing_games else 1
        self.game_folder = f"game{new_game_num}"
        if not os.path.exists(self.game_folder):
            os.makedirs(self.game_folder, exist_ok=True)
        self.game_folder_created = True
        self.logger.info(f"创建游戏文件夹: {self.game_folder}")

    def save_board_visualization(self, board_state: np.ndarray, action: str, move_count: int):
        """优化版的棋盘可视化保存"""
        # 1. 异步保存，不阻塞主线程
        import threading
        thread = threading.Thread(
            target=self._save_visualization_async,
            args=(board_state.copy(), action, move_count)
        )
        thread.daemon = True
        thread.start()

    def _save_visualization_async(self, board_state: np.ndarray, action: str, move_count: int):
        """异步保存可视化图像"""
        try:
            x, y, w, h = self.game_region
            # 2. 预计算常用值
            vis_height = h + 100
            vis_width = w * 2 + 50
            result_x = w + 50
            # 3. 优化截图和图像处理
            screenshot = ImageGrab.grab(bbox=(x, y, x + w, y + h))
            original_board_img = np.array(screenshot)
            original_board_img = cv2.cvtColor(original_board_img, cv2.COLOR_RGB2BGR)

            # 4. 预分配内存，避免重复创建数组
            if not hasattr(self, '_vis_template') or self._vis_template.shape != (vis_height, vis_width, 3):
                self._vis_template = np.ones((vis_height, vis_width, 3), dtype=np.uint8) * 255
            visualization = self._vis_template.copy()
            # 5. 直接复制图像数据，避免切片操作
            visualization[0:h, 0:w] = original_board_img
            # 6. 优化识别结果可视化
            result_img = self._create_result_visualization_fast(board_state, w, h)
            visualization[0:h, result_x:result_x + w] = result_img
            # 7. 批量绘制文本，减少函数调用
            self._draw_text_batch(visualization, action, move_count, vis_height)
            # 8. 选择性保存（每N步保存一次，或只在重要操作时保存）
            if move_count % self.save_move_interval == 0:
                vis_filename = os.path.join(self.game_folder, f"move_{move_count}.png")
                cv2.imwrite(vis_filename, visualization)
                self.logger.info(f"已保存棋盘可视化: {vis_filename}")

        except Exception as e:
            self.logger.error(f"保存可视化失败: {e}")

    @staticmethod
    def _create_result_visualization_fast(board_state: np.ndarray, width: int, height: int) -> np.ndarray:
        """优化版的识别结果可视化"""
        # 预分配内存
        result_img = np.ones((height, width, 3), dtype=np.uint8) * 255

        cell_height = height // 4
        cell_width = width // 4

        # 预计算所有格子位置
        for row in range(4):
            for col in range(4):
                x1 = col * cell_width
                y1 = row * cell_height
                x2 = x1 + cell_width
                y2 = y1 + cell_height

                # 绘制边框
                cv2.rectangle(result_img, (x1, y1), (x2 - 1, y2 - 1), (200, 200, 200), 1)

                # 只在有数字时绘制
                number = int(board_state[row][col])
                if number > 0:
                    text = str(number)
                    # 预计算文字位置
                    text_x = x1 + (cell_width - 30) // 2  # 估算位置，避免getTextSize
                    text_y = y1 + (cell_height + 20) // 2

                    cv2.putText(result_img, text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        return result_img

    @staticmethod
    def _draw_text_batch(img, action: str, move_count: int, vis_height: int):
        """批量绘制文本信息"""
        font_scale_large = 1
        font_scale_small = 0.8
        font_scale_tiny = 0.5

        texts = [
            (f"Action: {action.upper()}", 10, vis_height - 30, font_scale_large, (0, 0, 255), 2),
            (f"Move: {move_count}", 10, vis_height - 60, font_scale_small, (0, 0, 0), 2),
            (
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 10, vis_height - 10, font_scale_tiny, (0, 0, 0), 1)
        ]

        for text, x, y, scale, color, thickness in texts:
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

    def debug_screenshot(self, description: str = ""):
        """调试截图功能"""
        # 捕获全屏并保存
        full_screen = ImageGrab.grab()
        if self.game_folder_created:
            full_screen.save(os.path.join(self.game_folder, f"full_screen_{description}.png"))
        else:
            full_screen.save(f"full_screen_{description}.png")
        self.logger.info(f"已保存全屏截图")
        # 如果游戏区域已设置，捕获游戏区域
        if self.game_region:
            x, y, w, h = self.game_region
            game_area = ImageGrab.grab(bbox=(x, y, x + w, y + h))
            if self.game_folder_created:
                game_area.save(os.path.join(self.game_folder, f"game_area_{description}.png"))
            else:
                game_area.save(f"game_area_{description}.png")
            self.logger.info(f"已保存游戏区域截图")

    def locate_game_region_manual(self, x: int, y: int):
        """手动设置游戏区域（已知大小为800x800）"""
        self.game_region = (x, y, 800, 800)
        self.logger.info(f"手动设置游戏区域: {self.game_region}")
        self._calculate_cell_positions()
        # 保存调试截图
        self.debug_screenshot("after_manual_setup")
        return True

    def find_game_region_interactive(self):
        """交互式寻找游戏区域"""
        self.logger.info("开始交互式寻找游戏区域...")
        self.logger.info("请将鼠标移动到游戏区域的左上角，然后按回车键")
        input()  # 等待用户按回车
        x1, y1 = pyautogui.position()
        self.logger.info("请将鼠标移动到游戏区域的右下角，然后按回车键")
        input()  # 等待用户按回车
        x2, y2 = pyautogui.position()
        # 计算区域
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        self.game_region = (x, y, w, h)
        self.logger.info(f"交互式设置游戏区域: {self.game_region}")
        self._calculate_cell_positions()
        # 保存调试截图
        self.debug_screenshot("after_interactive_setup")
        return True

    def _calculate_cell_positions(self):
        """根据游戏区域计算每个格子的位置"""
        if not self.game_region:
            return
        x, y, w, h = self.game_region
        # 标准的4x4布局，计算每个格子的位置
        cell_width = w // 4
        cell_height = h // 4
        self.cell_positions = []
        for row in range(4):
            row_positions = []
            for col in range(4):
                # 计算每个格子的坐标（添加边距避免边框）
                margin = 8
                cell_x = x + col * cell_width + margin
                cell_y = y + row * cell_height + margin
                cell_w = cell_width - 2 * margin
                cell_h = cell_height - 2 * margin
                row_positions.append((cell_x, cell_y, cell_w, cell_h))
            self.cell_positions.append(row_positions)

    def capture_cell_image(self, board_pic,row: int, col: int):
        """捕获指定格子的图像"""
        if not self.cell_positions or row < 0 or row >= 4 or col < 0 or col >= 4:
            self.logger.error(f"无效的格子位置: ({row}, {col})")
            return None
        x, y, w, h = self.cell_positions[row][col]
        cell_image = board_pic[y:y + h, x:x + w]
        return cell_image

    @staticmethod
    def resize_template(template):
        height, width = template.shape[:2]
        # 计算裁剪区域
        if height<=150 or width<=150:
            return template
        new_w=150
        new_h=150
        start_x=int((width-new_w)/2)
        start_y=int((height-new_h)/2)
        return template[start_y:start_y+new_h,start_x:start_x+new_w]

    def template_match_cell_number(self, cell_image,template_name):
        """模板匹配单个格子的数字"""
        if cell_image is None:
            return 0
        best_prob=-1
        best_num=0
        for number in template_name:
            template=self.number_templates[number]
            template=self.resize_template(template)
            res = cv2.matchTemplate(cell_image, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val > best_prob:
                best_prob = max_val
                best_num = number
        return best_num,best_prob

    def get_ocr_list(self, move):
        ocr_list = []
        if self.ocr_mode == "all":
            for i in range(4):
                for j in range(4):
                    ocr_list.append([i, j, [0,2,4,8,16,32,64,128,256,512,1024,2048]])
            return ocr_list
        else:
            # if self.pic_recognize_mode != "Template":
            #     raise ValueError("we suggest you use ocr_mode='all' when you use ResNet ")
            self.game.move = move
            self.game.move_tile()
            new_grid = self.game.add_tile(random=False)
            new_board = self.game.get_board()
            # print(new_grid)
            # print(new_board)
            for i in range(4):
                for j in range(4):
                    if (i, j) in new_grid and len(new_grid):
                        ocr_list.append([i, j, [0, 2]])
                    else:
                        if self.ocr_mode == "prob":
                            ocr_list.append([i, j, [new_board[i][j]]])
                        elif self.ocr_mode == "new":
                            ocr_list.append([i, j, [-1]])
            return ocr_list

    def recognize_cell_number(self, cell_image,template_name) :
        """识别单个格子的数字"""
        if cell_image is None:
            return 0,0
        if self.pic_recognize_mode == "ResNet":
            return self.resnet_predictor.predict(cell_image)
        elif self.pic_recognize_mode == "Template":
            # 模板匹配
            return self.template_match_cell_number(cell_image,template_name)
        elif self.pic_recognize_mode == "And":
            res,prob=self.template_match_cell_number(cell_image,template_name)
            if prob>0.9:
                return res,prob
            else:
                return self.resnet_predictor.predict(cell_image)
        #模板匹配效果较差，主播不会用模板匹配，所以用的resnet预测效果更好，但慢.初期可以靠模板匹配大致筛选图片，然后手动分类去训练；
        #还有个粗暴的办法，去网页的sources文件夹找，一个个裁剪，复制粘贴，稍微抖动一下颜色和位置啥的。

    @staticmethod
    def save_training_data(final_num, cell_image):
        if not os.path.exists(f"dataset/{final_num}"):
            os.makedirs(f"dataset/{final_num}")
        cv2.imwrite(f"dataset/{final_num}/{final_num}_{datetime.now().strftime('%H%M%S')}.png", cell_image)

    def recognize_board_state(self, ocr_list,move) :
        """识别整个游戏板的状态"""
        if not self.cell_positions or ocr_list is None:
            return None
        board_state = np.zeros((4, 4), dtype=int)
        if move in [0,1,2,3]:
            self.game.move = move
            self.game.move_tile()
            prob_board_state = self.game.get_board()
        else:
            prob_board_state =self.game.get_board()
        recognition_success = True

        board_pic=ImageGrab.grab()
        board_pic=cv2.cvtColor(np.array(board_pic),cv2.COLOR_RGB2BGR)
        # print(f"prob_board_state:{prob_board_state}")
        for cell in ocr_list:
            conf = 0
            # print(cell)
            row=cell[0]
            col=cell[1]
            template_name=cell[2]
            if len(template_name)!=0:
                if template_name[0]==-1:
                    board_state[row][col]=prob_board_state[row][col]
                    continue
            start_time = time.time()
            while conf<0.9 and time.time() - start_time < 2:
                cell_image = self.capture_cell_image(board_pic,row, col)
                if cell_image is None:
                    recognition_success = False
                    continue
                final_num,conf= self.recognize_cell_number(cell_image,template_name)
                # print(f"识别{row}{col}为{final_num}，置信度{conf}")
                #保存识别到数字图片
                board_state[row][col] = final_num
                if final_num not in template_name :
                    conf=0
                elif self.ocr_mode=="all" and move!=-2:
                    if prob_board_state[row][col]!=0:
                        if final_num!=prob_board_state[row][col]:
                            conf=0
                    else:
                        if final_num not in [0,2]:
                            conf=0
                if self.save_train_data:
                    if conf<0.9:
                        final_num=-1
                        #这段代码主要是把置信度较低的图片人工分类，置信度较低可能识别错误
                    self.save_training_data(final_num,cell_image)
                if conf<0.9:
                    board_pic=ImageGrab.grab()
                    board_pic=cv2.cvtColor(np.array(board_pic),cv2.COLOR_RGB2BGR)
                    print(cell)
                    print(f"识别{row}{col}失败，置信度{conf},重新截图，尝试识别")
        return board_state if recognition_success else None

    def execute_action(self, best_move: int):
        key_map =['w','s','a','d']
        key = key_map[best_move]
        # 确保焦点（可选）
        # if self.game_region:
        #     x, y, w, h = self.game_region
        #     pyautogui.click(x + w // 2, y + h // 2)
        keyboard.press(key)
        time.sleep(0.05)  # 按住 50ms
        keyboard.release(key)
        self.logger.info(f"执行动作: {key} (WASD: {key})")

    def calibrate_templates(self):
        """模板校准工具"""
        self.logger.info("开始模板校准...")
        if not self.game_region:
            self.logger.error("游戏区域未设置，无法进行校准")
            return
        # 捕获当前游戏状态
        board_state = self.recognize_board_state(ocr_list=self.get_ocr_list(0),move=0)
        if board_state is None:
            return
        self.logger.info("校准完成，检查debug文件夹中的图像验证识别效果")

    def run(self):
        """运行自动化程序"""
        self.game.init_board()
        directions=['up','down','left','right']
        self.logger.info("开始H5版2048自动化游戏")
        # 首先创建游戏文件夹
        self.create_game_folder()
        # 设置游戏区域
        self.locate_game_region_manual(x=490, y=400)
        move_count = 0
        ocr_list=[]
        consecutive_failures = 0
        max_consecutive_failures = 10
        best_move = -2
        start=time.time()
        while  consecutive_failures < max_consecutive_failures and best_move!=-1:
                # 识别当前游戏状态
                start_time = time.time()
                if len(ocr_list)!=0:
                    time.sleep(0.15)
                    board_state = self.recognize_board_state(ocr_list,best_move)
                    self.game.set_board(board_state)
                    time.sleep(0.05)
                else:
                    ocr_mode=copy.deepcopy(self.ocr_mode)
                    self.ocr_mode="all"
                    ocr_list=self.get_ocr_list(best_move)
                    board_state = self.recognize_board_state(ocr_list,best_move)
                    self.ocr_mode=ocr_mode
                # 计算最佳动作
                self.game.set_board(board_state)
                print("*"*10,f"回合{move_count}，当前局面：","*"*10)
                best_move = self.game.get_move()
                if self.save_board:
                    self.save_board_visualization(board_state, directions[best_move], move_count)
                print(f"最佳动作：{directions[best_move]}")
                self.execute_action(best_move)
                ocr_list=self.get_ocr_list(best_move)
                self.game.set_board(board_state)
                print("*" * 10, f"回合{move_count}结束,耗时{time.time() - start_time:.2f}秒,总耗时{time.time() - start:.2f}秒","*" * 10)
                move_count += 1
# 使用示例
if __name__ == "__main__":
    automator = H52048Automation()
    automator.save_train_data=True
    automator.save_board=True
    #可选参数：ResNet,Template,And
    automator.pic_recognize_mode="Template"
    automator.ocr_mode="new"
    #日志保存间隔
    automator.save_move_interval=1
    #可选参数：all(匹配所有格子，所有数字)
    #new（只匹配可能新生成的2）得很慢，确保方块真的按照预期移动
    #prob（预测未来状态，再匹配）
    # 自动创建游戏文件夹并开始自动化
    automator.run()
