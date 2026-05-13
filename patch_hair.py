import cv2
import numpy as np
import random

def patch_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    start_idx = content.find('def add_hair_defect(')
    if start_idx == -1: return

    end_idx = content.find('def _generate_curved_hair_path(', start_idx)
    if end_idx == -1: return

    new_func = '''    def add_hair_defect(self, thickness, X, Y):
        \"\"\"添加头发丝遮挡缺陷 - 使用OpenCV确保连续绘制\"\"\"
        import cv2
        hair_thickness = thickness.copy()
        height, width = thickness.shape

        screen_length_x = getattr(self, 'screen_length_x', 0.030) if not hasattr(self, 'get_wavelength_meters') else self.screen_length_x
        screen_length_y = getattr(self, 'screen_length_y', 0.015) if not hasattr(self, 'get_wavelength_meters') else self.screen_length_y

        hair_topography = np.zeros((height, width))
        if not hasattr(self, 'hair_mask') or self.hair_mask is None:
            self.hair_mask = np.zeros((height, width), dtype=bool)

        for hair_index in range(self.hair_count):
            diameter_mm = random.uniform(0.04, 0.2)
            diameter_m = diameter_mm * 0.001
            radius_m = diameter_m / 2

            length_mm = random.uniform(5.0, 20.0)
            length_m = length_mm * 0.001

            is_newton = hasattr(self, 'get_R_meters')
            if is_newton:
                start_x = random.uniform(-screen_length_x / 2 + radius_m, screen_length_x / 2 - radius_m)
                start_y = random.uniform(-screen_length_y / 2 + radius_m, screen_length_y / 2 - radius_m)
            else:
                start_x = random.uniform(radius_m, screen_length_x - radius_m)
                start_y = random.uniform(radius_m, screen_length_y - radius_m)

            control_points = self._generate_curved_hair_path(
                start_x, start_y, length_m=length_m,
                screen_length_x=screen_length_x, screen_length_y=screen_length_y,
                min_radius=diameter_m / 2
            )

            curve_points = self._sample_bezier_curve(
                control_points, num_samples=max(50, int(length_m / 0.0001))
            )

            pts_px = []
            for px, py in curve_points:
                if is_newton:
                    qx = (px + screen_length_x / 2) / screen_length_x * (width - 1)
                    qy = (py + screen_length_y / 2) / screen_length_y * (height - 1)
                else:
                    qx = px / screen_length_x * (width - 1)
                    qy = py / screen_length_y * (height - 1)
                pts_px.append([int(qx), int(qy)])

            if len(pts_px) == 0: continue
            pts_arr = np.array(pts_px, dtype=np.int32).reshape((-1, 1, 2))
            
            mask_layer = np.zeros((height, width), dtype=np.uint8)
            px_size = screen_length_x / width
            thickness_px = max(2, int(diameter_m / px_size))
            cv2.polylines(mask_layer, [pts_arr], isClosed=False, color=1, thickness=thickness_px, lineType=cv2.LINE_AA)
            current_mask = mask_layer > 0
            
            if np.any(current_mask):
                hair_thickness[current_mask] = 1e-3
                self.hair_mask[current_mask] = True
                hair_topography[current_mask] = diameter_mm * random.uniform(0.9, 1.1)

        hair_thickness += hair_topography * 1e-6
        return hair_thickness

'''
    content = content[:start_idx] + new_func + content[end_idx:]
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Patched " + filepath)

patch_file('d:/2026物理竞赛/Physics/牛顿环瑕疵/parameters.py')
patch_file('d:/2026物理竞赛/Physics/劈尖瑕疵0329/parameters.py')
