import numpy as np
import cv2
import os
import json
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from datetime import datetime
from PIL import Image, ImageTk
import csv
import seaborn as sns
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Canvas, Scrollbar, Text
import threading
import glob
class IndependentColorAnalyzer:
    def __init__(self):
        """
        IndependentColorAnalyzer类用于独立的颜色分析。
        """
        self.standard_colors = {
            'yellow': {'bgr': np.array([0,255,255]), 'hsv': np.array([30,255,255]), 'lab': np.array([97.14,-21.56,94.48]), 'weight': 0.6},
            'blue':   {'bgr': np.array([255,180,50]), 'hsv': np.array([120,200,200]), 'lab': np.array([60.0,10.0,-45.0]), 'weight': 0.4}
        }
        self.analysis_params = {'s1_threshold':0.5,'s2_threshold':0.5,'diff_threshold':0.15,'blue_boost':1.3,'confidence_threshold':0.2}
        self._std_json_path = "standard_colors.json"
        try:
            if os.path.exists(self._std_json_path):
                self._load_standard_colors_from_json(self._std_json_path)
                print(f"[标准色] 已从 {self._std_json_path} 载入训练参数")
        except Exception as e:
            print(f"[标准色] 加载失败，继续使用内置默认：{e}")
        print("独立颜色分析器初始化完成")


    # ======== NEW: 标准色训练 / 读写 / ROI / 安全读图 ========
    def train_standard_colors(self, save_json_path=None):
        """
        训练标准颜色并保存到JSON文件。
        :param save_json_path: 保存JSON文件的路径，默认使用类属性 _std_json_path。
        """
        if save_json_path is None:
            save_json_path = self._std_json_path

        if os.path.exists(self._std_json_path):
            os.remove(self._std_json_path)
            
        mean= self._compute_folder_standard()
        self.standard_colors['yellow']['bgr'] = mean['BGR_Yellow']
        self.standard_colors['yellow']['hsv'] = mean['HSV_Yellow']      
        self.standard_colors['yellow']['lab'] = mean['LAB_Yellow']
        self.standard_colors['blue']['bgr']   = mean['BGR_Blue']
        self.standard_colors['blue']['hsv']   = mean['HSV_Blue']
        self.standard_colors['blue']['lab']   = mean['LAB_Blue']    
        yellow_std=self.standard_colors['yellow']
        blue_std=self.standard_colors['blue']
        try:
            with open(save_json_path,"w",encoding="utf-8") as f:
                json.dump(self.standard_colors,f,indent=2,ensure_ascii=False)
            print(f"[标准色] 训练完成并已保存：{save_json_path}")
            print(f"[标准色] 黄色(BGR/LAB/HSV): {yellow_std}")
            print(f"[标准色] 蓝色(BGR/LAB/HSV): {blue_std}")
        except Exception as e:
            print(f"[标准色] 保存JSON失败：{e}")
        return yellow_std,blue_std

    def _load_standard_colors_from_json(self, json_path):
        """
        从指定的 JSON 文件中加载标准颜色信息。
        """
        with open(json_path,"r",encoding="utf-8") as f:
            data=json.load(f)
        for key in ("yellow","blue"):
            self.standard_colors[key]['bgr']=np.array(data[key]["bgr"],dtype=float)
            print(self.standard_colors[key]['bgr'])
            self.standard_colors[key]['hsv']=np.array(data[key]["hsv"],dtype=float)
            print(self.standard_colors[key]['hsv'])
            self.standard_colors[key]['lab']=np.array(data[key]["lab"],dtype=float)
            print(self.standard_colors[key]['lab'])

    def split_train_val(self,folder,train_radio=0.9):
        """
        从指定文件夹中划分训练集和验证集。
        :param folder: 包含图像文件的文件夹路径。
        :param train_radio: 训练集占比，默认值为0.9。
        """
        file_list=[]
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith((".jpg", ".png", ".jpeg",".bmp")):
                    file_list.append(os.path.join(root, file))
        #按照比例划分训练集与验证集
        random.shuffle(file_list)
        train_file_list=file_list[:int(len(file_list)*train_radio)]
        val_file_list=file_list[int(len(file_list)*train_radio):]
        self.train_file_list_and_labels={file:1 if "yellow" in file else 0 for file in train_file_list}
        self.val_file_list_and_labels={file:1 if "yellow" in file else 0 for file in val_file_list}

    def _compute_folder_standard(self):
        """
        从训练集中计算黄色和蓝色的BGR、HSV、LAB均值。
        :return: 包含黄色和蓝色BGR、HSV、LAB均值的字典。
        """
        self.train_bgr_yellow_list, self.train_bgr_blue_list, self.train_lab_yellow_list, self.train_lab_blue_list, self.train_hsv_yellow_list, self.train_hsv_blue_list = [], [], [], [], [], []
        for file,num in self.train_file_list_and_labels.items():
            img=self._safe_imread(file)
            if img is None: continue
            processed_img=self.preprocess_image(img)
            if num==1:
                bgr_yellow, hsv_yellow, lab_yellow,_,_ = self._mean_colors(processed_img)
                self.train_bgr_yellow_list.append(bgr_yellow); self.train_lab_yellow_list.append(lab_yellow); self.train_hsv_yellow_list.append(hsv_yellow)
            else:
                bgr_blue, hsv_blue, lab_blue,_,_ = self._mean_colors(processed_img)
                self.train_bgr_blue_list.append(bgr_blue); self.train_lab_blue_list.append(lab_blue); self.train_hsv_blue_list.append(hsv_blue)
        self.val_bgr_yellow_list, self.val_bgr_blue_list, self.val_lab_yellow_list, self.val_lab_blue_list, self.val_hsv_yellow_list, self.val_hsv_blue_list = [], [], [], [], [], []
        # for file,num in self.val_file_list_and_labels.items():
        #     img=self._safe_imread(file)
        #     if img is None: continue
        #     if num==0:
        #         bgr_yellow, hsv_yellow, lab_yellow,_,_ = self._mean_colors(img)
        #         self.val_bgr_yellow_list.append(bgr_yellow); self.val_lab_yellow_list.append(lab_yellow); self.val_hsv_yellow_list.append(hsv_yellow)
        #     else:
        #         bgr_blue, hsv_blue, lab_blue,_,_ = self._mean_colors(img)
        #         self.val_bgr_blue_list.append(bgr_blue); self.val_lab_blue_list.append(lab_blue); self.val_hsv_blue_list.append(hsv_blue)
        # print(f"[标准色] 训练集数量：{len(self.train_file_list_and_labels)}")
        # print(f"[标准色] 验证集数量：{len(self.val_file_list_and_labels)}")
        # yellow_train_data=np.array(self.train_bgr_yellow_list+self.train_hsv_yellow_list+self.train_lab_yellow_list)
        # blue_train_data=np.array(self.train_bgr_blue_list+self.train_hsv_blue_list+self.train_lab_blue_list)
        # train_data=np.concatenate((yellow_train_data,blue_train_data),axis=0)
        # yellow_val_data=np.array(self.val_bgr_yellow_list+self.val_hsv_yellow_list+self.val_lab_yellow_list)
        # blue_val_data=np.array(self.val_bgr_blue_list+self.val_hsv_blue_list+self.val_lab_blue_list)
        # val_data=np.concatenate((yellow_val_data,blue_val_data),axis=0)
        # train_data_label=np.array([0]*len(yellow_train_data)+[1]*len(blue_train_data))
        # val_data_label=np.array([0]*len(yellow_val_data)+[1]*len(blue_val_data))
        # from sklearn import svm
        # clf=svm.SVC()
        # clf.fit(train_data,train_data_label)
        # val_pred=clf.predict(val_data)
        # from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
        # acc=accuracy_score(val_data_label,val_pred)
        # recall=recall_score(val_data_label,val_pred)
        # precision=precision_score(val_data_label,val_pred)
        # f1=f1_score(val_data_label,val_pred)
        # print(f"[标准色] 验证集准确率：{acc}")
        # print(f"[标准色] 验证集召回率：{recall}")
        # print(f"[标准色] 验证集精确率：{precision}")
        # print(f"[标准色] 验证集F1值：{f1}")
        # print(f"[标准色] 验证集混淆矩阵：\n{confusion_matrix(val_data_label,val_pred)}")



        self.plot_scatter_3d(np.array(self.train_bgr_yellow_list),np.array(self.train_bgr_blue_list),"rgb")
        self.plot_scatter_3d(np.array(self.train_hsv_yellow_list),np.array(self.train_hsv_blue_list),"hsv")
        self.plot_scatter_3d(np.array(self.train_lab_yellow_list),np.array(self.train_lab_blue_list),"lab")
        mean_bgr_yellow=np.mean(self.train_bgr_yellow_list,axis=0).tolist()
        mean_bgr_blue=np.mean(self.train_bgr_blue_list,axis=0).tolist()
        mean_lab_yellow=np.mean(self.train_lab_yellow_list,axis=0).tolist()
        mean_bgr_blue=np.mean(self.train_bgr_blue_list,axis=0).tolist()
        mean_lab_blue=np.mean(self.train_lab_blue_list,axis=0).tolist()
        mean_hsv_yellow=np.mean(self.train_hsv_yellow_list,axis=0).tolist()
        mean_hsv_blue=np.mean(self.train_hsv_blue_list,axis=0).tolist()
        return {"BGR_Yellow":mean_bgr_yellow,"LAB_Yellow":mean_lab_yellow,"HSV_Yellow":mean_hsv_yellow,
                "BGR_Blue":mean_bgr_blue,"LAB_Blue":mean_lab_blue,"HSV_Blue":mean_hsv_blue}
    
    def plot_scatter_3d(self,data_yellow,data_blue,mode):
        """
        绘制3D散点图，用于可视化数据分布。
        """
        # 创建3D图表
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        #绘制散点图
        ax.scatter(data_yellow[:, 0], data_yellow[:, 1], data_yellow[:, 2], c='y', marker='o', label='Yellow')
        ax.scatter(data_blue[:, 0], data_blue[:, 1], data_blue[:, 2], c='b', marker='^', label='Blue')
        # 设置轴标签
        # 设置轴标签
        if mode=="hsv":
            ax.set_xlabel('H Channel')
            ax.set_ylabel('S Channel')
            ax.set_zlabel('V Channel')
        elif mode=="rgb":
            ax.set_xlabel('R Channel')
            ax.set_ylabel('G Channel')
            ax.set_zlabel('B Channel')
        elif mode=="lab":
            ax.set_xlabel('L Channel')
            ax.set_ylabel('a Channel')
            ax.set_zlabel('b Channel')
        else:
            raise ValueError("Invalid mode. Choose from 'hsv', 'rgb', or 'lab'.")
        # 添加图例
        ax.legend()
        plt.savefig(f"{mode}分布.png")
        # 显示图表
        # plt.show()
    def preprocess_image(self,img):
        """
        对输入图像进行预处理，包括灰度化、高斯模糊、自适应阈值化。
        :param img: 输入图像，BGR格式。
        :return: 预处理后的图像。
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
  
    def _mean_colors(self, img):
        """
        计算图像中黄色和蓝色区域的BGR、HSV、LAB均值。
        :param img: 输入图像，BGR格式。
        :return: 包含黄色和蓝色BGR、HSV、LAB均值的元组。
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # 使用自适应阈值
        _,thresh=cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)# 按面积大小排序，从大到小
        if len(contours) == 0:
            raise ValueError("No contours found in image.")
        contours= contours[:1]  # 只保留最大的轮廓
        # cv2.drawContours(img,[np.array(contours).reshape(-1,1,2)],-1,(0,255,0),3)
        # cv2.imshow("contours",img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        black_points= np.array(contours).reshape(-1,2)
        if len(black_points) < 3:
            raise ValueError("Not enough contour points for circle fitting.")
        # 使用 RANSAC 拟合圆
        best_circle, best_inliers = self.circle_fit_ransac(black_points)
        center = best_circle[:2]
        radius = best_circle[2]
        rgb,hsv,lab,area=self.calculate_mean_rgb_hsv_lab(img,center,radius)
        return rgb,hsv,lab,center,radius
    
    def circle_fit_ransac(self,points, max_iterations=1000, distance_threshold=0.1):
        """
        使用 RANSAC 算法拟合圆。
        :param points: 输入点集，形状为 (N, 2) 的数组，其中 N 是点的数量。
        :param max_iterations: RANSAC 最大迭代次数，默认值为 1000。
        :param distance_threshold: 内点距离阈值，默认值为 0.1。
        :return: 最佳圆参数 (x0, y0, r) 和内点索引。
        """
        best_inliers = []
        best_circle = None
    
        for _ in range(max_iterations):
            # 随机选择三个点
            indices = np.random.choice(len(points), 3, replace=False)
            sample_points = points[indices]
    
            # 计算这三个点确定的圆的参数
            (x0, y0), r = self.fit_circle_point(sample_points)
    
            # 计算所有点到该圆的距离
            distances = np.sqrt((points[:, 0] - x0) ** 2 + (points[:, 1] - y0) ** 2) - r
            inliers = points[np.abs(distances) < distance_threshold]
    
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_circle = (x0, y0, r)
        
    
        return best_circle, best_inliers
 
    def fit_circle_point(self,points):
        """
        计算通过三个点确定的圆的参数。
        :param points: 输入点集，形状为 (3, 2) 的数组，其中包含三个点的坐标。
        :return: 圆的参数 (x0, y0, r)，其中 (x0, y0) 是圆心坐标，r 是半径。
        """
        A = np.zeros((len(points), 3))
        A[:, 0] = 2 * points[:, 0]
        A[:, 1] = 2 * points[:, 1]
        A[:, 2] = 1
        b = points[:, 0] ** 2 + points[:, 1] ** 2
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        x0 = x[0]
        y0 = x[1]
        r = np.sqrt(x0 ** 2 + y0 ** 2 + x[2])
        return (x0, y0), r
    
    def calculate_mean_rgb_hsv_lab(self,img,center,radius):
        """
        计算圆内区域的RGB、HSV、LAB均值。
        :param img: 输入图像，BGR格式。
        :param center: 圆的中心坐标 (x0, y0)。
        :param radius: 圆的半径。
        :return: 包含RGB、HSV、LAB均值的元组，以及圆内像素区域的面积。
        """
        # 提取圆内的像素
        mask = np.zeros(img.shape[:2],np.uint8)
        cv2.circle(mask, (int(center[0]), int(center[1])), int(radius), 255, -1)
        area=cv2.countNonZero(mask)
        rgb=cv2.mean(img, mask=mask)[:3]
        hsv=cv2.mean(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), mask=mask)[:3]
        lab=cv2.mean(cv2.cvtColor(img, cv2.COLOR_BGR2LAB), mask=mask)[:3]
        # cv2.imshow("mask",mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return rgb,hsv,lab,area

    def _safe_imread(self, path):
        """
        安全读取图像，支持中文路径。
        :param path: 图像文件路径。
        :return: 读取的图像，BGR格式。
        """
        try:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            return img
        except: return None
    def calculate_metric(self):
        """
        计算独立颜色识别的准确率、精确率、召回率和F1值。
        :return: 无返回值，打印准确率、精确率、召回率和F1值。
        """
        TP= TN = FP = FN = 0
        labels=[]
        predict_labels=[]
        for file,num in self.val_file_list_and_labels.items():
            img=self._safe_imread(file)
            if img is None: continue
            processed_img=self.preprocess_image(img)
            try:
                s1,s2,confidence,_,_,_=self.analyze_color_independent(processed_img)
                label="Yellow" if num==1 else "Blue"
                predict="Yellow" if s1 > s2 else "Blue"
                labels.append(num)
                predict_labels.append(1 if s1>s2 else 0)
                if label==predict=="Yellow":
                    TP+=1
                elif label==predict=="Blue":
                    TN+=1
                elif label=="Yellow" and predict=="Blue":
                    FN+=1
                elif label=="Blue" and predict=="Yellow":
                    FP+=1
                print(f"[独立] 文件: {os.path.basename(file)} | 真实: {label} |预测值：{predict}| S1: {s1:.3f} | S2: {s2:.3f} | 置信度: {confidence:.3f}")
            except Exception as e:
                print(f"[独立] 文件: {os.path.basename(file)} | 错误: {e}")
        accuracy=(TP+TN)/(TP+TN+FP+FN) if (TP+TN+FP+FN)>0 else 0
        precision=TP/(TP+FP) if (TP+FP)>0 else 0
        recall=TP/(TP+FN) if (TP+FN)>0 else 0
        f1=2*precision*recall/(precision+recall) if (precision+recall)>0 else 0
        cm=confusion_matrix(labels, predict_labels)
        class_names = ['Blue', 'Yellow']  # 类别名称
        # 可视化混淆矩阵
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig("confusion_matrix.png")
        # plt.show()
        print(f"[独立] 准确率: {accuracy:.3f} | 精确率: {precision:.3f} | 召回率: {recall:.3f} | F1值: {f1:.3f}")
        return accuracy,precision,recall,f1
    def analyze_color_independent(self, image_region):
        """
        计算依赖颜色识别的颜色分析结果。
        :param region_name: 区域名称，用于标识分析的区域。
        :param image_region: 输入图像区域，BGR格式。
        :return: 包含S1、S2、置信度、区域名称和颜色预测的元组。
        """
        try:
            if image_region is None or image_region.size == 0:
                return 0.5, 0.5, 0.0, "S0"
            s1_raw, s2_raw,center,radius = self._multi_space_analysis(image_region)
            confidence = abs(s1_raw - s2_raw)
            return s1_raw, s2_raw, confidence, "Yellow" if s1_raw > s2_raw else "Blue",center,radius
        except Exception as e:
            print(f"依赖分析错误: {e}")
            return 0.5, 0.5, 0.0, "S0"
    def analyze_single_color(self, image_path):
        image=cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        processed_image=self.preprocess_image(image)
        s1, s2, confidence, color,center,radius = self.analyze_color_independent(processed_image)
        # mask=np.zeros_like(image)
        result_image=processed_image.copy()
        if color=="Yellow":
            cv2.circle(result_image,(int(center[0]),int(center[1])),int(radius),(0,255,255),2)
            # image=cv2.addWeighted(image,0.8,mask,0.2,0)
            cv2.putText(result_image,f"{color}",(int(center[0]),int(center[1])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
        else:
            cv2.circle(result_image,(int(center[0]),int(center[1])),int(radius),(255,0,0),2)
            # image=cv2.addWeighted(image,0.8,mask,0.2,0)
            cv2.putText(result_image,f"{color}",(int(center[0]),int(center[1])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
        return result_image,processed_image
    def _multi_space_analysis(self, image_region):
        """
        计算多空间分析的S1和S2值。
        :param image_region: 输入图像区域，BGR格式。
        :return: 包含S1和S2值的元组。
        """
        try:
            s1_scores, s2_scores = [], []
            bgr_avg,hsv_avg,lab_avg,center,radius=self._mean_colors(image_region)
            s1_bgr = self._calculate_bgr_similarity(bgr_avg, self.standard_colors['yellow']['bgr'])
            s2_bgr = self._calculate_bgr_similarity(bgr_avg, self.standard_colors['blue']['bgr'])
            s1_scores.append(s1_bgr); s2_scores.append(s2_bgr)
            s1_hsv = self._calculate_hsv_similarity(hsv_avg, self.standard_colors['yellow']['hsv'])
            s2_hsv = self._calculate_hsv_similarity(hsv_avg, self.standard_colors['blue']['hsv'])
            s1_scores.append(s1_hsv); s2_scores.append(s2_hsv)
            s1_lab = self._calculate_lab_similarity(lab_avg, self.standard_colors['yellow']['lab'])
            s2_lab = self._calculate_lab_similarity(lab_avg, self.standard_colors['blue']['lab'])
            s1_scores.append(s1_lab); s2_scores.append(s2_lab)
            s1_final = np.average(s1_scores, weights=[0.4, 0.4, 0.2])
            s2_final = np.average(s2_scores, weights=[0.4, 0.4, 0.2])
            s1_final = max(0.0, min(1.0, s1_final))
            s2_final = max(0.0, min(1.0, s2_final))
            return s1_final, s2_final,center,radius
        except Exception as e:
            print(f"多空间分析错误: {e}")
            return 0.5, 0.5

    def _calculate_bgr_similarity(self, color1, color2):
        """
        计算BGR颜色空间的相似度。
        :param color1: 第一个颜色向量，BGR格式。
        :param color2: 第二个颜色向量，BGR格式。
        :return: 相似度值，范围在0到1之间。
        """
        try:
            d = np.linalg.norm(color1 - color2)
            return max(0.0, min(1.0, 1.0 - d / np.sqrt(3*255**2)))
        except: return 0.5

    def _calculate_hsv_similarity(self, color1, color2):
        """
        计算HSV颜色空间的相似度。
        """
        try:
            h1, s1, v1 = color1; h2, s2, v2 = color2
            dh = min(abs(h1-h2), 180-abs(h1-h2)) / 180.0
            h_sim = 1.0 - dh
            s_sim = 1.0 - abs(s1-s2)/255.0
            v_sim = 1.0 - abs(v1-v2)/255.0
            weights = [0.6,0.25,0.15] if 100<=h2<=130 else [0.5,0.3,0.2]
            return max(0.0, min(1.0, weights[0]*h_sim + weights[1]*s_sim + weights[2]*v_sim))
        except: return 0.5

    def _calculate_lab_similarity(self, color1, color2):
        """
        计算LAB颜色空间的相似度。
        :param color1: 第一个颜色向量，LAB格式。
        :param color2: 第二个颜色向量，LAB格式。
        :return: 相似度值，范围在0到1之间。
        """
        try:
            d = np.linalg.norm(color1 - color2)
            return max(0.0, min(1.0, 1.0 - d/200.0))#被调整为了0-255为什么这里二范数是200？
        except: return 0.5
    def batch_process_single(self,image_paths,progress):
        try:
            results=[]
            self.image_path=os.path.dirname(image_paths[0])
            total=len(image_paths)
            for i,file in enumerate(image_paths):
                img=self._safe_imread(file)
                if img is None: continue
                processed_img=self.preprocess_image(img)
                s1_raw, s2_raw,center,radius = self._multi_space_analysis(processed_img)
                if s1_raw>s2_raw:
                    color="Yellow"
                else:
                    color="Blue"
                confidence=abs(s1_raw-s2_raw)
                results.append({"file": os.path.basename(file),"s1":s1_raw,"s2":s2_raw,"confidence":confidence,"color":color})
                # 更新进度条
                try:
                    if progress is not None:
                        progress['value'] = int((i + 1) / max(1, total) * 100)
                        # 允许UI刷新
                        try:
                            progress.update_idletasks()
                        except: pass
                except Exception as e:
                        print(f"Error updating progress for image {i}: {e}") 
        except Exception as e:
            print(f"批量处理错误: {e}")
            return None
        return results
    def save_process_single_circle_to_csv(self,results):
        with open(os.path.join(self.image_path,'单圆图像独立分析结果.csv'), mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['File Name', 'S1', 'S2','Confidence','Color'])
            for row in results:
                writer.writerow([row['file'], row['s1'], row['s2'],row['confidence'],row['color']])

# ==================== 高精度QR码检测器 ====================
class HighPrecisionQRDetector:
    def __init__(self):
        self.color_analyzer = IndependentColorAnalyzer()
        

    def detect_reaction_regions(self, image):
        """
        检测图像中的区域。
        :param image: 输入图像。
        :return: 检测到的区域列表，每个区域包含名称和图像。
        """
        
        warped_image = self._precision_qr_detection(image)
        warped_image=self.color_analyzer.preprocess_image(warped_image)
        # cv2.imshow("image",warped_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        regions = self._precision_region_localization(warped_image)
        return warped_image,regions
    
    def analyze_reaction_regions(self, processed_image,regions):
        """
        分析检测到的区域。
        :param regions: 检测到的区域列表，每个区域包含名称和图像。
        :return: 分析结果字典，每个区域包含s1、s2、置信度、结果和颜色。
        """
        image=processed_image.copy()
        try:
            results = {}

            for region in regions:
                name=region["name"]
                x1,y1,x2,y2=region["position"]
                s1, s2, confidence, color,center,radius = self.color_analyzer.analyze_color_independent(region["image"])
                # mask=np.zeros_like(image)
                if color=="Yellow":
                    cv2.circle(image,(int(x1+center[0]),int(y1+center[1])),int(radius),(0,255,255),2)
                    # image=cv2.addWeighted(image,0.8,mask,0.2,0)
                    cv2.putText(image,f"{name}: {color}",(int(x1+center[0]),int(y1+center[1])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
                else:
                    cv2.circle(image,(int(x1+center[0]),int(y1+center[1])),int(radius),(255,0,0),2)
                    # image=cv2.addWeighted(image,0.8,mask,0.2,0)
                    cv2.putText(image,f"{name}: {color}",(int(x1+center[0]),int(y1+center[1])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
                # cv2.imshow("image",image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                if name=="Control":
                    if s1>s2 and s1>0.5:
                        logic = "S1>S2且S1>0.5→有效"
                    elif s2>s1 and s2>0.5:
                        logic = "S2>S1且S2>0.5→阴性"
                else:
                    if s1 > s2 and s1 > 0.5: logic = "S1>S2且S1>0.5→阳性"
                    elif s2 > s1 and s2 > 0.5: logic = "S2>S1且S2>0.5→阴性"
                    else:
                        logic = "分数接近或低于阈值→需验证"
                print(f"{name}: s1={s1:.3f}, s2={s2:.3f}, 置信度={confidence:.3f}, 区域名称={name}, 预测颜色={color},逻辑={logic}")
                results[name] = {"s1": s1, "s2": s2, "confidence": confidence, "region_name": name,"predict color": color,"logic":logic}
            return results,image
        except Exception as e:
            print(f"区域分析错误: {e}")
            return {}
    
    def _precision_qr_detection(self, image):
        """
        高精度QR码检测。
        :param image: 输入图像。
        :return: 二维码区域的透视变换图像。
        """
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        qr_detector = cv2.QRCodeDetector()
        retval, points, _ = qr_detector.detectAndDecode(gray)
        if points is not None:
            dst=np.array([[0,0],[900,0],[900,900],[0,900]],dtype=np.float32)
            M = cv2.getPerspectiveTransform(points.reshape(-1,2), dst)
            warped = cv2.warpPerspective(image, M, (900, 900))
        else:
            x1,y1,x3,y3=self._contour_based_detection(image)
            dst=np.array([[0,0],[900,0],[900,900],[0,900]],dtype=np.float32)
            src=np.array([[x1,y1],[x3,y1],[x3,y3],[x1,y3]],dtype=np.float32)
            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(image, M, (900, 900))
        return warped
    
    def _contour_based_detection(self, image):
        """
        基于轮廓检测的高精度QR码定位。
        :param image: 输入图像。
        :return: 二维码区域的四个角点坐标。
        """
        gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #轮廓最像正方形
        square_contours=[]
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour.reshape(-1, 2))
            aspect_ratio = w / float(h)
            if aspect_ratio > 0.9 and aspect_ratio < 1.1:
                square_contours.append(contour)
                # cv2.drawContours(image, [contour], -1, (0, 255, 255), 15)
        #找到最大的三个正方形轮廓
        square_contours=sorted(square_contours, key=lambda x: cv2.contourArea(x), reverse=True)[:3]
        # cv2.drawContours(image, square_contours, -1, (0, 255, 255), 15)
        square_contours=sorted(square_contours, key=lambda x: cv2.boundingRect(x)[0]+cv2.boundingRect(x)[1])
        x1,y1,w1,h1=cv2.boundingRect(square_contours[1])
        x2,y2,w2,h2=cv2.boundingRect(square_contours[2])
        if y1>y2:#左下角,右上角
            square_contours[1],square_contours[2]=square_contours[2],square_contours[1]
        for num,contour in enumerate(square_contours):
            if num==0:
                cv2.drawContours(image, [contour], -1, (0, 255, 0), 15)#绿色
                x1,y1,w1,h1=cv2.boundingRect(contour)
            elif num==1:
                cv2.drawContours(image, [contour], -1, (0, 0, 255), 15)#红色
                x2,y2,w2,h2=cv2.boundingRect(contour)
                x2=x2+w2
            else:
                cv2.drawContours(image, [contour], -1, (255, 0, 0), 15)#蓝色
                x4,y4,w4,h4=cv2.boundingRect(contour)
                y4=y4+h4
        x3,y3=x2,y4
        return x1,y1,x3,y3
    def _precision_region_localization(self, qr_image):
        """
        高精度区域定位。
        :param qr_image: 输入二维码图像。
        :return: 定位到的区域列表，每个区域包含名称、图像、中心坐标和位置。
        """
        h, w = qr_image.shape[:2]
        center_x, center_y = h//2, w//2
        regions_config = [
            {"name": "H1N1", "offset": (-0.26, 0.0), "radius": 0.1},
            {"name": "Rhinovirus", "offset": (-0.12, 0.23), "radius": 0.1},
            {"name": "Control", "offset": (0, -0.26), "radius": 0.1},
            {"name": "Flu B", "offset": (0.14, 0.23), "radius": 0.1},
            {"name": "RSV", "offset": (0.28, 0), "radius": 0.1}
        ]
        regions = []
        for config in regions_config:
            regions.append(self._extract_precision_region(qr_image, center_x, center_y, config))
        return regions

    def _extract_precision_region(self, qr_image, center_x, center_y, config):
        """
        提取高精度区域。
        :param qr_image: 输入二维码图像。
        :param center_x: 二维码中心的x坐标。
        :param center_y: 二维码中心的y坐标。
        :param config: 区域配置，包含名称、偏移量和半径。
        :return: 提取到的区域字典，包含名称、图像、中心坐标和位置。
        """
        h, w = qr_image.shape[:2]; dx, dy = config["offset"]; radius_px = int(min(h, w) * config["radius"])
        x = int(center_x + dx * w); y = int(center_y + dy * h)
        x1 = max(0, x - radius_px); y1 = max(0, y - radius_px)
        x2 = min(w, x + radius_px); y2 = min(h, y + radius_px)
        if x2 > x1 and y2 > y1 and (x2 - x1) > 5 and (y2 - y1) > 5:
            region_image = qr_image[y1:y2, x1:x2]
            cv2.rectangle(qr_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.imshow(config["name"],qr_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            return {"name": config["name"], "image": region_image, "center": (x, y), "position": (x1, y1, x2, y2)}
        
    def comprehensive_validation(self, image_path, progress=None):
        """
        分析文件夹下的所有图片（不同色温、距离、角度），生成验证报告
        """
        validation_results = []
        self.image_path=image_path
        self.csv_path=os.path.join(self.image_path, "独立分析验证.csv")
        self.last_folder_name=os.path.basename(image_path)
        try:
            temperature, distance, angle,device = self.last_folder_name.split("_")
        except ValueError:
            print(f"Error: Invalid folder name format '{self.last_folder_name}'")
            return 0.0
        
        images_paths = [p for p in os.listdir(self.image_path) if p.endswith((".jpg", ".png", ".bmp"))]
        total = len(images_paths)
        for idx, image_path in enumerate(images_paths):
            try:
                image = cv2.imdecode(np.fromfile(os.path.join(self.image_path, image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
                # image=self.color_analyzer.preprocess_image(image)
                warped_image, regions = self.detect_reaction_regions(image)
                results,_ = self.analyze_reaction_regions(warped_image, regions)
                validation_results.append({
                    "image_path": image_path,
                    "conditions": f"{temperature}K_{distance}cm_{angle}deg_{device}",
                    "temperature_k": float(temperature), "distance_cm": float(distance), "angle_deg": float(angle),
                    "results": results
                })
                # 更新进度条
                try:
                    if progress is not None:
                        progress['value'] = int((idx + 1) / max(1, total) * 100)
                        # 允许UI刷新
                        try:
                            progress.update_idletasks()
                        except: pass
                except Exception as e:
                        print(f"Error updating progress for image {image_path}: {e}")
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
        
        result = self._calculate_success_rate_and_report(validation_results)
        return result,total

    def _calculate_success_rate_and_report(self, validation_results):
        """
        读取csv的数据用于验证，计算准确率，并将结果写入现有的CSV文件
        """
        tp, fp, fn, tn = 0, 0, 0, 0
        df = pd.read_csv(self.csv_path)
        list_type = ['Control', 'H1N1', 'Rhinovirus', 'Flu B', 'RSV']
        
        # 添加新的列
        for key in list_type:
            df[f"{key} predict"] = None

        for validation_result in validation_results:
            # 在df中找到对应的图片路径的行
            normalized_target_path = validation_result["image_path"]
            row = df[df["image path"] == normalized_target_path]
            
            if row.empty:
                print(f"Warning: No matching row found for image {normalized_target_path} in CSV.")
                continue
            
            for key, value in validation_result["results"].items():
                index = list_type.index(key)
                true_label = row.iloc[0, index + 2]
                # print(f"true_label type: {type(true_label)}")
                predict_label = value["predict color"]
                if predict_label=="Blue":#Blue为0
                    predict_label=0
                else:#Yellow为1
                    predict_label=1
                if predict_label == true_label == 1:
                    tp += 1
                elif predict_label != true_label and predict_label == 1:
                    fp += 1
                elif predict_label != true_label and predict_label == 0:
                    fn += 1
                else:
                    tn += 1
                
                # 更新CSV文件中的对应行
                df.loc[df["image path"] == normalized_target_path, f"{key} predict"] = predict_label

        # 计算成功率
        success_rate = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision=tp/(tp+fp) if (tp+fp)>0 else 0
        recall=tp/(tp+fn) if (tp+fn)>0 else 0
        f1=2*precision*recall/(precision+recall) if (precision+recall)>0 else 0
        result=f"True Positive: {tp}, False Positive: {fp}, False Negative: {fn}, True Negative: {tn}"+"\n"+f"Success Rate: {success_rate:.4f},Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}"
        # cm=confusion_matrix(df.iloc[:,8:14].values.flatten(), df.iloc[:,2:8].values.flatten())
        # class_names = ['Yellow', 'Blue']  # 类别名称
        # # 可视化混淆矩阵
        # plt.figure(figsize=(8, 6))
        # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        # plt.xlabel('Predicted')
        # plt.ylabel('True')
        # plt.title('Confusion Matrix')
        # plt.savefig(os.path.join(self.image_path, "confusion_matrix.png"))
        # plt.show()
        # 保存更新后的CSV文件
        df.to_csv(self.csv_path, index=False, encoding="utf-8-sig")

        return result
        

    def image_path_to_csv(self,image_path):
        """
        读取图片路径,将图片路径写到csv文件当中，对应的图片路径的数据进行分析
        """
        def get_last_folder_name(path):
            return os.path.basename(os.path.normpath(path))
        self.last_folder_name=get_last_folder_name(image_path)
        self.image_path=image_path
        image_paths=[p for p in os.listdir(image_path) if p.endswith(".jpg") or p.endswith(".png") or p.endswith(".bmp")]
        self.csv_path=os.path.join(image_path, "独立分析验证.csv")
        with open(self.csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["condition","image path","Control label","H1N1 label","Rhinovirus label","Flu B label","RSV label",
                             "Control predict","H1N1 predict","Rhinovirus predict","Flu B predict","RSV predict"])
            for i in image_paths:
                writer.writerow([self.last_folder_name,i])
    
    def calculate_results(self,directory,progress=None):
        """
        计算结果，并将结果写入现有的CSV文件
        """
        validation_results = []
        self.image_path=directory
        images_paths = [os.path.join(self.image_path, p) for p in os.listdir(self.image_path) if p.endswith((".jpg", ".png", ".bmp"))]
        total = len(images_paths)
        for idx, image_path in enumerate(images_paths):
            try:
                image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                # image=self.color_analyzer.preprocess_image(image)
                warped_image, regions = self.detect_reaction_regions(image)
                results,_ = self.analyze_reaction_regions(warped_image, regions)
                validation_results.append({
                    "image_path": image_path,
                    "results": results
                })
                # 更新进度条
                try:
                    if progress is not None:
                        progress['value'] = int((idx + 1) / max(1, total) * 100)
                        try:
                            progress.update_idletasks()
                        except Exception as e: 
                            print(f"Error updating progress for image {image_path}: {e}")
                except Exception as e:
                        print(f"Error updating progress for image {image_path}: {e}")
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
        return validation_results
    def calculate_result_and_image(self,image_path):
        """
        从指定的图片路径开始分析，返回分析结果和图片
        """
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        warped_image, regions = self.detect_reaction_regions(image)
        # warped_image=self.color_analyzer.preprocess_image(warped_image)
        result,image = self.analyze_reaction_regions(warped_image, regions)
        return result,image,warped_image
    




# ==================== 独立分析GUI界面 ====================
class IndependentPathogenApp:
    def __init__(self, root):
        self.root = root
        self.root.title("病原检测彩色二维码分析软件 - （含标准色训练）")
        self.root.geometry("1800x900")# 窗口大小
        self.root.configure(bg='#2c3e50')# 背景颜色

        self.analyzer = IndependentColorAnalyzer()
        self.QRDetector= HighPrecisionQRDetector()
        self.current_results = None
        self.current_image_path = None
        self.result=None
        self.setup_ui()

    def setup_ui(self):
        """
        初始化用户界面布局
        """
        title_frame = tk.Frame(self.root, bg='#34495e', height=80)# 标题栏
        title_frame.pack(fill=tk.X, padx=10, pady=10); title_frame.pack_propagate(False)# 标题栏占满X轴
        title_label = tk.Label(title_frame, text="病原检测彩色二维码分析软件",
                               font=("微软雅黑", 18, "bold"), bg='#34495e', fg='white')# 标题栏字体
        title_label.pack(expand=True)# 标题栏字体居中
        model_info = tk.Label(title_frame, text="S1/S2评分算法",
                              font=("微软雅黑", 10), bg='#34495e', fg='#00ff00') # 模型信息字体
        model_info.pack(side=tk.BOTTOM, pady=5)
        main_frame = tk.Frame(self.root, bg='#2c3e50'); main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        control_frame = tk.LabelFrame(main_frame, text="独立分析控制面板", font=("微软雅黑", 12), bg='#34495e', fg='white', width=320)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10)); control_frame.pack_propagate(False)
        self.setup_control_panel(control_frame)
        display_frame = tk.Frame(main_frame, bg='#2c3e50'); display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.setup_display_panel(display_frame)

    def setup_control_panel(self, parent):
        """
        设置控制面板上的按钮和状态栏
        """
        button_frame = tk.Frame(parent, bg='#34495e'); button_frame.pack(fill=tk.X, padx=10, pady=10)
        buttons = [
            ("训练标准色", self._train_standard_colors_ui),
            ("选择独立分析图像", self.load_image),
            ("输出独立分析图像",self.output_independent_image),
            ("选择单圆图像", self.load_single_image),
            ("输出单圆图像",self.output_single_image),
            ("独立分析", self.independent_analyze),
            ("保存独立分析结果", self.save_independent_result),
            ("选择用于系统验证的文件夹",self.load_folder),
            ("系统验证", self.comprehensive_validation),
            ("批量处理独立分析图像并保存结果", self.batch_process),
            ("批量处理单圆图像并保存结果", self.batch_process_single),

        ]
        for text, command in buttons:
            btn = tk.Button(button_frame, text=text, command=command, font=("微软雅黑", 10), bg='#3498db', fg='white', relief=tk.RAISED, bd=2, height=2)
            btn.pack(fill=tk.X, pady=2)

        status_frame = tk.LabelFrame(parent, text="系统状态", font=("微软雅黑", 10), bg='#34495e', fg='white')
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        self.status_var = tk.StringVar(value="就绪")
        tk.Label(status_frame, textvariable=self.status_var, font=("微软雅黑", 9), bg='#34495e', fg='#00ff00').pack()
        self.progress = ttk.Progressbar(status_frame, mode='determinate'); self.progress.pack(fill=tk.X, pady=5)

        info_frame = tk.LabelFrame(parent, text="独立分析算法说明", font=("微软雅黑", 10), bg='#34495e', fg='white')
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        info_text = """核心特性:
        • 支持从黄/蓝文件夹训练标准色并自动应用
        • 支持单个圆分析
        • 支持独立分析，独立分析表格展示
        • 支持系统验证，系统验证数据演示
        • 支持批量处理，批量处理结果展示
        • 支持结果保存，结果保存为csv文件
        """
        tk.Label(info_frame, text=info_text, font=("微软雅黑", 8), bg='#34495e', fg='white', justify=tk.LEFT).pack()

    def setup_display_panel(self, parent):
        """
        显示图像显示和独立分析结果的标签页
        """
        self.notebook = ttk.Notebook(parent); self.notebook.pack(fill=tk.BOTH, expand=True)
        image_tab = ttk.Frame(self.notebook); self.notebook.add(image_tab, text="图像显示"); self.setup_image_tab(image_tab)
        result_tab = ttk.Frame(self.notebook); self.notebook.add(result_tab, text="独立分析结果"); self.setup_results_tab(result_tab)
        analysis_tab = ttk.Frame(self.notebook); self.notebook.add(analysis_tab, text="性能分析"); self.setup_analysis_tab(analysis_tab)

    def setup_image_tab(self, parent):
        """
        显示原图像和识别后的图像
        """
        # 使用 2x3 网格布局：
        # 第一行：独立分析 原图（orig_tl），独立分析 结果图（orig_tr），独立分析 附加图（orig_tm）
        # 第二行：单圆原图（single_bl），单圆结果（single_br），单圆 附加图（single_bm）
        grid_frame = tk.Frame(parent)
        grid_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        # 第一行 布局
        top_frame = tk.Frame(grid_frame)
        top_frame.pack(fill=tk.BOTH, expand=True)

        orig_tl_frame = tk.LabelFrame(top_frame, text="独立分析 原图", font=("微软雅黑", 10))
        orig_tl_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,5), pady=5)
        self.orig_canvas_tl = Canvas(orig_tl_frame, bg='black', highlightthickness=0)
        self.orig_canvas_tl.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        orig_tm_frame = tk.LabelFrame(top_frame, text="独立分析 增强图像", font=("微软雅黑", 10))
        orig_tm_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5,0), pady=5)
        self.orig_canvas_tm = Canvas(orig_tm_frame, bg='black', highlightthickness=0)
        self.orig_canvas_tm.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        orig_tr_frame = tk.LabelFrame(top_frame, text="独立分析 结果图", font=("微软雅黑", 10))
        orig_tr_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5,5), pady=5)
        self.orig_canvas_tr = Canvas(orig_tr_frame, bg='black', highlightthickness=0)
        self.orig_canvas_tr.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        

        # 第二行 布局
        bottom_frame = tk.Frame(grid_frame)
        bottom_frame.pack(fill=tk.BOTH, expand=True)

        single_bl_frame = tk.LabelFrame(bottom_frame, text="单圆原图", font=("微软雅黑", 10))
        single_bl_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,5), pady=5)
        self.single_canvas_bl = Canvas(single_bl_frame, bg='black', highlightthickness=0)
        self.single_canvas_bl.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        single_bm_frame = tk.LabelFrame(bottom_frame, text="单圆 增强图像", font=("微软雅黑", 10))
        single_bm_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5,0), pady=5)
        self.single_canvas_bm = Canvas(single_bm_frame, bg='black', highlightthickness=0)
        self.single_canvas_bm.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        single_br_frame = tk.LabelFrame(bottom_frame, text="单圆结果", font=("微软雅黑", 10))
        single_br_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5,5), pady=5)
        self.single_canvas_br = Canvas(single_br_frame, bg='black', highlightthickness=0)
        self.single_canvas_br.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        


    def setup_results_tab(self, parent):
        """
        显示独立分析的表格
        """
        table_frame = tk.Frame(parent, bg='#34495e'); table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        columns = ["检测项","结果","置信度","S1","S2","判断逻辑"]
        for i, col in enumerate(columns):
            tk.Label(table_frame, text=col, font=("微软雅黑", 9, "bold"), bg='#2980b9', fg='white', width=10, relief=tk.RAISED).grid(row=0, column=i, sticky="nsew", padx=1, pady=1)
        self.result_labels = []
        for row in range(5):
            row_labels = []
            for col in range(6):
                label = tk.Label(table_frame, text="-", font=("微软雅黑", 8), bg='white', fg='black', width=10, relief=tk.GROOVE)
                label.grid(row=row + 1, column=col, sticky="nsew", padx=1, pady=1); row_labels.append(label)
            self.result_labels.append(row_labels)
        for i in range(6): table_frame.columnconfigure(i, weight=1)
        for i in range(6): table_frame.rowconfigure(i, weight=1)

    def setup_analysis_tab(self, parent):
        """
        显示独立分析的性能分析结果
        """
        analysis_frame = tk.Frame(parent, bg='white'); analysis_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar = Scrollbar(analysis_frame); scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.analysis_text = Text(analysis_frame, font=("微软雅黑", 10), wrap=tk.WORD, yscrollcommand=scrollbar.set); self.analysis_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.analysis_text.yview)
        self.analysis_text.insert(tk.END, "独立分析性能结果将在这里显示...\n\n"); self.analysis_text.config(state=tk.DISABLED)

    def _train_standard_colors_ui(self):
        """
        训练标准色并更新标准色值
        """
        messagebox.showinfo("提示", "请先选择包含【yellow】和【blue】名称的图片文件夹，用于训练标准色。")
        yellow_and_blue = filedialog.askdirectory(title="选择【黄色】和【蓝色】图片文件夹")
        if not yellow_and_blue: return
        try:
            self.analyzer.split_train_val(yellow_and_blue,train_radio=0.9)
            yellow_std,blue_std=self.analyzer.train_standard_colors()
            accuracy,precision,recall,f1=self.analyzer.calculate_metric()
            messagebox.showinfo("训练完成", f"标准色已更新并保存；之后识别将使用新标准值。\n\n黄色: {yellow_std}\n蓝色: {blue_std}\n验证集指标\n准确率: {accuracy:.3f}\n精确率: {precision:.3f}\n召回率: {recall:.3f}\nF1值: {f1:.3f}")
        except Exception as e:
            messagebox.showerror("训练失败", str(e))

    def load_image(self):
        """
        加载用户选择的二维码图像文件
        """
        file_path = filedialog.askopenfilename(title="选择二维码图像", filetypes=[("图像文件","*.jpg *.jpeg *.png *.bmp *.tiff")])
        if file_path:
            self.current_image_path = file_path
            self.status_var.set(f"已加载: {os.path.basename(file_path)}")# 更新状态栏显示当前加载的图像文件名
            self.display_original_image(file_path,"up left")

    def load_single_image(self):
        file_path = filedialog.askopenfilename(title="选择单圆图像", filetypes=[("图像文件","*.jpg *.jpeg *.png *.bmp *.tiff")])
        if file_path:
            self.current_single_image_path = file_path
            self.status_var.set(f"已加载: {os.path.basename(file_path)}")# 更新状态栏显示当前加载的图像文件名
            self.display_original_image(file_path,"down left")

    def display_original_image(self, image_path, location):
        """
        显示用户选择的原始图像
        """
        # 支持三种输入类型：文件路径 (str)、PIL.Image.Image、numpy.ndarray (BGR或灰度)
        try:
            if isinstance(image_path, str):
                pil_image = Image.open(image_path)
            elif isinstance(image_path, Image.Image):
                pil_image = image_path
            elif isinstance(image_path, np.ndarray):
                arr = image_path
                if arr.ndim == 3 and arr.shape[2] == 3:
                    pil_image = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
                else:
                    pil_image = Image.fromarray(arr)
            else:
                # 最后尝试使用 PIL 打开传入的对象
                pil_image = Image.open(image_path)

            img_copy = pil_image.copy(); img_copy.thumbnail((400, 400))

            if location == "up left":
                self.orig_photo_tl = ImageTk.PhotoImage(img_copy)
                self.orig_canvas_tl.delete("all")
                w = self.orig_canvas_tl.winfo_width(); h = self.orig_canvas_tl.winfo_height()
                if w > 1 and h > 1:
                    self.orig_canvas_tl.create_image(w//2, h//2, image=self.orig_photo_tl, anchor=tk.CENTER)
            elif location == "up right":
                self.orig_photo_tr = ImageTk.PhotoImage(img_copy)
                self.orig_canvas_tr.delete("all")
                w = self.orig_canvas_tr.winfo_width(); h = self.orig_canvas_tr.winfo_height()
                if w > 1 and h > 1:
                    self.orig_canvas_tr.create_image(w//2, h//2, image=self.orig_photo_tr, anchor=tk.CENTER)
            elif location == "down left":
                self.single_photo_bl = ImageTk.PhotoImage(img_copy)
                self.single_canvas_bl.delete("all")
                w = self.single_canvas_bl.winfo_width(); h = self.single_canvas_bl.winfo_height()
                if w > 1 and h > 1:
                    self.single_canvas_bl.create_image(w//2, h//2, image=self.single_photo_bl, anchor=tk.CENTER)
            elif location == "down right":
                self.single_photo_br = ImageTk.PhotoImage(img_copy)
                self.single_canvas_br.delete("all")
                w = self.single_canvas_br.winfo_width(); h = self.single_canvas_br.winfo_height()
                if w > 1 and h > 1:
                    self.single_canvas_br.create_image(w//2, h//2, image=self.single_photo_br, anchor=tk.CENTER)
            elif location == "up middle":
                self.orig_photo_tm = ImageTk.PhotoImage(img_copy)
                self.orig_canvas_tm.delete("all")
                w = self.orig_canvas_tm.winfo_width(); h = self.orig_canvas_tm.winfo_height()
                if w > 1 and h > 1:
                    self.orig_canvas_tm.create_image(w//2, h//2, image=self.orig_photo_tm, anchor=tk.CENTER)
            elif location == "down middle":
                self.single_photo_bm = ImageTk.PhotoImage(img_copy)
                self.single_canvas_bm.delete("all")
                w = self.single_canvas_bm.winfo_width(); h = self.single_canvas_bm.winfo_height()
                if w > 1 and h > 1:
                    self.single_canvas_bm.create_image(w//2, h//2, image=self.single_photo_bm, anchor=tk.CENTER)
            else:
                raise ValueError(f"未知的位置: {location}")
        except Exception as e:
            # 把异常展示出来，便于定位问题
            messagebox.showerror("错误", f"显示图像失败: {str(e)}")

    def output_independent_image(self):
        result,image,processed_image=self.QRDetector.calculate_result_and_image(self.current_image_path)
        self.result=result
        self.display_original_image(processed_image,"up middle")
        self.display_original_image(image,"up right")
    

    def output_single_image(self):
        image,processed_image=self.analyzer.analyze_single_color(self.current_single_image_path)
        self.display_original_image(processed_image,"down middle")
        self.display_original_image(image,"down right")

    def independent_analyze(self):
        """
        独立分析用户选择的二维码图像
        """
        if not hasattr(self, 'current_image_path') or not self.current_image_path:
            messagebox.showwarning("警告", "请先选择图像文件")
            return
        if not self.result:
            messagebox.showwarning("警告", "请先进行分析")
            return
        self.root.after(0, self.display_independent_results, self.result)
        self.status_var.set("独立分析完成")

    def save_independent_result(self):
        """
        保存当前独立分析结果到CSV文件
        """
        if self.result is None:
            messagebox.showwarning("警告", "没有可保存的结果"); return
        filename = filedialog.asksaveasfilename(title="保存独立分析结果", defaultextension=".csv", filetypes=[("CSV文件","*.csv")])
        if filename:
            try:
                with open(filename, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f)
                    writer.writerow(["检测项","结果","置信度","S1","S2","判断逻辑"])
                    for key,value in self.result.items():
                        writer.writerow([key,value["predict color"],value["confidence"],value["s1"],value["s2"],value["logic"]])
                messagebox.showinfo("成功", f"独立分析结果已保存到: {filename}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {str(e)}")

    def display_independent_results(self, result):
        """
        在独立分析结果框架中显示独立分析结果
        """
        for i, (_,value) in enumerate(result.items()):
            if i < len(self.result_labels):
                labels = self.result_labels[i]
                labels[0].config(text=value["region_name"]); 
                if value["predict color"] == "Yellow": 
                    value["predict color"] = "阳性"
                elif value["predict color"] == "Blue": 
                    value["predict color"] = "阴性"

                labels[1].config(text=value["predict color"]); 
                labels[2].config(text=value["confidence"]); 
                labels[3].config(text=value["s1"]); 
                labels[4].config(text=value["s2"])
                labels[5].config(text=value["logic"]); 
                result_text = value["predict color"]
                if result_text =="阳性": labels[1].config(bg="#fff200")
                elif result_text == "阴性": labels[1].config(bg="#0015ff")
                else:
                    raise ValueError(f"未知的预测颜色: {result_text}")
        self.update_independent_analysis(result)

    def setup_analysis_tab(self, parent):
        """
        初始化独立分析性能分析界面
        """
        analysis_frame = tk.Frame(parent, bg='white'); analysis_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar = Scrollbar(analysis_frame); scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.analysis_text = Text(analysis_frame, font=("微软雅黑", 10), wrap=tk.WORD, yscrollcommand=scrollbar.set)
        self.analysis_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.analysis_text.yview)
        self.analysis_text.insert(tk.END, "独立分析性能结果将在这里显示...\n\n")
        self.analysis_text.config(state=tk.DISABLED)

    def update_independent_analysis(self, result):
        """
        更新独立分析性能分析结果
        """
        self.analysis_text.config(state=tk.NORMAL); self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(tk.END, "=== 独立分析性能分析 ===\n\n")
        s1_values = [float(res["s1"]) for res in result.values()]
        s2_values = [float(res["s2"]) for res in result.values()]
        self.analysis_text.insert(tk.END, "分数分布分析:\n")
        self.analysis_text.insert(tk.END, f"平均S1: {np.mean(s1_values):.3f}\n")
        self.analysis_text.insert(tk.END, f"平均S2: {np.mean(s2_values):.3f}\n")

        self.analysis_text.insert(tk.END, "===置信度分析===:\n")
        confidences = [float(res["confidence"]) for res in result.values()]
        s1_count = sum(1 for d in confidences if d >= 0.3)
        s2_count = sum(1 for d in confidences if 0.15 <= d < 0.3)
        s0_count = sum(1 for d in confidences if d < 0.15)
        total_count = len(confidences)
        self.analysis_text.insert(tk.END, f"高置信(S1): {s1_count}/{total_count} ({s1_count / total_count:.1%})\n")
        self.analysis_text.insert(tk.END, f"中置信(S2): {s2_count}/{total_count} ({s2_count / total_count:.1%})\n")
        self.analysis_text.insert(tk.END, f"需验证(S0): {s0_count}/{total_count} ({s0_count / total_count:.1%})\n\n")
        self.analysis_text.insert(tk.END, "=== 逻辑分析 ===\n\n")
        for i, (_,value) in enumerate(result.items()):
            self.analysis_text.insert(tk.END, f"{i+1}. {value['region_name']}: {value['logic']}\n")

        self.analysis_text.config(state=tk.DISABLED)
    
    def load_folder(self):
        """
        打开文件夹选择窗口以选择要加载的图像文件夹
        """
        self.image_path = filedialog.askdirectory(title="选择图像验证文件夹")
        if not self.image_path: return
        self.status_var.set(f"已加载文件夹: {self.image_path}")
        self.QRDetector.image_path_to_csv(self.image_path)
        


    def comprehensive_validation(self):
        """
        独立分析系统验证,对不同光照，不同距离，不同角度，不同设备的系统验证
        """
        self.image_path = filedialog.askdirectory(title="选择验证结果保存目录")
        if not self.image_path: return
        self.csv_path = os.path.join(self.image_path, "独立分析结果.csv")
        result,total=self.QRDetector.comprehensive_validation(self.image_path,self.progress)
        if result:
            self.analysis_text.config(state=tk.NORMAL); self.analysis_text.delete(1.0, tk.END)
            self.analysis_text.insert(tk.END, "=== 综合验证分析 ===\n\n")
            self.analysis_text.insert(tk.END, result)
            self.analysis_text.insert(tk.END, "\n\n独立分析结果已保存到: " + self.csv_path)
            self.analysis_text.config(state=tk.DISABLED)
            self.analysis_text.insert(tk.END, f"\n\n共处理 {total} 个文件")


    def batch_process(self):
        """
        批量处理图像文件
        """
        self.image_path = filedialog.askdirectory(title="选择包含图像的文件夹以及保存结果的目录")
        if self.image_path:
            self.csv_path=os.path.join(self.image_path, "独立分析结果.csv")
            self._batch_process()
    def batch_process_single(self):
        """
        批量处理单圆图像文件
        """
        self.image_path = filedialog.askdirectory(title="选择包含单圆图像的文件夹以及保存结果的目录")
        if self.image_path:
            self.csv_path=os.path.join(self.image_path, "单圆批量分析结果.csv")
            self._batch_process_single()

    def _batch_process(self):
        """
        批量处理图像文件,对目录中的所有jpg, png, jpeg, bmp图像文件进行独立分析
        并将结果保存到指定目录
        """
        try:
            self.status_var.set("开始批量处理...")
            image_files = glob.glob(os.path.join(self.image_path, "*.jpg")) + glob.glob(os.path.join(self.image_path, "*.png")) + glob.glob(os.path.join(self.image_path, "*.jpeg"))+glob.glob(os.path.join(self.image_path, "*.bmp"))
            if not image_files:
                messagebox.showwarning("警告", "未找到图像文件"); return
            results=self.QRDetector.calculate_results(self.image_path,self.progress)
            self._save_batch_results(results)
            self.status_var.set("批量处理完成")
            messagebox.showinfo("完成", f"已处理 {len(results)} 个文件")
        except Exception as e:
            messagebox.showerror("错误", f"批量处理失败: {str(e)}")
    def _batch_process_single(self):
        """
        批量处理单圆图像文件,对目录中的所有jpg, png, jpeg, bmp图像文件进行独立分析
        并将结果保存到指定目录
        """
        try:
            self.status_var.set("开始批量处理单圆图像...")
            image_files = glob.glob(os.path.join(self.image_path, "*.jpg")) + glob.glob(os.path.join(self.image_path, "*.png")) + glob.glob(os.path.join(self.image_path, "*.jpeg"))+glob.glob(os.path.join(self.image_path, "*.bmp"))
            if not image_files:
                messagebox.showwarning("警告", "未找到单圆图像文件"); return
            results=self.analyzer.batch_process_single(image_files,self.progress)
            self.analyzer.save_process_single_circle_to_csv(results)
            self.status_var.set("批量处理单圆图像完成")
            messagebox.showinfo("完成", f"已处理 {len(results)} 个单圆图像文件")
        except Exception as e:
            messagebox.showerror("错误", f"批量处理单圆图像失败: {str(e)}")
    def _save_batch_results(self, results):
        """
        保存批量处理结果到CSV文件
        """
        if self.csv_path is None:
            messagebox.showwarning("警告", "请先选择CSV文件路径"); return
        with open(self.csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["文件名","检测项","结果","置信度","S1","S2","判断逻辑"])
            for result in results:
                for key,value in result["results"].items():
                    writer.writerow([result["image_path"], key, value["predict color"], 
                                     value["confidence"], value["s1"], value["s2"],
                                     value["logic"]])

            
if __name__ == "__main__":
    """
    独立分析GUI界面
    """
    root = tk.Tk()
    app = IndependentPathogenApp(root)
    root.mainloop()