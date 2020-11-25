import os
import os.path as osp
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch.optim as optim
from face_recognize_sdk.config import Config as conf
from face_recognize_sdk.model import ResIRSE
from face_recognize_sdk.model.metric import ArcFace, CosFace
from face_recognize_sdk.model.loss import FocalLoss
from face_recognize_sdk.dataset import load_data

SIM_THRESHOLD = 0.5
class FaceRecognizer():

    def __init__(self, training=False):
        os.environ['CUDA_VISIBLE_DEVICES'] = conf.gpu_devices
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = ResIRSE(conf.embedding_size, conf.drop_ratio).to(self.device)
        self.model = nn.DataParallel(self.model)
        if training:
            self.dataloader, self.class_num = load_data(conf, training=True)
            self.embedding_size = conf.embedding_size
            self.metric = ArcFace(self.embedding_size, self.class_num).to(self.device) if conf.metric == 'arcface' \
                else CosFace(self.embedding_size, self.class_num).to(self.device)
            self.metric = nn.DataParallel(self.metric)
            self.criterion = FocalLoss(gamma=2) if conf.loss == 'focal_loss' \
                else nn.CrossEntropyLoss()
            self.optimizer = optim.SGD([{'params': self.model.parameters()}, {'params': self.metric.parameters()}],
                    lr=conf.lr, weight_decay=conf.weight_decay) if conf.optimizer == 'sgd' \
                else optim.Adam([{'params': self.model.parameters()}, {'params': self.metric.parameters()}],
                    lr=conf.lr, weight_decay=conf.weight_decay)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=conf.lr_step, gamma=0.1)
            self.checkpoints = conf.checkpoints
            os.makedirs(self.checkpoints, exist_ok=True)
            if conf.restore:
                self.weights_path = osp.join(self.checkpoints, conf.restore_model)
                self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
        else:
            self.model.load_state_dict(torch.load(conf.test_model, map_location=self.device))
            self.model.eval()


    def preprocess_batch(self, img_paths):
        """
        prepocess a bacth of imgs
        :param img: the path of a single image
        :return:
        """
        res = []
        for img_path in img_paths:
            im = Image.open(img_path)
            im = conf.test_transform(im)
            res.append(im)
        data = torch.cat(res, dim=0)  # shape: (batch, 128, 128)
        data = data[:, None, :, :]  # shape: (batch, 1, 128, 128)
        return data


    def preprocess_single(self, img_path):
        """
        prepocess a single img
        :param img: the path of a single image
        :return:
        """
        res = []
        im = Image.open(img_path)
        im = conf.test_transform(im)
        res.append(im)
        data = torch.cat(res, dim=0)  # shape: (batch, 128, 128)
        data = data[:, None, :, :]  # shape: (batch, 1, 128, 128)
        return data


    def generate_feature_batch(self, img, img_path):
        # generate features for a batch of images
        img = img.to(self.device)
        self.model = self.model.to(self.device)
        with torch.no_grad():
            features = self.model(img)
        res = {img: feature for (img, feature) in zip(img_path, features)}
        return res


    def generate_feature_single(self, img):
        # generate feature for a single image
        img = img.to(self.device)
        self.model = self.model.to(self.device)
        with torch.no_grad():
            features = self.model(img)
        return features


    def calculate_feature_sim(self, f1, f2) -> float:
        # calculate the similarity betweeen a single pair
        similarity = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))
        return similarity


    def verify(self, gallery_img, query_img, sim_threshold=SIM_THRESHOLD) -> bool:
        f1 = self.generate_feature_single(gallery_img)
        f2 = self.generate_feature_single(query_img)
        x1 = f1[0].cpu().numpy()
        x2 = f2[0].cpu().numpy()
        sim = self.calculate_feature_sim(x1, x2)

        if sim > sim_threshold:
            return True
        else:
            return False


    def verify_test(self, gallery_img, query_img, sim_threshold=SIM_THRESHOLD) -> bool:
        # just for unit test
        f1 = self.generate_feature_single(gallery_img)
        f2 = self.generate_feature_single(query_img)
        sim = self.calculate_feature_sim(f1, f2)
        return sim


    def train(self):
        # Start training
        self.model.train()
        for e in range(conf.epoch):
            for data, labels in tqdm(self.dataloader, desc="Epoch {}/{}".format(e, conf.epoch),
                                     ascii=True, total=len(self.dataloader)):
                data = data.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                embeddings = self.model(data)
                thetas = self.metric(embeddings, labels)
                loss = self.criterion(thetas, labels)
                loss.backward()
                self.optimizer.step()

            print("Epoch {}/{}, Loss: {}".format(e, conf.epoch, loss))

            backbone_path = osp.join(self.checkpoints, "{}.pth".format(e))
            torch.save(self.model.state_dict(), backbone_path)
            self.scheduler.step()


    def cal_far_frr_acc(self, y_score, y_true, threshold):
        """
        calulate the FAR, FRR and accuracy for a specific threshold
        :param y_score: the similarities of the pairs
        :param y_true: the label of the pairs (positive or negative)
        :param threshold: the threshold
        :return: FAR, FRR, acc
        """
        FP = 0
        FN = 0
        totalP = 0
        totalN = 0

        y_score = np.asarray(y_score)
        y_true = np.asarray(y_true)  # true y
        y_test = (y_score >= threshold)  # predicted y
        acc = np.mean((y_test == y_true).astype(int))
        for i in range(len(y_score)):
            if y_true[i] != 1 and y_test[i] == 1:
                FP += 1
            elif y_true[i] == 1 and y_test[i] != 1:
                FN += 1
            if y_true[i] == 1:
                totalP += 1
            else:
                totalN += 1
        FAR = FP / float(totalN)  # False Accept Rate in percentage
        FRR = FN / float(totalP)  # False Reject Rate in percentage
        return FAR, FRR, acc


    def cal_list_far_frr_acc(self, y_score, y_true):
        """
        calulate the list of FAR and FRR for different choice of threshold, and find the best one
        :param y_score: the similarities of the pairs
        :param y_true: the label of the pairs (positive or negative)
        :param threshold: the threshold
        :return: FAR, FRR, best_acc, best_th
        """
        far = dict()
        frr = dict()
        best_th = 1.0  # the best threshold
        best_far = 1.0
        best_frr = 1.0
        best_far_p_frr = 1.0  # the best result of far plus frr
        best_acc = 0
        step = 1

        for i in range(0, 100, step):  # try different threshold and further find a best one
            threshold = i / float(100)
            far[i], frr[i], acc = self.cal_far_frr_acc(y_score, y_true, threshold)
            if (far[i] + frr[i]) < best_far_p_frr:
                best_th = threshold
                best_far = far[i]
                best_frr = frr[i]
                best_far_p_frr = far[i] + frr[i]
                best_acc = acc
        return far, frr, best_far, best_frr, best_acc, best_th


    def compute_criteria(self, feature_dict, pair_list, test_root):
        with open(pair_list, 'r') as f:
            pairs = f.readlines()

        similarities = []
        labels = []
        for pair in pairs:
            img1, img2, label = pair.split()
            img1 = osp.join(test_root, img1)
            img2 = osp.join(test_root, img2)
            feature1 = feature_dict[img1].cpu().numpy()
            feature2 = feature_dict[img2].cpu().numpy()
            label = int(label)

            similarity = self.calculate_feature_sim(feature1, feature2)
            similarities.append(similarity)
            labels.append(label)

        # accuracy, threshold = threshold_search(similarities, labels) # calculate the accuracy and find the best threshold
        # FAR, FRR = cal_far_frr(similarities, labels, threshold) # calculate the FAR and FRR
        # return accuracy, threshold, FAR, FRR
        FAR_list, FRR_list, FAR, FRR, accuracy, threshold = self.cal_list_far_frr_acc(similarities,
                                                                                 labels)  # return the FAR and FRR list for different threshold, and pick the best threshold
        return FAR_list, FRR_list, FAR, FRR, accuracy, threshold


    def plot_far_frr(self, far, frr):
        """
        plot the far and frr of different thresholds
        :param far: the list of far
        :param frr: the list of frr
        :return:
        """
        axisVal = np.arange(0, 1.00, 0.01)

        # PLOT FAR FRR
        plt.figure()
        lw = 2
        plt.plot(far.values(), axisVal, label='False Accept Rate', color='blue', lw=lw)
        plt.plot(axisVal, frr.values(), label='False Reject Rate', color='red', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Treshold')
        plt.ylabel('Errors')
        plt.title('FAR and FRR')
        plt.legend(loc="lower right")
        plt.savefig("FAR_FRR.png")



if __name__ == "__main__":
    face_recognizer = FaceRecognizer()

    image_path_1 = '../data/1.png'
    # image_path_1 = '../data/5.jpeg'
    img_1 = face_recognizer.preprocess_single(image_path_1)
    image_path_2 = '../data/2.png'
    # image_path_2 = '../data/6.jpeg'
    img_2 = face_recognizer.preprocess_single(image_path_2)

    # execute verify
    sim = face_recognizer.verify_test(img_1, img_2, sim_threshold=0.38)

    # print
    print(sim)