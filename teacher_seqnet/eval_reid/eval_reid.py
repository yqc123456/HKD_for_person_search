import torch
import numpy as np
# from tools import time_now, NumpyCatMeter, TorchCatMeter, CMC, CMCWithVer
import os
import os.path as osp
from scipy.io import loadmat
from .retrieval2_cuhk import CMCWithVer
from datasets.prw import PRW
import re


def analysis_img_info(file_name):
    """
    param file_name: format like 0844_c3s2_107328_01.jpg
    :return:
    """
    split_list = file_name.replace('.jpg', '').replace('c', '').replace('s', '_').split('_')
    identi_id, camera_id = int(split_list[0]), int(split_list[1])
    return identi_id, camera_id


def testwithVer2(config, loaders, gallery_features, query_features):
    qry_name_to_feat = {}
    glry_name_to_feat = {}
    loaders = [loaders.duke_query_loader, loaders.duke_gallery_loader]

    for path, feat in zip(loaders[0].dataset.samples, query_features):
        imname = path[0].split('/')[-1]
        qry_name_to_feat[imname] = feat

    for path, feat in zip(loaders[1].dataset.samples, gallery_features):
        imname = path[0].split('/')[-1]
        glry_name_to_feat[imname] = feat

    # custom made
    testflag = 'TestG50'
    gallery_path = config.gallery_path
    if gallery_path is not None:
        detect_root = '/'.join(gallery_path.split('/')[:-2]) + '/'
        if 'fcos' in gallery_path:
            ids_dir = os.path.join(detect_root, 'trainedfcos_ids')
        elif 'faster' in gallery_path:
            ids_dir = os.path.join(detect_root, 'trainedfaster_ids')
    else:
        detect_root = r'/home/Newdisk/yangqingchun/data/G2APS_in_market1501_style/market1501/'
        ids_dir = os.path.join(detect_root, 'bounding_box_test_ids')

    print('Now detector root is: ', detect_root)

    testmat = r'/home/Newdisk/yangqingchun/data/G2APS/ssm/annotation/test/train_test/TestG50.mat'
    project_file = r'/home/Newdisk/yangqingchun/data/G2APS/ssm/annotation/PidProjectIndex.mat'
    project = loadmat(project_file)['PidProjectIndex'].squeeze()
    testinfo = loadmat(testmat)[testflag][0]
    allglry = []
    maps = []
    cmcs = []

    # todo:不要忘记去掉AP计算中对-1的筛选
    for idx, item in enumerate(testinfo):
        # print(idx, len(testinfo))
        query = item[0][0][0]
        imname = query[0][0]
        height = imname[3]
        frame = imname.split('.')[0].split('_')[-1]
        frame = frame.replace('0', height, 1)
        idname = query[3][0]
        conciseid = str(project[int(idname) - 1, 1])  # 转换为连续id，这里下标从0开始，matlab从1开始

        qry_mkt = str(conciseid).zfill(4) + '_c1s1_' + frame + '_01.jpg'

        qry_pid, qry_camid = analysis_img_info(qry_mkt)
        glry = item[1][0]

        imgs = [g[0][0] for g in glry]
        gtnum = 0
        for g in glry:
            if len(g[1][0]) != 0:
                gtnum += 1

        boxes = []
        for img in imgs:
            pth = osp.join(ids_dir, img.split('.')[0])
            # 随着得分阈值的升高，有的图像上可能一个box都没有检测出来
            if not osp.exists(pth):
                continue
            boxes.extend(os.listdir(pth))
        allglry.append(boxes)
        glry_pids, glry_camids = [], []

        for x in boxes:
            pid, camid = analysis_img_info(x)
            glry_pids.append(pid)
            glry_camids.append(camid)

        qry_id = np.array(int(qry_mkt.split('_')[0]))
        tp_num = len(np.argwhere(glry_pids == qry_id))
        recall_rate = tp_num / gtnum * 1.0

        # feat1
        qry_feat = qry_name_to_feat[qry_mkt][None, :]
        glry_feats = [glry_name_to_feat[x] for x in boxes if x in glry_name_to_feat]
        if len(glry_feats) != len(boxes):
            print('Warning: num not same')
            imgs = [x for x in boxes if x not in glry_name_to_feat]
            print(imgs)

        glry_feats = np.stack(glry_feats)
        # glry_feats = np.concatenate(glry_feats)
        qry_pid = np.array([qry_pid])
        qry_camid = np.array([qry_camid])
        glry_pids = np.array(glry_pids)
        glry_camids = np.array(glry_camids)

        query_info = (qry_feat, qry_camid, qry_pid)
        gallery_info = (glry_feats, glry_camids, glry_pids)

        mAP, CMC = CMCWithVer()(query_info, gallery_info)

        real_map = min(mAP, mAP * recall_rate)
        maps.append(real_map)
        cmcs.append(CMC)
        CMC = np.array(cmcs)
        mean_cmc = np.mean(CMC, axis=0)

        if idx == len(testinfo) - 1:
            print('current ap:', round(mAP, 4), 'ap after scaled by recall:', round(real_map, 4))
            print('till now map:', round(np.array(maps).mean(), 4))
            print('till now rank k:', round(mean_cmc[0], 4), round(mean_cmc[1], 4), round(mean_cmc[2], 4))


def testwithVer2_PRW(config, loaders, gallery_features, query_features, ignore_cam_id=True, ):
    qry_name_to_feat = {}
    glry_name_to_feat = {}
    loaders = [loaders.duke_query_loader, loaders.duke_gallery_loader]

    for path, feat in zip(loaders[0].dataset.samples, query_features):
        imname = path[0].split('/')[-1]
        qry_name_to_feat[imname] = feat

    for path, feat in zip(loaders[1].dataset.samples, gallery_features):
        imname = path[0].split('/')[-1]
        glry_name_to_feat[imname] = feat

    prwpth = r'/home/Newdisk/yangqingchun/data/PRW-v16.04.20'
    query_dataset = PRW(prwpth, None, 'query').annotations
    gallery_dataset = PRW(prwpth, None, 'gallery').annotations
    ids_dir = '/home/Newdisk/yangqingchun/data/PRW_market_style/bounding_box_test'

    maps, cmcs = [], []
    for i in range(len(query_dataset)):
        query_imname = query_dataset[i]["img_name"]
        qry_pid = query_dataset[i]["pids"]
        qry_camid = query_dataset[i]["cam_id"]
        qry_mkt = str(qry_pid[0]) + '_' + query_imname

        # Find all occurence of this query
        gallery_imgs = []
        for x in gallery_dataset:
            if qry_pid in x["pids"] and x["img_name"] != query_imname:
                gallery_imgs.append(x)
        gtnum = len(gallery_imgs)

        boxes = os.listdir(ids_dir)
        glry_pids, glry_camids = [], []

        for x in boxes:
            pid, camid = analysis_img_info(x)
            glry_pids.append(pid)
            glry_camids.append(camid)

        qry_id = np.array(int(qry_mkt.split('_')[0]))
        tp_num = len(np.argwhere(glry_pids == qry_id))
        recall_rate = tp_num / gtnum * 1.0
        if recall_rate > 1:
            recall_rate = 1

        # feat1
        qry_feat = qry_name_to_feat[qry_mkt][None, :]
        glry_feats = [glry_name_to_feat[x] for x in boxes if x in glry_name_to_feat]
        if len(glry_feats) != len(boxes):
            print('Warning: num not same')
            imgs = [x for x in boxes if x not in glry_name_to_feat]
            print(imgs)

        glry_feats = np.stack(glry_feats)
        qry_pid = np.array([qry_pid])
        qry_camid = np.array([qry_camid])
        glry_pids = np.array(glry_pids)
        glry_camids = np.array(glry_camids)

        query_info = (qry_feat, qry_camid, qry_pid)
        gallery_info = (glry_feats, glry_camids, glry_pids)

        mAP, CMC = CMCWithVer()(query_info, gallery_info)

        real_map = min(mAP, mAP * recall_rate)
        maps.append(real_map)
        cmcs.append(CMC)
        CMC = np.array(cmcs)
        mean_cmc = np.mean(CMC, axis=0)

        if i == len(query_dataset) - 1:
            print('current ap:', round(mAP, 4), 'ap after scaled by recall:', round(real_map, 4))
            print('till now map:', round(np.array(maps).mean(), 4))
            print('till now rank k:', round(mean_cmc[0], 4), round(mean_cmc[1], 4), round(mean_cmc[2], 4))



def testwithVer2_CUHKSYSU(config, loaders, gallery_features, query_features):
    qry_name_to_feat = {}
    glry_name_to_feat = {}
    loaders = [loaders.duke_query_loader, loaders.duke_gallery_loader]

    for path, feat in zip(loaders[0].dataset.samples, query_features):
        imname = path[0].split('/')[-1]
        qry_name_to_feat[imname] = feat

    for path, feat in zip(loaders[1].dataset.samples, gallery_features):
        imname = path[0].split('/')[-1]
        glry_name_to_feat[imname] = feat

    # custom made
    testflag = 'TestG50'
    ids_dir = r'/home/Newdisk/yangqingchun/data/cuhk_market_style/bounding_box_test'
    testmat = r'/home/Newdisk/yangqingchun/data/CUHK-SYSU/dataset/annotation/test/train_test/TestG50.mat'

    testinfo = loadmat(testmat)[testflag][0]
    maps = []
    cmcs = []

    # todo:不要忘记去掉AP计算中对-1的筛选
    for idx, item in enumerate(testinfo):
        if idx % 50 == 0:
            print(idx, len(testinfo))
        query = item[0][0][0]
        imname = query[0][0]
        frame = imname.split('.')[0][1:]
        idname = query[3][0][1:]

        qry_mkt = str(idname).zfill(5) + '_c1s1_' + frame.zfill(6) + '_01.jpg'

        qry_pid, qry_camid = analysis_img_info(qry_mkt)
        glry = item[1][0]

        gtnum = 0
        for g in glry:
            if len(g[1][0]) != 0:
                gtnum += 1

        boxes = os.listdir(ids_dir)
        glry_pids, glry_camids = [], []

        for x in boxes:
            pid, camid = analysis_img_info(x)
            glry_pids.append(pid)
            glry_camids.append(camid)

        qry_id = np.array(int(qry_mkt.split('_')[0]))
        tp_num = len(np.argwhere(glry_pids == qry_id))
        recall_rate = tp_num / gtnum * 1.0
        if recall_rate > 1:
            recall_rate = 1

        # feat1
        qry_feat = qry_name_to_feat[qry_mkt][None, :]
        glry_feats = [glry_name_to_feat[x] for x in boxes if x in glry_name_to_feat]
        if len(glry_feats) != len(boxes):
            print('Warning: num not same')
            imgs = [x for x in boxes if x not in glry_name_to_feat]
            print(imgs)

        glry_feats = np.stack(glry_feats)
        qry_pid = np.array([qry_pid])
        qry_camid = np.array([qry_camid])
        glry_pids = np.array(glry_pids)
        glry_camids = np.array(glry_camids)

        query_info = (qry_feat, qry_camid, qry_pid)
        gallery_info = (glry_feats, glry_camids, glry_pids)

        mAP, CMC = CMCWithVer()(query_info, gallery_info)

        real_map = min(mAP, mAP * recall_rate)
        maps.append(real_map)
        cmcs.append(CMC)
        CMC = np.array(cmcs)
        mean_cmc = np.mean(CMC, axis=0)

        if idx == len(testinfo) - 1:
            print('current ap:', round(mAP, 4), 'ap after scaled by recall:', round(real_map, 4))
            print('till now map:', round(np.array(maps).mean(), 4))
            print('till now rank k:', round(mean_cmc[0], 4), round(mean_cmc[1], 4), round(mean_cmc[2], 4))
