# -*- coding: utf-8 -*-
# @Time    : 2018/11/13 4:07 PM
# @Author  : Inf.Turing
# @Site    : 
# @File    : stacking_model.py
# @Software: PyCharm

import os
import pandas as pd

import lightgbm as lgb
import numpy as np
from sklearn.metrics import f1_score

path = '../'

w2v_path = path + 'data/w2v'
train = pd.read_csv(path + '/train_2.csv')
test = pd.read_csv(path + '/test_2.csv')

train_first = pd.read_csv(path + '/train_all.csv')

# 用 data_type==0 来标记复赛数据
train['data_type'] = 0
test['data_type'] = 0
# 用 data_type==1 来标记初赛数据
train_first['data_type'] = 1

# 合并数据并把缺失的label填充为0，label=0表示测试数据
data = pd.concat([train, test, train_first], ignore_index=True).fillna(0)

# 当前套餐
data['label'] = data.current_service.astype(int)

# 套餐费空值处理
data = data.replace('\\N', 999)

# 性别处理
data['gender'] = data.gender.astype(int)

# 原始类别特征
origin_cate_feature = ['service_type', 'complaint_level', 'contract_type', 'gender', 'is_mix_service',
                       'is_promise_low_consume',
                       'many_over_bill', 'net_service']

# 原始数值特征
origin_num_feature = ['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee',
                      'age', 'contract_time',
                      'former_complaint_fee', 'former_complaint_num',
                      'last_month_traffic', 'local_caller_time', 'local_trafffic_month', 'month_traffic',
                      'online_time', 'pay_num', 'pay_times', 'service1_caller_time', 'service2_caller_time']

# 数值类型特征转float
for i in origin_num_feature:
    data[i] = data[i].astype(float)

# 将这四个特征映射成对应的向量特征
# 存放w2v特征的名字
w2v_features = []
for col in ['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee']:
    df = pd.read_csv(w2v_path + '/' + col + '.csv')
    df = df.drop_duplicates([col])
    # 所有列名
    fs = list(df)
    fs.remove(col)
    # 记住所有w2v特征列名
    w2v_features += fs
    # 把w2v特征加入data中
    data = pd.merge(data, df, on=col, how='left')


# 存放统计数量的特征名字
count_feature_list = []
def feature_count(data, features=[]):
    if len(set(features)) != len(features):
        print('equal feature !!!!')
        return data
    new_feature = 'count'
    # 把特征名改为count_featureName
    for i in features:
        new_feature += '_' + i.replace('add_', '')
    try:
        del data[new_feature]
    except:
        pass
    # 对特征进行数量统计
    temp = data.groupby(features).size().reset_index().rename(columns={0: new_feature})

    data = data.merge(temp, 'left', on=features)
    count_feature_list.append(new_feature)
    return data

# 统计单特征
data = feature_count(data, ['1_total_fee'])
data = feature_count(data, ['2_total_fee'])
data = feature_count(data, ['3_total_fee'])
data = feature_count(data, ['4_total_fee'])
data = feature_count(data, ['former_complaint_fee'])
data = feature_count(data, ['pay_num'])
data = feature_count(data, ['contract_time'])
data = feature_count(data, ['last_month_traffic'])
data = feature_count(data, ['online_time'])

# 统计双特征
for i in ['service_type', 'contract_type']:
    data = feature_count(data, [i, '1_total_fee'])
    data = feature_count(data, [i, '2_total_fee'])
    data = feature_count(data, [i, '3_total_fee'])
    data = feature_count(data, [i, '4_total_fee'])

    data = feature_count(data, [i, 'former_complaint_fee'])

    data = feature_count(data, [i, 'pay_num'])
    data = feature_count(data, [i, 'contract_time'])
    data = feature_count(data, [i, 'last_month_traffic'])
    data = feature_count(data, [i, 'online_time'])

# 差值特征
diff_feature_list = ['diff_total_fee_1', 'diff_total_fee_2', 'diff_total_fee_3', 'last_month_traffic_rest',
                     'rest_traffic_ratio',
                     'total_fee_mean', 'total_fee_max', 'total_fee_min', 'total_caller_time', 'service2_caller_ratio',
                     'local_caller_ratio',
                     'total_month_traffic', 'month_traffic_ratio', 'last_month_traffic_ratio', 'pay_num_1_total_fee',
                     '1_total_fee_call_fee', '1_total_fee_call2_fee', '1_total_fee_trfc_fee']

# 每月话费差
data['diff_total_fee_1'] = data['1_total_fee'] - data['2_total_fee']
data['diff_total_fee_2'] = data['2_total_fee'] - data['3_total_fee']
data['diff_total_fee_3'] = data['3_total_fee'] - data['4_total_fee']

data['pay_num_1_total_fee'] = data['pay_num'] - data['1_total_fee']

# 流量转结差
data['last_month_traffic_rest'] = data['month_traffic'] - data['last_month_traffic']
# 上个月的转结流量<0的置为0
data['last_month_traffic_rest'][data['last_month_traffic_rest'] < 0] = 0
# 转结流量的费用占比
data['rest_traffic_ratio'] = (data['last_month_traffic_rest'] * 15 / 1024) / data['1_total_fee']


total_fee = []
for i in range(1, 5):
    total_fee.append(str(i) + '_total_fee')
# 对费用去均值，最大值，最小值
data['total_fee_mean'] = data[total_fee].mean(1)
data['total_fee_max'] = data[total_fee].max(1)
data['total_fee_min'] = data[total_fee].min(1)

# 套外主叫通话时长
data['total_caller_time'] = data['service2_caller_time'] + data['service1_caller_time']
# 套外主叫2占比
data['service2_caller_ratio'] = data['service2_caller_time'] / data['total_caller_time']
# 本地主叫时长占比
data['local_caller_ratio'] = data['local_caller_time'] / data['total_caller_time']
# 总流量 = 本地流量 + 转结流量
data['total_month_traffic'] = data['local_trafffic_month'] + data['month_traffic']
# 当月转结流量占比
data['month_traffic_ratio'] = data['month_traffic'] / data['total_month_traffic']
# 上月转结流量占比
data['last_month_traffic_ratio'] = data['last_month_traffic'] / data['total_month_traffic']
# 总话费 - 套外主叫话费1
data['1_total_fee_call_fee'] = data['1_total_fee'] - data['service1_caller_time'] * 0.15
# 总话费 - 套外主叫话费2
data['1_total_fee_call2_fee'] = data['1_total_fee'] - data['service2_caller_time'] * 0.15
# 总话费 - 额外流量费
data['1_total_fee_trfc_fee'] = data['1_total_fee'] - (data['month_traffic'] - 2 * data['last_month_traffic']) * 0.3
#
data.loc[data.service_type == 1, '1_total_fee_trfc_fee'] = None

cate_feature = origin_cate_feature
num_feature = origin_num_feature + count_feature_list + diff_feature_list + w2v_features

# 特征类型转换
for i in cate_feature:
    data[i] = data[i].astype('category')
for i in num_feature:
    data[i] = data[i].astype(float)

# 所有的特征名
feature = cate_feature + num_feature
print(len(feature), feature)

# label为999999的数据很少，因此去掉
data = data[data.label != 999999]

# 训练集 初赛训练集划分
train_x = data[(data.data_type == 1)][feature]
train_y = data[(data.data_type == 1)].label
# 测试集 复赛训练集划分
test_x = data[(data.data_type == 0) & (data.label != 0)][feature]
test_y = data[(data.data_type == 0) & (data.label != 0)].label

lgb_model = lgb.LGBMClassifier(
    boosting_type="gbdt", num_leaves=120, reg_alpha=0, reg_lambda=0.,
    max_depth=-1, n_estimators=100, objective='multiclass', metric="None",
    subsample=0.9, colsample_bytree=0.5, subsample_freq=1,
    learning_rate=0.035, random_state=2018, n_jobs=10
)

# 用类别型特征训练
lgb_model.fit(train_x, train_y, categorical_feature=cate_feature)
print(lgb_model.best_score_)

# 用初赛的数据训练个概率模型，用这个模型去预测复赛的数据，得到的结果作为stacking层的输入特征
stacking_path = path + 'data/stack'
if not os.path.exists(stacking_path):
    print(stacking_path)
    os.makedirs(stacking_path)
    train_proba = lgb_model.predict_proba(test_x[feature])
    # label==0表示最后要提交的数据
    test_proba = lgb_model.predict_proba(data[data.label == 0][feature])
    print(len(train_proba), len(test_proba))
    # stacking层训练集和测试集的用户id
    stacking_train = data[(data.data_type == 0) & (data.label != 0)][['user_id']]
    stacking_test = data[data.label == 0][['user_id']]
    for i in range(11):
        stacking_train['stacking_' + str(i)] = train_proba[:, i]
        stacking_test['stacking_' + str(i)] = test_proba[:, i]
    stacking_train.to_csv(stacking_path + '/train.csv', index=False)
    stacking_test.to_csv(stacking_path + '/test.csv', index=False)

score = f1_score(y_true=test_y, y_pred=lgb_model.predict(test_x), average=None)
print(score)