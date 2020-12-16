import pandas as pd
import numpy as np
from datetime import date
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 数据预处理部分


# 读取数据
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 获取日期（数据值最后修改日期）
today = pd.to_datetime(date(2020, 11, 15))
# 获得球员年龄
train['birth_date'] = pd.to_datetime(train['birth_date'])
train['age'] = (today - train['birth_date']).apply(lambda x: x.days) / 365.

test['birth_date'] = pd.to_datetime(test['birth_date'])
test['age'] = (today - test['birth_date']).apply(lambda x: x.days) / 365.

# 计算球员的身体质量指数（BMI）
train['BMI'] = 10000. * train['weight_kg'] / (train['height_cm'] ** 2)
test['BMI'] = 10000. * test['weight_kg'] / (test['height_cm'] ** 2)

# 判断一个球员是否是守门员
train['is_gk'] = train['gk'] > 0
test['is_gk'] = test['gk'] > 0

# 对分类数据进行独热编码
columns_to_enconding=['work_rate_att', 'work_rate_def', 'preferred_foot']
train = pd.get_dummies(train, columns=columns_to_enconding)
test = pd.get_dummies(test, columns=columns_to_enconding)

# 获得球员最擅长位置上的评分
positions = ['rw', 'rb', 'st', 'lw', 'cf', 'cam', 'cm', 'cdm', 'cb', 'lb', 'gk']
train['best_pos'] = train[positions].max(axis=1)
test['best_pos'] = test[positions].max(axis=1)

# 删除已经处理过的特征值
cols = ['rw', 'rb', 'st', 'lw', 'cf', 'cam', 'cm', 'cdm', 'cb', 'lb', 'gk', 'weight_kg', 'height_cm', 'birth_date', 'nationality']
train = train.drop(cols, axis=1)
test = test.drop(cols, axis=1)

# 根据是否是守门员来切分数据集
train1 = train.drop(['y', 'is_gk'], axis=1)
used_feat = train1.columns
train_no_gk = train[train['is_gk'] == False][used_feat].copy()
y_train_no_gk = train[train['is_gk'] == False]['y'].copy()
test_no_gk = test[test['is_gk'] == False][used_feat].copy()
train_is_gk = train[train['is_gk'] == True][used_feat].copy()
y_train_is_gk = train[train['is_gk'] == True]['y'].copy()
test_is_gk = test[test['is_gk'] == True][used_feat].copy()
test['pred'] = 0
# 进行特征提取(用非守门员数据进行训练）
X_train,X_test,y_train,y_test = train_test_split(train_no_gk,y_train_no_gk,test_size=0.3,random_state=0)

# 导入到lightgbm矩阵
lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)
# 设置参数
params = {
    'num_leaves': 5,
    'metric': ('auc', 'logloss'),
    'verbose': 0
}

evals_result = {}

print('开始训练...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=[lgb_train, lgb_test],
                evals_result=evals_result,
                verbose_eval=10)

print('画出训练结果...')
ax = lgb.plot_metric(evals_result, metric='auc')
plt.show()

print('画特征重要性排序...')
ax = lgb.plot_importance(gbm, max_num_features=20)
plt.show()

# 挑选出前20个特征
F_cols = ['best_pos', 'potential', 'age', 'vision', 'finishing', 'long_shots', 'heading_accuracy', 'sho', 'positioning', 'phy', 'ball_control', 'short_passing', 'sliding_tackle', 'interceptions', 'dribbling', 'standing_tackle', 'aggression', 'volleys', 'reactions', 'long_passing']

# 用非守门员数据训练随机森林
reg_ngk = RandomForestRegressor( max_depth=85,
                         min_samples_split = 2,
                         min_samples_leaf = 2,
                         n_estimators=450,
                         verbose = 0)
reg_ngk.fit(train_no_gk[F_cols], y_train_no_gk)
preds1 = reg_ngk.predict(test_no_gk[F_cols])
test.loc[test['is_gk'] == False, 'pred'] = preds1

# 用守门员数据训练随机森林模型
reg_gk = RandomForestRegressor( max_depth=85,
                         min_samples_split = 2,
                         min_samples_leaf = 2,
                         n_estimators=450,
                         verbose = 0)
reg_gk.fit(train_is_gk[F_cols], y_train_is_gk)
preds2 = reg_gk.predict(test_is_gk[F_cols])
test.loc[test['is_gk'] == True, 'pred'] = preds2
"""
手动调参，调试的参数主要是随机森林模型中的max_depth。尝试了70,75,85,95，其mae值分别为	20.997147，	20.958567，	20.926438，	20.983001 ，故最终选定为max_depth=85
"""
test['pred'].to_csv('submission1.txt', index=None)
# 由于我不会将输出的数据中第一行pred给删除掉。。因而我选择先导出为submission1.txt，然后再手动删除掉第一行的‘pred’，最后再保存为submission.txt。