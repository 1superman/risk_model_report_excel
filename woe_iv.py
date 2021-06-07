import numpy as np
import pandas as pd
import xlsxwriter

def ChiMerge(df, variable, flag, confidenceVal=3.841, bin=10, sample_rate=0.05, sample=None):
    if sample != None:
        df = df.sample(n=sample)
    else:
        df
    # 进行数据格式化录入
    total_num = df.groupby([variable])[flag].count()  # 统计需分箱变量每个值数目
    total_num = pd.DataFrame({'total_num': total_num})  # 创建一个数据框保存之前的结果
    positive_class = df.groupby([variable])[flag].sum()  # 统计需分箱变量每个值正样本数
    positive_class = pd.DataFrame({'positive_class': positive_class})  # 创建一个数据框保存之前的结果
    regroup = pd.merge(total_num, positive_class, left_index=True, right_index=True,
                       how='inner')  # 组合total_num与positive_class
    regroup.reset_index(inplace=True)
    regroup['negative_class'] = regroup['total_num'] - regroup['positive_class']  # 统计需分箱变量每个值负样本数
    regroup = regroup.drop('total_num', axis=1)
    np_regroup = np.array(regroup)  # 把数据框转化为numpy（提高运行效率）
    i = 0
    while (i <= np_regroup.shape[0] - 2):
        if ((np_regroup[i, 1] == 0 and np_regroup[i + 1, 1] == 0) or (
                np_regroup[i, 2] == 0 and np_regroup[i + 1, 2] == 0)):
            np_regroup[i, 1] = np_regroup[i, 1] + np_regroup[i + 1, 1]  # 正样本
            np_regroup[i, 2] = np_regroup[i, 2] + np_regroup[i + 1, 2]  # 负样本
            np_regroup[i, 0] = np_regroup[i + 1, 0]
            np_regroup = np.delete(np_regroup, i + 1, 0)
            i = i - 1
        i = i + 1

    # 对相邻两个区间进行卡方值计算
    chi_table = np.array([])  # 创建一个数组保存相邻两个区间的卡方值
    for i in np.arange(np_regroup.shape[0] - 1):
        chi = (np_regroup[i, 1] * np_regroup[i + 1, 2] - np_regroup[i, 2] * np_regroup[i + 1, 1]) ** 2               * (np_regroup[i, 1] + np_regroup[i, 2] + np_regroup[i + 1, 1] + np_regroup[i + 1, 2]) /               ((np_regroup[i, 1] + np_regroup[i, 2]) * (np_regroup[i + 1, 1] + np_regroup[i + 1, 2]) * (
                  np_regroup[i, 1] + np_regroup[i + 1, 1]) * (np_regroup[i, 2] + np_regroup[i + 1, 2]))
        chi_table = np.append(chi_table, chi)

    # 把卡方值最小的两个区间进行合并（卡方分箱核心）
    while (1):
        try:
            if (len(chi_table) <= (bin - 1) and min(chi_table) >= confidenceVal):
                break
        except:
            break
        chi_min_index = np.argwhere(chi_table == min(chi_table))[0]  # 找出卡方值最小的位置索引
        np_regroup[chi_min_index, 1] = np_regroup[chi_min_index, 1] + np_regroup[chi_min_index + 1, 1]
        np_regroup[chi_min_index, 2] = np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 2]
        np_regroup[chi_min_index, 0] = np_regroup[chi_min_index + 1, 0]
        np_regroup = np.delete(np_regroup, chi_min_index + 1, 0)

        if (chi_min_index == np_regroup.shape[0] - 1):  # 最小值是最后两个区间的时候
            # 计算合并后当前区间与前一个区间的卡方值并替换
            chi_table[chi_min_index - 1] = (
                                           np_regroup[chi_min_index - 1, 1] * np_regroup[chi_min_index, 2] - np_regroup[
                                               chi_min_index - 1, 2] * np_regroup[chi_min_index, 1]) ** 2 \
                                           * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2] +
                                              np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) / \
                                           ((np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2]) * (
                                           np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * (
                                            np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index, 1]) * (
                                            np_regroup[chi_min_index - 1, 2] + np_regroup[chi_min_index, 2]))
            # 删除替换前的卡方值
            chi_table = np.delete(chi_table, chi_min_index, axis=0)

        else:
            # 计算合并后当前区间与前一个区间的卡方值并替换
            chi_table[chi_min_index - 1] = (
                                           np_regroup[chi_min_index - 1, 1] * np_regroup[chi_min_index, 2] - np_regroup[
                                               chi_min_index - 1, 2] * np_regroup[chi_min_index, 1]) ** 2 \
                                           * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2] +
                                              np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) / \
                                           ((np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2]) * (
                                           np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * (
                                            np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index, 1]) * (
                                            np_regroup[chi_min_index - 1, 2] + np_regroup[chi_min_index, 2]))
            # 计算合并后当前区间与后一个区间的卡方值并替换
            chi_table[chi_min_index] = (np_regroup[chi_min_index, 1] * np_regroup[chi_min_index + 1, 2] - np_regroup[
                chi_min_index, 2] * np_regroup[chi_min_index + 1, 1]) ** 2 \
                                       * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2] + np_regroup[
                chi_min_index + 1, 1] + np_regroup[chi_min_index + 1, 2]) / \
                                       ((np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * (
                                       np_regroup[chi_min_index + 1, 1] + np_regroup[chi_min_index + 1, 2]) * (
                                        np_regroup[chi_min_index, 1] + np_regroup[chi_min_index + 1, 1]) * (
                                        np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 2]))
            # 删除替换前的卡方值
            chi_table = np.delete(chi_table, chi_min_index + 1, axis=0)
    # 把分组样本小于一定阈值的组进行合并
    while (1):
        if np.size(list(set(np_regroup[:, 1] + np_regroup[:, 2] > df.shape[0] * sample_rate))) == 1:
            break
        i = np.argmin(np_regroup[:, 1] + np_regroup[:, 2])

        if i == 0:
            chi_min_index = 0
        elif i == np_regroup.shape[0]-1:
            chi_min_index = i - 1
        else:
            chi_min_index = np.argwhere(chi_table == min(chi_table[i-1], chi_table[i]))[0]  # 找出卡方值最小的位置索引
        np_regroup[chi_min_index, 1] = np_regroup[chi_min_index, 1] + np_regroup[chi_min_index + 1, 1]
        np_regroup[chi_min_index, 2] = np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 2]
        np_regroup[chi_min_index, 0] = np_regroup[chi_min_index + 1, 0]
        np_regroup = np.delete(np_regroup, chi_min_index + 1, 0)

        if (chi_min_index == np_regroup.shape[0] - 1):  # 最小值是最后两个区间的时候
            # 计算合并后当前区间与前一个区间的卡方值并替换
            chi_table[chi_min_index - 1] = (
                                                   np_regroup[chi_min_index - 1, 1] * np_regroup[chi_min_index, 2] -
                                                   np_regroup[
                                                       chi_min_index - 1, 2] * np_regroup[chi_min_index, 1]) ** 2 \
                                           * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2] +
                                              np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) / \
                                           ((np_regroup[chi_min_index - 1, 1] + np_regroup[
                                               chi_min_index - 1, 2]) * (
                                                    np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * (
                                                    np_regroup[chi_min_index - 1, 1] + np_regroup[
                                                chi_min_index, 1]) * (
                                                    np_regroup[chi_min_index - 1, 2] + np_regroup[
                                                chi_min_index, 2]))
            # 删除替换前的卡方值
            chi_table = np.delete(chi_table, chi_min_index, axis=0)

        else:
            # 计算合并后当前区间与前一个区间的卡方值并替换
            chi_table[chi_min_index - 1] = (
                                                   np_regroup[chi_min_index - 1, 1] * np_regroup[chi_min_index, 2] -
                                                   np_regroup[
                                                       chi_min_index - 1, 2] * np_regroup[chi_min_index, 1]) ** 2 \
                                           * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2] +
                                              np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) / \
                                           ((np_regroup[chi_min_index - 1, 1] + np_regroup[
                                               chi_min_index - 1, 2]) * (
                                                    np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * (
                                                    np_regroup[chi_min_index - 1, 1] + np_regroup[
                                                chi_min_index, 1]) * (
                                                    np_regroup[chi_min_index - 1, 2] + np_regroup[
                                                chi_min_index, 2]))
            # 计算合并后当前区间与后一个区间的卡方值并替换
            chi_table[chi_min_index] = (np_regroup[chi_min_index, 1] * np_regroup[chi_min_index + 1, 2] -
                                        np_regroup[
                                            chi_min_index, 2] * np_regroup[chi_min_index + 1, 1]) ** 2 \
                                       * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2] + np_regroup[
                chi_min_index + 1, 1] + np_regroup[chi_min_index + 1, 2]) / \
                                       ((np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * (
                                               np_regroup[chi_min_index + 1, 1] + np_regroup[
                                           chi_min_index + 1, 2]) * (
                                                np_regroup[chi_min_index, 1] + np_regroup[chi_min_index + 1, 1]) * (
                                                np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 2]))
            # 删除替换前的卡方值
            chi_table = np.delete(chi_table, chi_min_index + 1, axis=0)
    # print('已完成卡方分箱核心操作，正在保存结果')

    # 把结果保存成一个dataFrame
    result_data = pd.DataFrame()
    result_data['variable'] = [variable] * np_regroup.shape[0]  # 结果表第一列：变量名
    list_temp = []
    for i in np.arange(np_regroup.shape[0]):
        if i == 0:
            x = '(-inf,' + str(round(np_regroup[i, 0], 3)) + ']'
        elif i == np_regroup.shape[0] - 1:
            x = '[' + str(round(np_regroup[i - 1, 0],3)) + ',inf)'
        else:
            x = '(' + str(round(np_regroup[i-1, 0], 3)) + ',' + str(round(np_regroup[i, 0],3)) + ']'
        list_temp.append(x)
    result_data['interval'] = list_temp  # 结果表第二列：区间
    result_data['flag_0'] = np_regroup[:, 2]  # 结果表第三列：负样本数目
    result_data['flag_1'] = np_regroup[:, 1]  # 结果表第四列：正样本数目
    return result_data

def cut_bin(data, rule, y='y', feature='字段英文名', binx='binx',woex='woe',delimiter=','):
    rule['binx'] = rule['binx'].map(str)
    features = rule[feature].drop_duplicates()
    result1 = data.drop(features,axis=1)
    result2 = data.drop(features,axis=1)
    for col in features:
        try:
            ll = []
            for tmp in rule[rule[feature]==col][binx]:
                ll = ll + [tmp.split(delimiter)[0][1:]]
                ll = ll+[tmp.split(delimiter)[1][0:-1]]
            ll = sorted(map(float, set(ll)))
            lab = []
            inter = []
            for tmp in ll:
                for tm in rule[rule[feature]==col][binx]:
                    if float(tm.split(delimiter)[0][1:]) == tmp:
                        lab = lab + [round(rule[(rule[feature]==col) & (rule[binx]==tm)][woex].iat[0],9)]
                        inter = inter + [rule[(rule[feature]==col) & (rule[binx]==tm)][binx].iat[0]]
            lab = pd.unique(lab)
            inter = pd.unique(inter)
            result1[col] = pd.cut(data[col],pd.unique(ll),labels=lab)
            result2[col] = pd.cut(data[col],pd.unique(ll),labels=inter)     
        except:
            continue
    return result1,result2

def get_woe_iv(f_bin):
    try:
        f_bin = f_bin[['字段英文名', 'binx', 'good_cnt', 'bad_cnt']]
    except:
        f_bin.columns = ['字段英文名', 'binx', 'good_cnt', 'bad_cnt']
    f_bin['total'] = f_bin['good_cnt']+f_bin['bad_cnt']
    f_bin['pct'] = f_bin['total']/f_bin['total'].sum()
    f_bin['bad_pct'] = f_bin['bad_cnt']/f_bin['bad_cnt'].sum()
    f_bin['good_pct'] = f_bin['good_cnt']/f_bin['good_cnt'].sum()
    f_bin['bad_rate'] = f_bin['bad_cnt']/f_bin['total']
    f_bin['woe'] = np.log(f_bin['bad_pct']/f_bin['good_pct'])
    f_bin['iv'] = (f_bin['bad_pct']-f_bin['good_pct'])*f_bin['woe']
    f_bin['iv_sum'] = f_bin['iv'].sum()
    return f_bin[['字段英文名', 'binx', 'good_cnt', 'bad_cnt', 'total', 'pct', 'bad_pct','good_pct', 'bad_rate', 'woe', 'iv', 'iv_sum']]

def get_auto_bin(df, feature):
    label = 'y_label'
    f_bin = pd.DataFrame(columns=['字段英文名', 'binx', 'good_cnt', 'bad_cnt', 'total', 'pct', 'bad_pct','good_pct', 'bad_rate', 'woe', 'iv', 'iv_sum'])
    for i in feature:
        try:
            f_binx = ChiMerge(df[[i, label]], i, label)
            f_binx = get_woe_iv(f_binx)
            f_bin = f_bin.append(f_binx)
        except:
            continue
    f_bin.reset_index(inplace=True)
    del f_bin['index']
    return f_bin

def compute_woe_iv_psi_of_one_grouped_var(data, flag=True, y='cheat',x = 'office_no_dup_mark',delimiter=', '):
    if flag:
        train = data[data['type']=='train']
        test = data[data['type']=='test']
        train1 = compute_woe_iv_of_one_grouped_var(train,y,x,delimiter)['summary']
        test0 = compute_woe_iv_of_one_grouped_var(test,y,x,delimiter)
        test1 = test0['pct_mapping'].rename(columns={'pct':'test_pct'})
        test2 = test0['summary']
        result = pd.merge(train1,test1,how='left',on='binx')
        result['psi'] = sum((result['test_pct']-result['pct'])*np.log(result['test_pct']/result['pct']))
        return result.drop('test_pct',axis=1),test2
    else :
        result = compute_woe_iv_of_one_grouped_var(data,y,x,delimiter)['summary']
        result['psi'] = ''
        return result,None

def compute_woe_iv_of_one_grouped_var(data,y='cheat',x = 'office_no_dup_mark',delimiter=', '):
    df = data.copy(deep = True)
    df[y] = df[y].astype(int)
    summary = df.groupby(x,as_index = True).agg({y:[np.size,np.sum,lambda x:np.size(x) - np.sum(x)]})
    summary.columns = ['cnt','cnt_1','cnt_0']
    summary['pct'] = summary['cnt'] / summary['cnt'].sum()
    summary['pct_1'] = summary['cnt_1'] / summary['cnt_1'].sum()
    summary['pct_0'] = summary['cnt_0'] / summary['cnt_0'].sum()
    summary['cum_pct_1'] = summary['pct_1'].cumsum()
    summary['cum_pct_0'] = summary['pct_0'].cumsum()
    summary['woe'] = np.log(summary['pct_1'] / summary['pct_0'])
    summary['iv'] = summary['woe'] * (summary['pct_1'] - summary['pct_0'])
    summary['binx'] = summary.index
    summary['cnt_1_rate'] = summary['cnt_1'] / summary['cnt']
    summary['iv_sum'] = summary['iv'].sum()
    summary['order'] = summary['binx'].map(lambda m: float(str(m).split(delimiter)[0][1:]))
    summary = summary.sort_values(by=['order']).drop('order',axis=1)
    summary = summary[['binx','cnt','cnt_1','cnt_0','pct','pct_1','pct_0','cum_pct_1','cum_pct_0','woe','iv','cnt_1_rate','iv_sum']]
    summary[['cnt','cnt_1','cnt_0','pct','pct_1','pct_0','cum_pct_1','cum_pct_0','woe','iv','cnt_1_rate','iv_sum']] = \
        summary[['cnt','cnt_1','cnt_0','pct','pct_1','pct_0','cum_pct_1','cum_pct_0','woe','iv','cnt_1_rate','iv_sum']].applymap(lambda x: round(x, 4))
    return({'summary':summary,'iv':summary['iv'].sum(),'woe_mapping':summary[['binx','woe',]],'pct_mapping':summary[['binx','pct']]})

def down_data_picture(data, features,y='y',flag=True, delimiter=', '):
    ivandwoelist1 = pd.DataFrame()
    ivandwoelist2 = pd.DataFrame()
    for x in features:
        train,test = compute_woe_iv_psi_of_one_grouped_var(data,flag,y,x,delimiter)
        train['feature'] = x
        test['feature'] = x
        ivandwoelist1 = pd.concat(([ivandwoelist1, train[['feature']+train.columns[:-1].tolist()]]), ignore_index=True)
        ivandwoelist2 = pd.concat(([ivandwoelist2, test[['feature']+test.columns[:-1].tolist()]]), ignore_index=True)
    return ivandwoelist1, ivandwoelist2


def woe_picture(result2, rule, excel_name, y='y_label', flag=False, feature='x', binx='binx', woex='woex', delimiter=', '):
    features = list(rule[feature].drop_duplicates())
    ivandwoelist1, ivandwoelist2 = down_data_picture(result2,features,y,flag,delimiter)
    return ivandwoelist1, ivandwoelist2