from common.spark import spark_init, read_parquet
import pyspark.sql.functions as F
from pyspark.sql.types import *
import random
import numpy as np
import os

spark = spark_init()

file_config = {
    'data': '',
    'demographic': ''
}

global_params = {
    'output_dir': '',
    'train': '',
    'test': '',
    'min_visit': 5,
    'span': 6,
    'code_col': 'code',
    # 'hf_list':  ['E100','E101','E102','E103','E104','E105','E106','E107','E108','E109',
    #              'E110','E111','E112','E113','E114','E115','E116','E117','E118','E119',
    #              'E120','E121','E122','E123','E124','E125','E126','E127','E128','E129',
    #              'Q242'] diabetes
    'hf_list':  ['F320', 'F321', 'F322', 'F323', 'F328', 'F329', 'F341', 'F331', 'F332',
                 'F333', 'F330', 'F334', 'F338', 'F339', 'F381']

}

data = read_parquet(spark.sqlContext, file_config['data']).na.drop()
demographic = read_parquet(spark.sqlContext, file_config['demographic'])
data = data.join(demographic, data.patid==demographic.patid, 'left').drop(demographic.patid).\
    select(['patid', 'code', 'age', 'gender', 'yob', 'region'])


leng = F.udf(lambda s: len([i for i in range(len(s)) if s[i] == 'SEP']))
span=  F.udf(lambda x: int(x[-1])-int(x[0]))

data = data.withColumn('length',  leng(global_params['code_col']))
data = data[data['length'] >= global_params['min_visit']]
data = data.withColumn('span',  span('age'))
data = data[data['span'] > global_params['span']]
datanew = data.randomSplit([0.8,0.2])
train = datanew[0]
test = datanew[1]

schema = StructType([StructField('patid',StringType(),True),
                     StructField(global_params['code_col'],ArrayType(StringType(),True),True),
                     StructField('age',ArrayType(StringType(),True),True),
                     # StructField('year',ArrayType(StringType(),True),True),
                     StructField('yob',StringType(),True),
                     StructField('gender',StringType(),True),
                     StructField('region',StringType(),True),
                     StructField('label',IntegerType(),True)])


# def full_gen(code, age, patid, year, gender, region, yob):
def full_gen(code, age, patid, gender, region, yob):
    hf_index = [i for i in range(len(code)) if code[i] in global_params['hf_list']]
    sep_index = [i for i in range(len(code)) if code[i] == 'SEP']
    if len(hf_index) == 0:
        if len(sep_index) == global_params['min_visit']:
            sample_index = None
        elif len(sep_index) > global_params['min_visit']:
            last_usable = [i for i in range(len(sep_index)) if
                           (int(age[sep_index[-1]]) - int(age[sep_index[i]])) >= global_params['span']][-1]
            if (last_usable + 1) <= (global_params['min_visit'] - 1):
                sample_index = None
            else:
                sample_index = random.choice(np.arange(global_params['min_visit'] - 1, last_usable + 1))
        else:
            sample_index = None
        if sample_index is None:
            # return '0', ['0'], ['0'], ['0'], '0', '0', '0', -1
            return '0', ['0'], ['0'], '0', '0', '0', -1
        else:
            age = age[:(sep_index[sample_index] + 1)]
            code = code[:(sep_index[sample_index] + 1)]
            patid = patid
            label = 0
            # year = year[:(sep_index[sample_index] + 1)]
            yob = yob
            gender = gender
            region = region
            # return patid, code, age, year, yob, gender, region, label
            return patid, code, age, yob, gender, region, label
    else:
        hf_index = hf_index[0]
        if hf_index < sep_index[0]:
            # return '0', ['0'], ['0'], ['0'], '0', '0', '0', -1
            return '0', ['0'], ['0'], '0', '0', '0', -1
        else:
            visit_index = \
            [i for i in range(1, len(sep_index)) if sep_index[i] > hf_index and sep_index[i - 1] < hf_index][0]
            if (int(age[sep_index[visit_index]]) - int(age[sep_index[visit_index - 1]])) < global_params['span']:
                sample_index = visit_index - 1
                age = age[:(sep_index[sample_index] + 1)]
                code = code[:(sep_index[sample_index] + 1)]
                patid = patid
                # year = year[:(sep_index[sample_index] + 1)]
                yob = yob
                gender = gender
                region = region
                label = 1
                if visit_index >= global_params['min_visit']:
                    # return patid, code, age, year, yob, gender, region, label
                    return patid, code, age, yob, gender, region, label
                else:
                    # return '0', ['0'], ['0'], ['0'], '0', '0', '0', -1
                    return '0', ['0'], ['0'], '0', '0', '0', -1
            else:
                # return '0', ['0'], ['0'], ['0'], '0', '0', '0', -1
                return '0', ['0'], ['0'], '0', '0', '0', -1

test_udf = F.udf(full_gen, schema)

test = test.select(test_udf(global_params['code_col'],'age', 'patid', 'gender', 'region', 'yob').alias("test"))
train = train.select(test_udf(global_params['code_col'],'age', 'patid', 'gender', 'region', 'yob').alias("test"))

test = test.select("test.*")
train = train.select("test.*")

test = test[test['patid']!='0']
train = train[train['patid']!='0']
test = test[test['label']!=-1]
train = train[train['label']!=-1]
test = test[test['yob']!='0']
train = train[train['yob']!='0']
test = test[test['gender']!='0']
train = train[train['gender']!='0']
test = test[test['region']!='0']
train = train[train['region']!='0']

test.write.parquet(os.path.join(global_params['output_dir'], global_params['test']))
train.write.parquet(os.path.join(global_params['output_dir'], global_params['train']))