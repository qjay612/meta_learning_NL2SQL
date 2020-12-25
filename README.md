Data and Code for paper *Leveraging Table Content for Zero-shot Text-to-SQL with Meta-Learning* is available for research purposes.

#ESQL Dataset
ESQL is a Chinese single-table text-to-SQL dataset. Its original version contains 17 tables, these tables are obtained from the real electric energy field, such as electricity mechanism, electricity sales and prices, etc.

### Desensitization
However, because its schema and content involve commercial secrets, we have to desensitize the initial version for public availability, and here is the released version. Specific desensitization practises include the following:
* Hiding vocabulary related to electricity appeared in table schema and Use finance, supply chain vocabulary instead. 
* Replacing the names of companies, mechanisms, projects and names.
* Randomly generate numeric data, including floating point, integer, and date.

### Data Format
#### Question&SQL
They are contained in the `*.jsonl` files. A line looks like the following:
```json
{
   "question":"由李尘负责且主营业务收入大于1130.66的项目有哪些",
   "sql":{
      "sel": [0],
      "agg": [0],
      "cond_conn_op": 0,
      "conds":[
         [
            0,
            1,
            2,
            "李尘",
            null
         ],
         [
            0,
            2,
            3,
            "1130.66",
            null
         ]
      ],
      "ord_by": [
        -1,
        2,
        -1
      ]
   },
   "table_id":"table_10"
}
``` 
The fields represent the following:
* `question`: the natural language question.
* `table_id`: the ID of the table to which this question is addressed.
* `sql`: the SQL query corresponding to the question. This has the following subfields: 
  * `sel`: the numerical index list of the columns that are being selected.
  * `agg`: the numerical index list of the aggregation operators that are being used.
  * `cond_conn_op`:  the numerical index of connection operators in conditions, such as **AND** or **OR**.
  * `conds`: a list of quintuples (`agg_index`, `column_index`, `operator_index`, `value_1`, `value_2`) where: 
    * `agg_index`: the numerical index list of the aggregation operators that are being used in **WHERE**-clause.
    * `column_index`: the numerical index of the condition column that is being used.
    * `operator_index`: the numerical index of the condition operator that is being used.
    * `value_1`: the first comparison value for the condition, in either string or float type.
    * `value_2`: the second comparison value for the condition in float type. It is used for **BETWEEN** key word. (not used)
  * `ord_by`: a list of triplets (`order_index`, `column_index`, `limit`) where: 
    * `order_index`: the numerical index list of the order operators, such as **ASC**
    * `column_index`: the numerical index of the column that is being used in **ORDER BY**-clause.
    * `limit`: the limit number of **ORDER BY**.

Their range of values is shown below:
```pyhon
agg_ops = ['MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
order_ops = ['DESC', 'ASC', None]
cond_ops = ['BETWEEN', '=', '>', '<', '>=', '<=', '!=']
cond_conn_ops = [None, 'AND', 'OR']
```
#### Table
These files are contained in the `tables.jsonl` files. A line looks like the following:
```json
{
  "id": "table_8",
  "name": "表_8", 
  "header": [
    "省", 
    "市", 
    "公司名称", 
    "负责人", 
    "人口总数", 
    "当前用户规模",
    "用户占总人口比", 
    "活跃用户数", 
    "用户增长率（较上一年）", 
    "成本开销", 
    "区域进价", 
    "区域售价", 
    "区域营业收入", 
    "货品滞销率", 
    "工业用户数量", 
    "商业用户数量", 
    "家庭用户数量", 
    "在建工程数", 
    "已完成工程数", 
    "国家重点工程数目"
  ], 
  "header_en": [
    "province", 
    "city", 
    "company", 
    "boss", 
    "population", 
    "current_user_size", 
    "users_as_a_percentage_of_population", 
    "active_users", 
    "user_growth_rate_(year_over_year)", 
    "cost_of_overhead", 
    "regional_purchase_price", 
    "regional_price", 
    "regional_operating_revenue", 
    "unsalable_rate_of_goods", 
    "number_of_industrial_users", 
    "number_of_business_users", 
    "number_of_home_users", 
    "number_of_projects_under_construction", 
    "number_of_projects_completed", 
    "number_of_state_key_projects"
  ], 
  "types": [
    "string", 
    "string", 
    "string", 
    "string", 
    "number", 
    "number", 
    "number", 
    "number", 
    "number",   
    "number", 
    "number", 
    "number", 
    "number", 
    "number", 
    "number", 
    "number", 
    "number", 
    "number", 
    "number", 
    "number"
  ],
  "rows": [
    [
      "黑龙江", 
      "东营", 
      "盈凯证券", 
      "杨翰文", 
      "163", 
      "154", 
      "58.36%", 
      "174", 
      "43.06%", 
      "4782.84", 
      "268.3", 
      "3559.86", 
      "2145.97", 
      "63.19%", 
      "194", 
      "74", 
      "9", 
      "162", 
      "162", 
      "164"
    ], 
    [
      "安徽", 
      "丹东", 
      "本发重工", 
      "翁意晴", 
      "19", 
      "184", 
      "14.22%", 
      "176", 
      "13.86%", 
      "3773.11", 
      "2451.08", 
      "3555.56", 
      "4875.79", 
      "9.29%", 
      "0", 
      "26", 
      "180", 
      "181", 
      "174", 
      "170"
    ]
  ]
}
```
The fields represent the following:
* `id`: the table ID.
* `name`: the table name.
* `header`: a list of Chinese column names in the table.
* `header_en`: a list of English column names in the table.
* `types`: the value type of each column.
* `rows`: a list of rows. Each row is a list of row entries. Here, `rows` only shows the first two rows of the table.

### Setting
To simulate a zero-shot scenario, we set up the data set as follows:
* `train.jsonl` only contains the questions that rely on first 10 tables in `tables.jsonl`.
* `dev.jsonl` and `test.jsonl` contains the questions of all the tables in `tables.jsonl`.
* `dev_zs.jsonl` and `test_zs.jsonl` are the zero-shot subsets of `dev.jsonl` and `test.jsonl`, respectively. They only contains the questions of last 7 tables in `tables.jsonl`, which do not appear in `train.jsonl`.

The question numbers of the dataset are as following table:

| train   | dev    | dev_zs | test   | test_zs |
| --------| :---:  | :----: | :---:  | :-----: |
| 10,000  | 1,000  |   443  | 2,000  | 860    | 

#MC-SQL
MC-SQL is a semantic parsing model used to transform natural language questions into corresponding SQL queries in the single-table scenario.
The implementation of our proposed MC-SQL is based on the [SQLova](https://github.com/naver/sqlova) ([Hwang et al., 2019](https://arxiv.org/pdf/1902.01069.pdf))

## Requirements
* python 3.6
* pytorch 1.4.0
* nltk 3.5
* pytorch-pretrained-bert 0.6.2
* transformers 3.5.1

## Directory
```
├── data                        # Datasets and Pre-trained BERT model
│  ├── esql                     # ESQL
│  │  ├── train.jsonl         
│  │  ├── dev.jsonl           
│  │  ├── test.jsonl         
│  │  ├── dev_zs.jsonl          # zero-shot development set
│  │  ├── test_zs.jsonl         # zero-shot test set
│  │  ├── tables.jsonl       
│  ├── wikisql                  # WikiSQL
│  │  ├── train_tok.jsonl       
│  │  ├── train.tables.jsonl        
│  │  ├── dev_tok.jsonl         
│  │  ├── dev.tables.jsonl          
│  │  ├── test_tok.jsonl        
│  │  ├── test.tables.jsonl        
│  │  ├── dev_zs.jsonl          # zero-shot development set
│  │  ├── dev_zs.tables.jsonl   
│  │  ├── test_zs.jsonl         # zero-shot test set
│  │  ├── test_zs.tables.jsonl
│  ├── bert-base-uncased
│  ├── bert-base-chinese
├── src                         # Src Code
│  ├── bert                     # related code of BERT model
│  ├── meta                     # Meta-Learning 
│  │  ├── meta_esql.py          # meta learner of ESQL         
│  │  ├── meta_wikisql.py       # meta learner of WikiSQL
│  ├── models                   # Text-to-SQL models
│  │  ├── model_esql.py         # table-enhanced model of ESQL
│  │  ├── model_wikisql.py      # table-enhanced model of WikiSQL  
│  │  ├── modules.py            # sub-modules of both text-to-sql models
│  │  ├── nn_utils.py           # utils of models 
│  ├── preprocess                       # Preprocessing data to find the matching content from table
│  │  ├── enhance_header_esql.py        # retrieve value for ESQL
│  │  ├── enhance_header_wikisql.py     # retrieve value for WikiSQL 
│  ├── utils                    # Utils including load_data, vocabulary 
│  │  ├── dbengine.py              
│  │  ├── dictionary.py 
│  │  ├── utils.py              
│  │  ├── utils_esql.py
│  │  ├── utils_wikisql.py
│  ├── pargs.py                 # Hyper-parameters
│  ├── test_esql.py
│  ├── test_esql.sh
│  ├── test_wikisql.py
│  ├── test_wikisql.sh
│  ├── train_esql.py            # training on ESQL
│  ├── train_esql.sh
│  ├── train_wikisql.py         # training on WikiSQL
│  ├── train_wikisql.sh
```
The link of WikiSQL origin data is [here](https://github.com/salesforce/WikiSQL).

The zero-shot subset of WikiSQL is obtained from ([Chang et al., 2019](https://github.com/JD-AI-Research-Silicon-Valley/auxiliary-task-for-text-to-sql))

## Prepare Data
Download pre-trained BERT model [bert-base-uncased](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz) and [bert-base-chinese](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz), unzip and put them under `data/`.

## Running Code
### WikiSQL
#### 1. Preprocess data for perform *Coarse-grained Filtering*
```bash
cd ./src/preprocess
python enhance_header_wikisql.py
cd ..
```

#### 2. Training  
Then, execute the following command for training MC-SQL.
```bash
sh train_wiksql.sh
```
The trained model file is saved under `runs/wikisql/` directory.  
The path format of the trained model is `./runs/wikisql/RUN_ID/checkpoints/best_snapshot_epoch_xx_best_val_acc_xx_meta_learner.pt`


#### 3. Testing
Modify the following content in `test_wikisql.sh` to use the trained moel checkpoint.
```bash
--cpt ./runs/wikisql/RUN_ID/checkpoints/best_snapshot_epoch_xx_best_val_acc_xx_meta_learner.pt
``` 
Then, execute the following command.
```bash
sh test_wikisql.sh
```

### ESQL
#### 1. Preprocess data for perform *Coarse-grained Filtering*
```bash
cd ./src/preprocess
python enhance_header_esql.py
cd ..
```

#### 2. Training  
Then, execute the following command for training MC-SQL.
```bash
sh train_esql.sh
```
The trained model file is saved under `runs/esql/` directory.  
The path format of the trained model is `./runs/esql/RUN_ID/checkpoints/best_snapshot_epoch_xx_best_val_acc_xx_meta_learner.pt`


#### 3. Testing
Modify the following content in `test_esql.sh` to use the trained moel checkpoint.
```bash
--cpt ./runs/esql/RUN_ID/checkpoints/best_snapshot_epoch_xx_best_val_acc_xx_meta_learner.pt
``` 
Then, execute the following command for structure prediction.
```bash
sh test_esql.sh
```

## Results
### WikiSQL

| **Dataset**   | Sel-Col <br />Acc. | Sel-Agg <br />Acc. | Where-Num <br />Acc. | Where-Col <br />Acc. | Where-Op <br />Acc. | Where-Val <br />Acc. | Logical Form <br />Acc. |
| ------------- | :----------------: | :----------------: | :------------------: | :------------------: | :-----------------: | :------------------: | :---------------------: |
|   Dev         | 96.9               | 90.5               | 99.1                 | 97.9                 | 97.5                | 96.7                 | 84.1                    |
|   Test        | 96.4               | 90.6               | 98.8                 | 97.8                 | 97.8                | 96.9                 | 83.7                    |
|   Dev-ZS      | 96.4               | 91.1               | 98.7                 | 96.6                 | 97.1                | 94.8                 | 82.4                    |
|   Test-ZS     | 95.5               | 91.0               | 98.1                 | 96.3                 | 96.7                | 94.2                 | 80.5                    |

Here, ZS is short for zero-shot.

### ESQL(original)

| **Dataset**   | Sel-Num <br />Acc. | Sel-Col <br />Acc. | Sel-Agg <br />Acc. | Where-Conn <br />Acc. | Where-Num <br />Acc. | Where-Col <br />Acc. | Where-Agg <br />Acc. | Where-Op <br />Acc. | Where-Val <br />Acc. | Ord-Ord <br />Acc. | Ord-Col <br />Acc. | Ord-Limit <br />Acc. | Logical Form <br />Acc. |
| ------------- | :----------------: | :----------------: | :-------------------: | :------------------: | :-----------------: | :------------------: | :---------------------: | ------------- | :----------------: | :----------------: | :-------------------: | :------------------: | :-----------------: |
|   Dev         | 100.0              | 99.9               | 97.3                  | 100.0                 | 100.0               | 89.0                  | 98.6                  | 99.3                 | 88.9                  | 91.9                  | 92.2                  | 97.1                  | 75.7                 |
|   Test         | 100.0              | 98.8               | 97.6                  | 100.0                 | 100.0               | 89.4                  | 98.4                  | 99.5                 | 88.8                  | 91.4                  | 92.3                  | 96.9                  | 75.3                 |
|   Dev-ZS      | 100.0              | 99.1               | 95.3                  | 100.0                 | 100.0               | 87.9                  | 98.2                  | 99.6                 | 87.9                  | 91.5                  | 91.3                  | 97.2                  | 74.3                 |
|   Test-ZS      | 100.0              | 99.3               | 95.6                  | 100.0                 | 100.0               | 87.6                  | 98.5                  | 99.6                 | 87.3                  | 91.2                  | 91.4                  | 96.8                  | 72.3                 |

### ESQL(Desensitization)

| **Dataset**   | Sel-Num <br />Acc. | Sel-Col <br />Acc. | Sel-Agg <br />Acc. | Where-Conn <br />Acc. | Where-Num <br />Acc. | Where-Col <br />Acc. | Where-Agg <br />Acc. | Where-Op <br />Acc. | Where-Val <br />Acc. | Ord-Ord <br />Acc. | Ord-Col <br />Acc. | Ord-Limit <br />Acc. | Logical Form <br />Acc. |
| ------------- | :----------------: | :----------------: | :-------------------: | :------------------: | :-----------------: | :------------------: | :---------------------: | ------------- | :----------------: | :----------------: | :-------------------: | :------------------: | :-----------------: |
|   Dev         | 100.0              | 97.2               | 99.1                  | 99.3                 | 98.9                | 93.6                  | 98.9                 | 97.5                 | 92.9                  | 94.2                  | 99.2                  | 99.3                  | 82.8                  |
|   Test         | 100.0              | 97.3               | 99.2                  | 99.6                 | 98.9                | 93.3                  | 98.9                  | 96.8                 | 92.6                  | 93.6                  | 98.8                 | 99.4                  | 82.7                  |
|   Dev-ZS         | 100.0              | 94.6               | 98.0                  | 98.4                 | 97.5                | 93.7                  | 97.5                  | 96.2                 | 91.9                  | 92.1                  | 98.2                  | 99.3                  | 76.7                  |
|   Test-ZS        | 100.0              | 94.2               | 98.0                  | 99.1                 | 97.3                | 92.0                  | 97.3                  | 94.8                 | 90.5                  | 91.0                  | 97.2                  | 98.5                  | 74.8                  |

* We found that MC-SQL had improved performance in ESQL (after desensitization), possibly due to the fact that the desensitized table content data (e.g., company name, person name) was sourced from a limited vocabulary. Thus, the diversity of table content is weakened to some extent. In the future we will update the data set with a more diverse lexicon.
* On the other hand, the decline of MC-SQL in zero-shot tasks has increased, which may be due to the fact that after desensitization, the data set is no longer confined to a single power field, but involves multiple fields such as finance, supply chain, and corporate management. This increases the challenge for the transfer ability of the model.