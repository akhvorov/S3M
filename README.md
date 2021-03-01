# S3M

This repository contains the code of our 
[MSR 2021](https://2021.msrconf.org/details/msr-2021-technical-papers/26/S3M-Siamese-Stack-Trace-Similarity-Measure) 
paper "S3M: Siamese Stack (Trace) Similarity Measure".

# Data

## NetBeans
You can find NetBeans dataset from paper to reproduce experiments 
on [Figshare](https://figshare.com/articles/dataset/netbeans_stacktraces_json/14135003).

## Format
You can use other datasets that match the following format.
The data presented as a list of report in the following JSON:
```
[
    {
        "bug_id": 1234,
        "dup_id": 12,
        "creation_ts": 1234567891234,
        "stacktrace": {
            "exception": ["java.lang.Exception"], 
            "frames": [
                {"function": "java.util.ArrayList.get"}, 
                {"function": "com.company.Class1.method1"}, 
                {"function": "com.company.Class2.method2"}, 
                {"function": "com.company.Class1.method2"}
            ]
        }
    },
    {
        "bug_id": 1235,
        "dup_id": null,
        "creation_ts": 1234567898765,
        "stacktrace": {
            "exception": ["com.company.MyException"], 
            "frames": [  
                {"function": "com.company.Class1.method2"}, 
                {"function": "com.company.Class1.main"}
            ]
        }
    }
]
```
where
- `bug_id` – identifier of report
- `dup_id` – identifier of first report in group. If this report is the first in this group, this field should be `null`
- `creation_ts` – timestamp of report creation

`stacktrace` field contains information about report content such as 
exception classes and stack frames.

# Usage
### Install
```
git clone git@github.com:akhvorov/S3M.git
cd S3M
pip install -r requirements.txt
```

### Run
```
cd src
python main.py --data_path path_to_reports.json
```
The script will produce some state files in `event_state` directory in one level with `src`. 
This precomputed states will speed up data reading in the next runs.
