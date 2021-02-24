# S3M

This repository contains the code of our 
[MSR 2021](https://2021.msrconf.org/details/msr-2021-technical-papers/26/S3M-Siamese-Stack-Trace-Similarity-Measure) 
paper "S3M: Siamese Stack (Trace) Similarity Measure".

# Data
The data presented in the following `JSON` format:
```
[
    {
        "bug_id": 1234,
        "dup_id": 12,
        "creation_ts": 1234567891234,
        "stacktrace": {
            "exception": ["java.lang.Exception"], 
            "frames": [
                {"function": "java.util.ArrayList.get", "depth": 0}, 
                {"function": "com.company.method1", "depth": 1}, 
                {"function": "com.company.method2", "depth": 2}, 
                {"function": "com.company.method1", "depth": 3}
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
exception classes and stack frames. Frames should include information
about the depth from the top of the stack.

# Usage
```
cd src
python main.py --data_path path_to_stacktraces.json
```
The script will produce some state files in `event_state` directory in one level with `src`. 
This precomputed states will speed up data reading in the next runs.
