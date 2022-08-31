## DeepPerf

We implement a rule-based static checker, named DeepPerf, to detect PPs in DL systems. DeepPerf is implemented with two static analysis tools, AST and Jedi. It currently supports three types of PPs whose detection rules are manually derived from our empirical study.


### Requirements
- Python >= 3.8.0
- jedi >= 0.18.0

### Checker 1
Run `python graph_detect.py [project_path]` to check all python files in `[project_path]`.
You could specify `[project_path]` as `./test_code/graph` to check the default examples.
### Checker 2 and Checker 3
We have implemented Checker 2 and Checker 3 at the same time. Run `python data_detect.py [project_path]` to check all python files in `[project_path]`.
You could specify `[project_path]` as `./test_code/data` to check the default examples.
