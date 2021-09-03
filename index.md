Deep learning (DL) has been increasingly applied to a variety of domains. The programming paradigm shift from traditional systems to DL systems poses unique challenges in engineering DL systems. Performance is one of the challenges, and performance bugs (PBs) in DL systems can cause severe consequences such as excessive resource consumption and financial loss. While bugs in DL systems have been extensively investigated, PBs in DL systems have never been explored. To bridge this gap, we present the first comprehensive study to characterize symptoms, root causes, and introducing and exposing stages of PBs in DL systems developed in TensorFLow and Keras, using a total of **238** PBs collected from **225** StackOverflow posts. Our findings shed light on the implications on developing high-performance DL systems, and detecting and localizing PBs in DL systems. We also build the first benchmark of **56** PBs in DL systems, and assess the capability of existing approaches in tackling them. Moreover, we develop a static checker *DeepPerf* to detect three types of PBs, and identify **488** new PBs in **130** GitHub projects. **55** and **18** of them have been respectively confirmed and fixed by developers.

## Empirical Study
We present the first comprehensive study to characterize PBs in DL systems developed in TensorFlow and Keras. We collect **238** PBs from **225** StackOverflow posts, and manually analyze these PBs to answer three research questions:

- RQ1 Symptom: what are the symptoms of PBs?
- RQ2 Root Cause: what are the root causes of PBs?
- RQ3 Stage: what are the stages of introducing and exposing PBs?
  
**Empirical study data is available [here](https://github.com/DLPerf/DLPerf.github.io/blob/main/empirical_study)**.


## Benchmark and Assessment
We reproduce and build a benchmark of **56** PBs from the **238** PBs in our empirical study with four person-months effort, which can be used to facilitate the future research on PBs in DL systems. We also assess the capability of existing approaches in addressing them.

**The procudure to reproduce benchmark is available [here](https://github.com/DLPerf/DLPerf.github.io/blob/main/benchmark)**.
## Detection
To demonstrate the usefulness of our findings, we implement a rule-based static checker, named *DeepPerf*, to detect PBs in DL systems. *DeepPerf* is implemented with two static analysis tools, [AST](https://docs.python.org/3/library/ast.html) and [Jedi](https://github.com/davidhalter/jedi/). It currently supports three types of PBs whose detection~rules are manually derived from our empirical study. 

**The code is available [here](https://github.com/DLPerf/DLPerf.github.io/blob/main/tool).**
