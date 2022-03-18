Deep learning (DL) has been widely applied to many domains. Unique challenges in engineering DL systems are posed by the programming paradigm shift from traditional systems to DL systems, and performance is one of the challenges. Performance problems (PPs) in DL systems can cause severe consequences such as excessive resource consumption and financial loss. While bugs in DL systems have been extensively investigated, PPs in DL systems have hardly been explored. To bridge this gap, we present the first comprehensive study to i) characterize symptoms, root causes, and introducing and exposing stages
of PPs in DL systems developed in TensorFLow and Keras, with **224** PPs collected from **210** StackOverflow posts, and to ii) assess the capability of existing performance analysis approaches in tackling PPs, with a constructed benchmark of 58 PPs in DL systems. Our findings shed light on the implications on developing high-performance DL systems, and detecting and localizing PPs in DL systems. To demonstrate the usefulness of our findings, we develop a static checker DeepPerf to detect three types of PPs. It has detected **488** new PPs in **130** GitHub projects. **105** and **27** PPs have been confirmed and fixed.

## Empirical Study
We present the first comprehensive study to characterize PPs in DL systems developed in TensorFlow and Keras and to assess existing approaches in tackling PPs. We collect **224** PPs from **210** StackOverflow posts, and manually analyze these PPs to answer four research questions:

- RQ1 Symptom: what are the symptoms of PPs?
- RQ2 Root Cause: what are the root causes of PPs?
- RQ3 Stage: what are the stages of introducing and exposing PPs?
- RQ4 Asessment:how is the capability of existing performance analysis approaches in tackling PPs?
  
All **empirical study data** is available [here](https://github.com/DLPerf/DLPerf.github.io/blob/main/empirical_study).

### Keywords Set
Following the procedure presented in the paper, we derived a keyword set that contains 11 1-gram and 13 2-gram keywords. Every post that contain at least one of them will be considered a potential PB post, and should be manually checked later. Words in posts should be stemmed with nltk.stem.PorterStemmer firstly, and then matched with the keywords set. 

Difference between this keywords set and those used in previous empirical studies about performance bugs[1][2]:
1. This keywords set contains 2-gram keywords, which were derived to imrpove the matching accuracy.
2. There are some deep learning specific keywords, such as "gpu util" and "gpu ram".

The full keywords list:

### Codebook

Benchmark and Assessment

We reproduce and build a benchmark of **56** PBs from the **238** PBs in our empirical study with four person-months effort, which can be used to facilitate the future research on PBs in DL systems. We also assess the capability of existing approaches in addressing them.

**The procudure to reproduce benchmark is available [here](https://github.com/DLPerf/DLPerf.github.io/blob/main/benchmark)**.
## Application
To demonstrate the usefulness of our findings, we implement a rule-based static checker, named *DeepPerf*, to detect PBs in DL systems. *DeepPerf* is implemented with two static analysis tools, [AST](https://docs.python.org/3/library/ast.html) and [Jedi](https://github.com/davidhalter/jedi/). It currently supports three types of PBs whose detection~rules are manually derived from our empirical study. 

**The code is available [here](https://github.com/DLPerf/DLPerf.github.io/blob/main/tool).**
