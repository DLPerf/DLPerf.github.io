Deep learning (DL) has been widely applied to many domains. Unique challenges in engineering DL systems are posed by the programming paradigm shift from traditional systems to DL systems, and performance is one of the challenges. Performance problems (PPs) in DL systems can cause severe consequences such as excessive resource consumption and financial loss. While bugs in DL systems have been extensively investigated, PPs in DL systems have hardly been explored. To bridge this gap, we present the first comprehensive study to i) characterize symptoms, root causes, and introducing and exposing stages
of PPs in DL systems developed in TensorFLow and Keras, with **224** PPs collected from **210** StackOverflow posts, and to ii) assess the capability of existing performance analysis approaches in tackling PPs, with a constructed benchmark of 58 PPs in DL systems. Our findings shed light on the implications on developing high-performance DL systems, and detecting and localizing PPs in DL systems. To demonstrate the usefulness of our findings, we develop a static checker DeepPerf to detect three types of PPs. It has detected **488** new PPs in **130** GitHub projects. **105** and **27** PPs have been confirmed and fixed.

## Empirical Study
We present the first comprehensive study to characterize PPs in DL systems developed in TensorFlow and Keras and to assess existing approaches in tackling PPs. We collect **224** PPs from **210** StackOverflow posts, and manually analyze these PPs to answer four research questions:

- RQ1 Symptom: what are the symptoms of PPs?
- RQ2 Root Cause: what are the root causes of PPs?
- RQ3 Stage: what are the stages of introducing and exposing PPs?
- RQ4 Assessment:how is the capability of existing performance analysis approaches in tackling PPs?
  
All **empirical study data** is available [here](https://github.com/DLPerf/DLPerf.github.io/blob/main/empirical_study).

### Keywords Set
Following the procedure presented in the paper, we derived a keyword set that contains 11 1-gram and 13 2-gram keywords to obtain candidate PP posts. Every post that contains at least one of them will be considered a potential PP post, and should be manually checked later. Words in posts should be stemmed with nltk.stem.PorterStemmer firstly, and then matched with the keywords set. 

Difference between this keywords set and those used in previous empirical studies about performance problems:
1. This keywords set contains 2-gram keywords, which were derived to imrpove the matching accuracy.
2. There are some deep learning specific keywords, such as "gpu util" and "gpu ram".

The **full keywords list**: hangs, slowli, slower, slowest, faster, fastest, speed, oom, throughput, effici, overhead, gpu util, extrem slow, take longer, run, slow, very slow, cpu ram, gpu ram, memori leak, cpu time, per second, slowi tri, take long, perform issu.

### Codebook
The actionable code of the final taxonomies for symptoms, root causes and stages is presented below.

#### Code for Symptoms
1. **Time:** The buggy code version exhibits high time cost, and the fixed code version imrpoves the time cost. 
  - Slow Execution Time: PPs manifest slow execution time during the execution of DL systems, including data preparation, model building, training, evaluation, hyper parameter tuning, or prediction.
  - Slow Initialization Time: PBs manifest slow initialization time before the actual execution of the program, including  library importing or setting up execution environment. 
  - Increasing Time Over Time: PBs manifest longer and longer time as the program ran, including across different steps or epochs during the training and different samples during the prediction. 
  - Program Hang: PBs make DL systems run forever, or too slow for questioners in the post to measure their time.
2. **Memory:** The buggy code version exhibits high memory usage, and the fixed code version improves the memory usage.
  - Out of Memory: PBs make DL systems terminate with Out of Memory error.
  - Memory Leak: PBs manifest more and more memory usage as the program ran, and may cause Out of Memory in the end.
  - Abnormal GPU Memory Usage: PBs manifest an abnormal low or high GPU Memory usage, which indicate the GPU may not be efficiently utilized or wasted. 
3. **Processor:** The buggy code version exhibits abnormal processor utilization, and the fixed code version fixes it.
  - Abnormal GPU Utilization: PBs manifest an abnromal low or high GPU utilization during the program execution, which indicate the GPU may not be efficiently utilized or wasted.
  - Not Using GPU: PBs make DL systems not utilize the GPU at all.
  - Abnormal CPU Utilization: PBs manifest an abnromal low or high CPU utilization during the program execution, which indicate the CPU may not be efficiently utilized or wasted.
4. **Unkown:** If the symptom of a PB was not determined if the questioner explicitly reported the symptom in the post,  we conservatively marked it as "Unknown".
#### Code for Root Causes
1. **API:** PBs are caused by improper TensorFLow API usages.
  - Not Using Efficient API: Instead of using the highly optimized TensorFlow or Keras APIs, programmers may perform the computation with manually written code or sub-optimal APIs.
  - Not Using Batch API: Instead of using the batch processing APIs, programmers may load all data at once (causing OOM), or perform the computation one by one in a loop (causing long execution time).
  - Inefficient API Usage: Programmers choose the correct APIs, but use them wrongly, including sub-optimal API parameters or API call orders.
2. **Model:** PBs are caused by inefficient DL models, which means the fixing patch will change the buggy model (computation graph) structure or parameters.
  - Confusion with Computation Graph: The computation graphs are not constucted as the programmers expected, such as repeatedly created the same computation graph nodes, because programmers lack of knowledge about TensorFlow computation graph programming model.
  - Inefficient Model Structure: PBs are caused by sub-optimal model structure, e.g. redundant computation or improper activation function.
  - Improper Model Parameter: PBs are caused by improper model parameter, e.g. parameters of each layer and parameters of each oepration.
  - Improper Hyper Parameter: PBs are caused by improper hyper parameter, e.g. batch size and learning rate.
3. **Library:** PBs are caused by inefficient library versions.
  - Buggy Library Version: There are PBs in the underlying libraries (e.g. TensorFlow, Numpy, ...) for some specific versions.
  - Mismatched Library Version: The version restrictions of different libraries (e.g. TensorFlow, CUDA, Numpy, ...) are not satisfied for full GPU usage.
4. **Data:** PBs with root causes related to data processing.
  - Inefficient Data Transmission: PBs are caused by inefficient data transmission, e.g. loading input data over the network or copying weights from CPU to GPU.
  - Inefficient Data Preprocessing: PBs are caused by inefficient data preprocessing, e.g. lack of image normalization before changing an image to a tensor.
  - Improper Data Input: PBs are caused by improper input data, e.g. improper data format or size that consumes excessive resources.
5. **Hardware:** PBs with root causes related to hardware issues.
  - Hardware and Library Mismatch: The GPU/TPU could not be fully utilized with the combination of hardware and library version.
  - Hardware and Code Mismatch: The GPU/TPU could not be fully utilized with the combination of hardware and program code (including model).
  - Improper Configuration: PBs are caused by the misconfiguration of hardware related parameters.
#### Code for Stages
1. **Environment Setting:** The stage where DL environment like libraries and hardware are properly installed and configured.
2. **Initialization:** The stage where DL system is initialized (e.g. importing libraries and initializing parameters) before the ececution stages.
3. **Data Preparation:** The stage involving data cleaning, data preprocessing and data loading for later stages.
4. **Model Building:** The stage where a proper model is chosen and constucted with DL framework APIs (TensorFlow computation graph APIs or Keras APIs in our study).
5. **Training:** The stage where the model is refined iteratively with training data.
6. **Evaluation:** The stage where the model is evaluated with validation data.
7. **Hyper Parameter Tuning:** The stage where hyper parameters are tuned to improve the accuracy of the model.
8. **Prediction:** The stage where the model is deployed and used to predict the real world data.
9. **Unkown:** The stage could not be inferred directly from the post due to the incomplete code snippets.

#### Disambiguation guide
We provide further disambiguation guide for codes that may be misclassified with each other, such as Inefficient API Usage and Not Using Efficient API, Confusion with Computation Graph and Inefficient Model Structure, etc. It is available [here](https://github.com/DLPerf/DLPerf.github.io/blob/main/empirical_study).

### Benchmark and Assessment

We reproduce and build a benchmark of **58** PPs from the **224** PPs in our empirical study with four person-months effort, which can be used to facilitate the future research on PBs in DL systems. We also assess the capability of existing approaches in addressing them.

**The procudure to reproduce benchmark is available [here](https://github.com/DLPerf/DLPerf.github.io/blob/main/benchmark)**.
## Application
To demonstrate the usefulness of our findings, we implement a rule-based static checker, named *DeepPerf*, to detect PBs in DL systems. *DeepPerf* is implemented with two static analysis tools, [AST](https://docs.python.org/3/library/ast.html) and [Jedi](https://github.com/davidhalter/jedi/). It currently supports three types of PBs whose detection~rules are manually derived from our empirical study. 

**The code is available [here](https://github.com/DLPerf/DLPerf.github.io/blob/main/tool).**

## Reference
\[1] Guoliang Jin, Linhai Song, Xiaoming Shi, Joel Scherpelz, and Shan Lu. 2012. Understanding and Detecting Real-World Performance Bugs. In Proceedings of the 33rd ACM SIGPLAN Conference on Programming Language Design and Implementation. 77–88.
\[2] Shahed Zaman, Bram Adams, and Ahmed E Hassan. 2012. A qualitative study on performance bugs. In Proceedings of the 9th IEEE working conference on mining software repositories. 199–208.
