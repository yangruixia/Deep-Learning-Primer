

# 1. 服务器要求

Linux系统即可，有无GPU均可进行conda环境配置。

# 2. Miniconda环境配置

Miniconda是一个轻量级的Anaconda发行版，它是Python的包管理器和环境管理器。它提供了一个简单和快速的方法来安装和管理Python包及其依赖项，同时允许用户创建和管理独立的Python环境。

## 2.1 服务器配置流程

### 下载Miniconda包

去[Miniconda官网](https://docs.conda.io/en/latest/miniconda.html)选择对应版本的Miniconda并复制链接。

![image](https://github.com/yangruixia/Deep-Learning-Primer/assets/32283868/55baf977-899e-4c2d-aedb-7fc46e35a292)

```
wget <miniconda官网连接>
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

### 运行下载的sh文件

```
sh Miniconda的sh文件
sh Miniconda3-latest-Linux-x86_64.sh
```

### 安装完成

## 2.2 环境配置流程

### 创建环境
```
conda create --name myenv
conda create -n <envname>
```
如果要指定Python版本，需在后面添加参数。
```
conda create --name myenv python=3.10
conda create -n <envname> python=3.9
```
### 检查Python版本
```
python –-version
```
### 激活环境
```
conda activate myenv
```
### 查询环境列表
```
conda info --envs
conda env list
```
### 退出环境
```
conda deactivate
```
### 清除环境
```
conda env remove --name <envname>
```
### 激活环境后查看包列表
```
pip list
```
### 激活环境后安装包
```
pip install <package name>
```
### 查看GPU使用情况
如果存在GPU则会显示GPU情况。
```
nvidia-smi
```

# 3. 校内服务器使用
- 校内服务器仅限北语校园网使用。
- 校内集群服务器使用[官方网址](https://hpc.litmind.ink/docs/#/)。
## 3.1 配置环境
```
module add anaconda3
conda create -n envname python==3.10
```

## 3.2 校内的6块GPU
- 其中1-4的运算速度相对较慢，但是内存比较大，适合跑一些数据量比较大的项目。
- 5-6的运算速度比较快，但是内存比较小，适合跑一些比较小的项目。

## 3.3 查看队列作业信息
squeue用于查看当前slurm的调度队列中的作业信息。

## 3.4 salloc申请GPU
申请2块GPU和1个节点。
```
salloc -G 2 -N 1
```
## 3.5 查看GPU使用情况
```
ssh compute1
nvidia-smi
```


