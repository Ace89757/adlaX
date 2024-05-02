# adlaX :rocket:
本工程基于[OpenMMLab](https://openmmlab.com/codebase)开发，用于个人学习、记录。

:link: [MMEngine 官方文档](https://mmengine.readthedocs.io/zh-cn/latest/)

******

## :hammer: Install

**步骤0. 创建并激活一个 conda 环境**

~~~bash
conda create --name alchemy python=3.8 -y
conda activate alchemy
~~~

**步骤1. 基于 [PyTorch 官方说明](https://pytorch.org/get-started/locally/)安装 PyTorch**

在 GPU 平台上：

```bash
conda install pytorch torchvision -c pytorch
```

在 CPU 平台上：

```bash
conda install pytorch torchvision cpuonly -c pytorch
```

**步骤2. 安装MIM**

~~~bash
pip install -U openmim
~~~

**步骤3. 使用mim安装MMEngine**

~~~Bash
mim install mmengine
~~~

**步骤4. 安装相关依赖包**

~~~bash
pip install future tensorboard
~~~

**步骤5. 将alchemy作为Python包以开发模式安装**

~~~bash
python setup.py develop
~~~