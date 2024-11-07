# Installation of Megatron-LM (Experimental)

## Update 7/29/2024

**Don't use cuda 12.1. Either use 11.8 or 12.4.**

Don't use syntax like `conda install anaconda::cudnn`. Use `conda install -c anaconda cudnn`.


## Version 12/2/2023

**This guide is an alternative to the official Megatron-LM readme, which recommends using the NGC pytorch container.**
**Wherever possible, use the pytorch container. Refer to your nvidia driver for choosing the right version.**

Create environment:

conda create -n l2 python=3.10
conda activate l2


Use g++/gcc to compile. Just use the latest version.


```bash
conda install -c conda-forge cxx-compiler
```


## Install Cuda and CUDNN (Ignore if installed)

look for entries like "cudnn", "cuda-toolkit" in the list of packages. Packages starting with "nv-" are ignored.
Choose a cuda version supported by your nvidia driver.

`conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit`
`conda install -c conda-forge cudnn`    (This is for TransformerEngine)
or
`conda install -c anaconda cudnn`

Make sure to install the right version of cudnn. 

## Install PyTorch for Cuda 12.1 

You can choose from stable or (nightly) version. Choose YOUR cuda version.

pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118

## Install Apex

pip install packaging
pip install ninja
https://github.com/NVIDIA/apex#linux

```bash
git clone https://github.com/NVIDIA/apex
cd apex
git checkout 2386a912164b0c5cfcd8be7a2b890fbac5607c82
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

Some issue:

 https://github.com/pybind/pybind11/issues/4606
```
.../include/pybind11/detail/../cast.h:45:120: error: expected template-name before ‘<’ token
   45 |     return caster.operator typename make_caster<T>::template cast_op_type<T>();
      |                                                                                                    
```

to solve this: open pybind11/cast.h (should be in pytorch directory) 

```diff
-    return caster.operator typename make_caster<T>::template cast_op_type<T>();
+    return caster;
```

or:
`python -m pip install pybind11`
and see

## Install Flash-Attn & TransformerEngine


```
# pip install flash-attn==2.0.4 --no-build-isolation
pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable
```



## Install Pytorch for CUDA 11.8
`conda install -c "nvidia/label/cuda-11.8.0" cuda`
`conda install -c anaconda cudnn`
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Install other packages

six
regex
sentencepiece
datasets
nltk

