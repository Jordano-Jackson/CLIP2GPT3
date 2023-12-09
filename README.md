# CLIP2GPT3
Vision Question Answering using LLaMA with CLIP module

<img width="1422" alt="figure1" src="https://github.com/Jordano-Jackson/CLIP2GPT3/assets/19871043/3af13a01-7cad-49a5-98b1-de5573da710e">

## How to use
* Activate conda environment

  
`conda env create -f py39.yaml`

`conda activate py39`

* To train

`python main.py --load_state_dict params/params.pth --mode train`

* To test
  
`python main.py --load_state_dict params/params.pth --mode test`

  
