# A Multimodal and Sentiment-Based Trading System for Financial Portfolio Optimisation

This is the readme document for the implementation of the proposed Multimodal and Sentiment-based Adaptive framework for portfolio optimisation (MuSA).

## Project Structure

- **./benchmark**: Contain the implementation of comparative portfolio optimization approachs.  
- **./data**: Contain the stock data in three market indexes.
- **./fin_sentiment**: Preprocess the financial text data.
- **./RL_controller**
    - `controllers.py`: Implement the solver-based agent.
    - `market_obs.py`: Implement the market observer.
    - `TD3_controller.py`: Implement the RL-based agent.
- **./utils**
    - `callback_func.py`: Implement the solver-based agent.
    - `featGen.py`: Preprocess the features. 
    - `model_pool.py`: Model zoo.
    - `tradeEnv.py`: Implement the trading environment and performance calculation.
    - `utils.py`: Summarize the trading performance.
- `config.py`: Configure the algorithms and experiments.
- `entrance.py`: Entrance script.
- `requirements.txt`: Required packages.


## Requirements

Please run on Python 3.x, and install the libraries by running the command:
```
python -m pip install -r requirements.txt
```
- The experiments of the MuSA framework and baseline approaches are run on a GPU server machine installed with the AMD Ryzen 9 3900X 12-Core processor running at 3.8 GHz and two Nvidia RTX 3090 GPU cards.

## Entrance Script

You may configure the algorithm and trading settings in ```config.py```. After that, run the below command to start training.
```
python entrance.py
```

## Details of Building Running Environment
If encountering any problems when building the running environment, you may follow the below steps by using docker containers:
1. **Download the pytorch image.**
```
docker pull pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
``` 
2. **Build the docker container**
```
docker run --gpus all -itd -p {host_port}:{container_port} --name musa -v {filepath_in_desktop}:{filepath_in_container} pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
```
- host_port is the host communication port between the host desktop and the container, you may spcify a free port in the desktop. (e.g., 55555)
- container_port is the container communication port. (e.g., 22)
- filepath_in_desktop: the absolute path of musa files in the host desktop.
- filepath_in_container: the absolute mapping path of musa files in the container.


3. **Enter the container** 

```
docker exec -it musa bash
```
- Then locate to the mapping musa file path.

4. **Install relevant python libraries**
```
python -m pip install -r requirements.txt
```
5. **Manually install more libraries**

Except for the listed libraries in the requirements.txt that can be installed via **pip install**, there are a few other libraries that have to be manually installed.

- **TA-LIB** (Installation reference source: https://github.com/TA-Lib/ta-lib-python)
    - Download the source file via https://sourceforge.net/projects/ta-lib/, and move the .tar.gz file to the musa folder in the container.
    - Execute the commands:
        - tar -xzf ta-lib-0.4.0-src.tar.gz   (change the file name when using other versions of TA-LIB)
        - cd ta-lib/
        - ./configure --prefix=/usr
        - make
        - make install
        - pip install TA-LIB

6.**Others**:
- If having the error: "*ImportError: libGL.so.1: cannot open shared object file: No such file or directory*", run the command **apt-get update**, and then **apt-get install libgl1-mesa-glx**.
- If having the error: "*ImportError: libgthread-2.0.so.0: cannot open shared object file: No such file or directory*", run the command **apt-get update**, and then **apt-get install libglib2.0-0**.


Thanks!

MuSA
