# Landmark Complex Exploration via Deep Reinforcement Learning

This project aims to achieve multi-agent cooperative exploration in sparse landmark complex environments. Landmark complex is an abstract simplicial complex proposed 
for assisting multi-agent exploration. Although traditional landmark complex exploration methods are frontier-based, our method uses deep reinforcement learning to learn 
a policy for exploration and a three-stage curriculum to help mitigate reward sparsity. This Github repository includes all necessary codes to reproduce our result in [Multi-Agent Exploration of an Unknown Sparse Landmark Complex via
Deep Reinforcement Learning]().

<p alilgn="center">
  <img src="https://user-images.githubusercontent.com/48639163/185293301-3ecce675-d87b-45a2-9ccd-f6118a5c4dfd.gif" />
</p>


## Requirements
- Unity 2020.3.35
- ML-agents Release 19
- Python 3.8
- Docker

## Usage
- First navigate to the Python directory and build the docker image

```sh
docker build -t lcserver .
```

- Run the Docker container
```sh
docker run -d --name lcserver -p 80:80 lcserver
```

- Open the root directory with Unity Editor 2020.3.35

- In the Unity Editor, navigate to ***Assets/LandmarkComplex/Scenes*** and open ***LandmarkComplex_Experiment***

- Press the Play button, and the exploration will begin.

- By pressing the Play button again, the exploration will be terminated.

- To get the exploration data, export the container

```sh
docker export container_id > res.tar
```

- The experiment data can be found at app/data_output/DRL/out.csv

