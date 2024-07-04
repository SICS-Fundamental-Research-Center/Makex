# DAG-Explainer

This is an adapted version for using existing work DAG for the link prediction tasks.

## Installation
* Clone the repository 
* Create the env and install the requirements

```shell script
$ source ./install.sh
```

## Usage
* Download the required [datasets](https://hkustconnect-my.sharepoint.com/:f:/g/personal/glvab_connect_ust_hk/EqFR8NjD49tLtPp9TgicvjQBxkj_15wDT4D2fdrJ6Adx2A?e=P9NeHI) to `/data`
  > The candidates were generated using [gSpan](https://github.com/betterenvi/gSpan).
* Please get the GNN scores of your own models before run this script and put the GNN scores to `/data`
* Run the searching scripts with corresponding dataset.
```shell script
$ source ./scripts.sh
``` 
The hyper-parameters used for different datasets are shown in this script.

## Time Cost Collection
In this implement, the processing time cost does not include the time cost for run gSpan and the time cost for geting the GNN scores of each subgraph. 
Please add these two time cost after running them, as we did in our experiments.
```shell script
$ Total time cost of DAG  = T(gspan) + T(GNN) + T(pipline)
```
where T denote the time cost
