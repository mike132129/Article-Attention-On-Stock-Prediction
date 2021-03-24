# Article-Attention-On-Stock-Prediction
We devote to find influence of articles about a company on its own stock price

## Setting Up

### 0. Docker
1. Install docker
2. Get mongo image `dokcer pull mongo`
3. create volume `docker volume create AASP_V`, you may use `docker volume ls` to check.
4. Run docker container `docker run -v AASP_V -d -p 27017-27019:27017-27019 --name AASP mongo`
       
       *Only you start the container can you connect it with pymongo client

### 1. Crawl the data
1. Get stock value `python3 stock.py`, the arg will be added.
2. Get stock related news `python3 news.py`, the arg will be added
    
       Note that the data will be stored in mongodb volume, though I have no idea how to get you the complete volume of my own. You still can get data by running `news.py`
