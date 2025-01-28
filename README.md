## Running the task

### Initial set up
Copy `docker-compose.dev.yml` to `docker-compose.yml` and add your OPENAI_KEY

Copy `transcripts_v3` to `src/data`

Copy `depression.csv` to `src/data`


Run ```docker-compose up -d```


#### Question 1
Run 
```bash
docker-compose exec python bash
cd src/sentiment_classification
python main.py
```

To run tests, run 
```bash
docker-compose exec python bash
pytest
```

#### Question 2
Navigate to `localhost:8889`