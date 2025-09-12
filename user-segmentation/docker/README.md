# Dockerized PostgreSQL Setup for User Segmentation

This module uses **PostgreSQL 15** inside Docker to store and preprocess 
the Taobao ad dataset.

## 🚀 1. Start PostgreSQL with Docker
```bash
# Pull the image
docker pull postgres:15

# Run the container (name = taobao-db)
docker run --name taobao-db \
    -e POSTGRES_PASSWORD=taobaodb \
    -e POSTGRES_USER=postgres \
    -e POSTGRES_DB=taobao \
    -p 5432:5432 \
    -d postgres:15

----------- 2. Manage the container


# Enter container shell
docker exec -it taobao-db bash

# Exit container without stopping it
exit

# Stop the container
docker stop taobao-db

# Restart the container
docker start taobao-db

------------ 3. Create tables


DROP TABLE IF EXISTS raw_sample;

CREATE TABLE raw_sample (
    user_id BIGINT,
    time_stamp BIGINT,
    adgroup_id INT,
    pid VARCHAR(50),
    nonclk INT,
    clk INT
);

DROP TABLE IF EXISTS user_profile;

CREATE TABLE user_profile (
    user_id BIGINT,
    cms_segid INT,
    cms_group_id INT,
    final_gender_code INT,   -- 1=male, 2=female
    age_level INT,           
    pvalue_level INT,        
    shopping_level INT,      
    occupation INT,          
    new_user_class_level INT 
);

DROP TABLE IF EXISTS ad_feature;

CREATE TABLE ad_feature (
    adgroup_id INT,
    cate_id INT,
    campaign_id INT,
    customer INT,    
    brand INT,
    price FLOAT
);


------------- 4. Load CSV files into the container

docker cp /Users/hilda/Desktop/raw_sample.csv 
taobao-db:/tmp/raw_sample.csv
docker cp /Users/hilda/Desktop/user_profile.csv 
taobao-db:/tmp/user_profile.csv
docker cp /Users/hilda/Desktop/ad_feature.csv 
taobao-db:/tmp/ad_feature.csv

------------- 5. Import CSV into Postgres

COPY raw_sample FROM '/tmp/raw_sample.csv' DELIMITER ',' CSV HEADER;
COPY user_profile FROM '/tmp/user_profile.csv' DELIMITER ',' CSV HEADER;
COPY ad_feature FROM '/tmp/ad_feature.csv' DELIMITER ',' CSV HEADER;


------------- 6. Create materialized view

DROP MATERIALIZED VIEW IF EXISTS user_features;

CREATE MATERIALIZED VIEW user_features AS
SELECT
    r.user_id,
    COUNT(*) AS total_impressions,
    SUM(r.clk) AS total_clicks,
    SUM(r.nonclk) AS total_nonclicks,
    (SUM(r.clk)::float / NULLIF(COUNT(*),0)) AS ctr,
    up.final_gender_code,
    up.age_level,
    up.pvalue_level,
    up.shopping_level,
    up.occupation,
    up.new_user_class_level
FROM raw_sample r
JOIN user_profile up
    ON r.user_id = up.user_id
GROUP BY r.user_id, up.final_gender_code, up.age_level, up.pvalue_level,
         up.shopping_level, up.occupation, up.new_user_class_level;


-------------- 7. Export results

docker exec -t taobao-db \
  psql -U postgres -d taobao \
  -c "\COPY (SELECT * FROM user_features) TO '/tmp/user_features.csv' DELIMITER ',' CSV HEADER;"

docker cp taobao-db:/tmp/user_features.csv /Users/hilda/Desktop/user_features.csv



