-- Drop old tables if they exist
DROP TABLE IF EXISTS raw_sample;
DROP TABLE IF EXISTS user_profile;
DROP TABLE IF EXISTS ad_feature;

-- Raw ad display/click logs
CREATE TABLE raw_sample (
    user_id BIGINT,
    time_stamp BIGINT,
    adgroup_id INT,
    pid VARCHAR(50),
    nonclk INT,
    clk INT
);

-- User profile information
CREATE TABLE user_profile (
    user_id BIGINT,
    cms_segid INT,
    cms_group_id INT,
    final_gender_code INT,   -- 1 = male, 2 = female
    age_level INT,           -- age category
    pvalue_level INT,        -- consumption level
    shopping_level INT,      -- shopping depth
    occupation INT,          -- 1 = college student, 0 = no
    new_user_class_level INT -- city tier
);

-- Ad feature metadata
CREATE TABLE ad_feature (
    adgroup_id INT,
    cate_id INT,
    campaign_id INT,
    customer INT,    -- advertiser ID
    brand INT,
    price FLOAT
);

-- Drop old materialized view if exists
DROP MATERIALIZED VIEW IF EXISTS user_features;

-- Create materialized view with aggregated user features
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

